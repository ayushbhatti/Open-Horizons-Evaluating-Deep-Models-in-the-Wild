#!/usr/bin/env python3
"""
Training script for CIFAR-10 FSCIL with Baseline, OrCo and CONCM strategies.
"""

import os
import random
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import (
    ResNet50Encoder,
    LinearClassifier,
    unfreeze_last_blocks,
    supervised_contrastive_loss,
    orthogonality_loss,
    generate_orthogonal_targets,
    target_contrastive_loss,
)
from test import (
    compute_prototypes,
    evaluate_proto_bank,
    few_shot_indices,
    concm_calibrate_torch,
)
from concm_full import ConCMFramework

# ---------------- Config ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
SAVE_DIR = "../../checkpoints_cifar_resnet50"
os.makedirs(SAVE_DIR, exist_ok=True)

BASE_CLASSES = [0, 1, 2, 3, 4, 5, 6]
NEW_CLASSES = [7, 8, 9]
NUM_TOTAL_CLASSES = len(BASE_CLASSES) + len(NEW_CLASSES)
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

BATCH_SIZE = 128
EPOCHS = 30
LR_ADAM = 1e-3
LR_SGD = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

SUPCON_WEIGHT = 0.3
ORTHO_WEIGHT = 0.01
SUPCON_TEMP = 0.07
CONCM_ALPHA = 0.5
TARGET_WEIGHT = 0.5
LAMBDA_PERT = 1e-2
ORCO_BLEND = 0.5
ORCO_PSEUDO_TARGETS = None
CONCM_FULL_FRAMEWORK = None


# ---------------- Utilities ----------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ---------------- Training ----------------
def train_base(
    encoder, classifier, loader, strategy, exp_id, freeze_encoder=False, tag=""
):
    global ORCO_PSEUDO_TARGETS, CONCM_FULL_FRAMEWORK
    model_path = os.path.join(SAVE_DIR, f"{strategy}_{tag}_best.pt")

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        print("🔒 Encoder frozen — training classifier head only.")

    encoder, classifier = encoder.to(DEVICE), classifier.to(DEVICE)
    ce = nn.CrossEntropyLoss().to(DEVICE)

    # Initialize CONCM_FULL framework if needed
    concm_framework = None
    if strategy == "CONCM_FULL":
        print("🔧 Initializing full ConCM framework...")
        concm_framework = ConCMFramework(
            feat_dim=encoder.out_dim, attr_dim=128, geom_dim=128
        ).to(DEVICE)
        CONCM_FULL_FRAMEWORK = concm_framework

    params = [
        p
        for p in list(encoder.parameters()) + list(classifier.parameters())
        if p.requires_grad
    ]

    # Add CONCM_FULL parameters
    if concm_framework is not None and not freeze_encoder:
        params += list(concm_framework.parameters())

    opt = (
        torch.optim.Adam(params, lr=LR_ADAM)
        if exp_id == 1
        else torch.optim.SGD(
            params, lr=LR_SGD, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
        )
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    base_label_map = {c: i for i, c in enumerate(BASE_CLASSES)}
    best_acc = 0.0

    if strategy == "ORCO":
        print("🔧 Generating OrCo pseudo-targets...")
        ORCO_PSEUDO_TARGETS = generate_orthogonal_targets(
            NUM_TOTAL_CLASSES, encoder.out_dim, device=DEVICE
        ).to(DEVICE)
        base_targets = ORCO_PSEUDO_TARGETS[: len(BASE_CLASSES)]
        all_targets = ORCO_PSEUDO_TARGETS
    else:
        base_targets, all_targets = None, None

    for e in range(EPOCHS):
        running, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            feats = encoder(x)
            logits = classifier(feats)
            ce_labels = torch.tensor(
                [base_label_map[int(lbl)] for lbl in y], dtype=torch.long, device=DEVICE
            )
            loss = ce(logits, ce_labels)

            if strategy == "ORCO" and not freeze_encoder:
                loss += SUPCON_WEIGHT * supervised_contrastive_loss(
                    feats, y, SUPCON_TEMP
                )
                loss += ORTHO_WEIGHT * orthogonality_loss(feats, y)
                loss += TARGET_WEIGHT * target_contrastive_loss(
                    feats,
                    ce_labels,
                    base_targets,
                    all_targets,
                    LAMBDA_PERT,
                    SUPCON_TEMP,
                )

            elif (
                strategy == "CONCM_FULL"
                and concm_framework is not None
                and not freeze_encoder
            ):
                # Add DSM alignment loss
                dsm_loss = concm_framework.dsm.align_features(
                    feats, y, temperature=SUPCON_TEMP
                )
                loss += 0.5 * dsm_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            preds = logits.argmax(1)
            correct += (preds == ce_labels).sum().item()
            total += ce_labels.size(0)
        sched.step()
        acc = 100.0 * correct / total

        # Initialize CONCM_FULL base memory after first epoch
        if strategy == "CONCM_FULL" and concm_framework is not None and e == 0:
            print("   Initializing ConCM base memory...")
            with torch.no_grad():
                all_feats, all_labels = [], []
                for x, y in loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    feats = encoder(x)
                    all_feats.append(feats)
                    all_labels.append(y)
                all_feats = torch.cat(all_feats, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                concm_framework.train_base_session(encoder, all_feats, all_labels)

        if acc > best_acc:
            best_acc = acc
            checkpoint = {
                "encoder": encoder.state_dict(),
                "classifier": classifier.state_dict(),
                "epoch": e + 1,
                "accuracy": acc,
                "strategy": strategy,
                "experiment": tag,
            }
            if concm_framework is not None:
                checkpoint["concm_framework"] = concm_framework.state_dict()
            torch.save(checkpoint, model_path)
        print(
            f"[{strategy}] Epoch {e+1:02d}/{EPOCHS} | Loss {running/len(loader):.4f} | Acc {acc:.2f}%"
        )
    print(f"✅ Best Training Accuracy ({strategy}, {tag}): {best_acc:.2f}%")
    return best_acc, model_path


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 FSCIL Models")
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["all", "BASELINE", "ORCO", "CONCM", "CONCM_FULL"],
        help="Strategy to train (default: all)",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Experiment to run (1=pretrained_frozen, 2=scratch, 3=partial_unfreeze). If not specified, runs all applicable experiments",
    )
    args = parser.parse_args()

    # Determine which experiments to run
    if args.experiment is not None:
        experiments = [args.experiment]
    else:
        experiments = [1, 2, 3]

    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Strategy: {args.strategy}")
    print(f"Experiments: {experiments}")
    print(f"Device: {DEVICE}")
    print(f"{'='*70}\n")

    for exp_id in experiments:
        if exp_id == 1:
            pretrained_flag, freeze_encoder, tag = True, True, "exp1_pretrained_frozen"
        elif exp_id == 2:
            pretrained_flag, freeze_encoder, tag = (
                False,
                False,
                "exp2_scratch_fulltrain",
            )
        else:
            pretrained_flag, freeze_encoder, tag = (
                True,
                False,
                "exp3_partial_unfreeze_better",
            )

        print(f"\n🧪 Experiment {exp_id}: {tag}\n")

        # Setup data transforms
        mean, std = (
            ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            if pretrained_flag
            else ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        )
        transform_train = transforms.Compose(
            [
                transforms.Resize(224) if pretrained_flag else transforms.Resize(32),
                (
                    transforms.RandomResizedCrop(224, (0.8, 1.0))
                    if pretrained_flag
                    else transforms.RandomCrop(32, padding=4)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(224) if pretrained_flag else transforms.Resize(32),
                (
                    transforms.CenterCrop(224)
                    if pretrained_flag
                    else transforms.CenterCrop(32)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        # Load CIFAR-10 dataset
        print(
            f"📂 Loading CIFAR-10 with {'ImageNet' if pretrained_flag else 'CIFAR-10'} normalization..."
        )
        trainset = datasets.CIFAR10(
            "../../data", train=True, download=True, transform=transform_train
        )
        testset = datasets.CIFAR10(
            "../../data", train=False, download=True, transform=transform_test
        )

        base_idx = [i for i, (_, y) in enumerate(trainset) if int(y) in BASE_CLASSES]
        base_train = Subset(trainset, base_idx)
        base_loader = DataLoader(
            base_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        print(
            f"✓ Base training set: {len(base_train)} samples from {len(BASE_CLASSES)} classes"
        )
        print(f"✓ Base classes: {[CLASS_NAMES[i] for i in BASE_CLASSES]}")

        # Determine which strategies to run for this experiment
        if args.strategy == "all":
            # BASELINE only for exp2 (scratch training - simplest case)
            if exp_id == 2:
                strategies = ["BASELINE", "ORCO", "CONCM", "CONCM_FULL"]
            else:
                strategies = ["ORCO", "CONCM", "CONCM_FULL"]
        else:
            # Run only the specified strategy
            strategies = [args.strategy]

        for strategy in strategies:
            # Skip BASELINE for pretrained experiments
            if strategy == "BASELINE" and exp_id != 2:
                print(
                    f"\n⚠️  Skipping {strategy} for {tag} (BASELINE only runs for Exp2)"
                )
                continue

            print(f"\n{'='*70}")
            print(f"{strategy} STRATEGY - {tag.upper()}")
            print(f"{'='*70}")

            encoder = ResNet50Encoder(pretrained=pretrained_flag).to(DEVICE)
            classifier = LinearClassifier(encoder.out_dim, len(BASE_CLASSES)).to(DEVICE)
            if exp_id == 3:
                encoder = unfreeze_last_blocks(encoder)
            elif freeze_encoder:
                for p in encoder.parameters():
                    p.requires_grad = False

            model_path = os.path.join(SAVE_DIR, f"{strategy}_{tag}_best.pt")
            if not os.path.exists(model_path):
                print(f"🔨 Training {strategy} model from scratch...")
                train_base(
                    encoder,
                    classifier,
                    base_loader,
                    strategy,
                    exp_id,
                    freeze_encoder,
                    tag,
                )
            else:
                print(f"📥 Loading existing checkpoint: {model_path}")
                ckpt = torch.load(model_path, map_location=DEVICE)
                encoder.load_state_dict(ckpt["encoder"])
                classifier.load_state_dict(ckpt["classifier"])

                # Load CONCM_FULL framework if exists
                if strategy == "CONCM_FULL" and "concm_framework" in ckpt:
                    global CONCM_FULL_FRAMEWORK
                    CONCM_FULL_FRAMEWORK = ConCMFramework(
                        feat_dim=encoder.out_dim, attr_dim=128, geom_dim=128
                    ).to(DEVICE)
                    CONCM_FULL_FRAMEWORK.load_state_dict(ckpt["concm_framework"])
                    print("✓ Loaded ConCM framework")

                print(
                    f"✓ Loaded checkpoint (epoch {ckpt.get('epoch', 'N/A')}, acc {ckpt.get('accuracy', 0.0):.2f}%)"
                )

            # Evaluate base prototypes
            print(f"\n📊 Evaluating base model performance...")
            _, acc_base = evaluate_proto_bank(
                encoder,
                compute_prototypes(encoder, base_train, BASE_CLASSES, device=DEVICE),
                testset,
                BASE_CLASSES,
                device=DEVICE,
            )
            print(f"[{strategy} ({tag})] Base Classes Accuracy (0-6): {acc_base:.2f}%")

            # Skip incremental evaluation for baseline (it only knows base classes)
            if strategy == "BASELINE":
                print(
                    f"\n📌 {strategy} training complete. Model only knows base classes."
                )
                print(
                    f"   Use test.py with --strategy BASELINE to evaluate on all 10 classes."
                )
                continue

            # Incremental evaluation with different k-shot settings
            for kshot in [1, 5, 10]:
                print(f"\n{'─'*70}")
                print(f"📈 {strategy} ({tag}): {kshot}-SHOT INCREMENTAL EVALUATION")
                print(f"{'─'*70}")

                proto_bank = OrderedDict(
                    compute_prototypes(encoder, base_train, BASE_CLASSES, device=DEVICE)
                )
                seen_classes, novel_classes = BASE_CLASSES.copy(), []

                for nc in NEW_CLASSES:
                    print(
                        f"\n➕ Adding class {nc} ({CLASS_NAMES[nc]}) with {kshot} samples..."
                    )
                    kidx = few_shot_indices(trainset, nc, kshot)
                    novel_set = Subset(trainset, kidx)
                    novel_proto = compute_prototypes(
                        encoder, novel_set, [nc], device=DEVICE
                    )[nc]

                    # Apply strategy-specific calibration
                    if strategy == "CONCM":
                        novel_proto = concm_calibrate_torch(
                            novel_proto, proto_bank, CONCM_ALPHA
                        )
                        print(f"   ✓ Applied CONCM calibration (α={CONCM_ALPHA})")
                    elif strategy == "CONCM_FULL" and CONCM_FULL_FRAMEWORK is not None:
                        novel_proto = CONCM_FULL_FRAMEWORK.calibrate_novel_class(
                            novel_proto, nc, alpha=0.6
                        )
                        print(f"   ✓ Applied full ConCM calibration (MPC + DSM)")
                    elif strategy == "ORCO" and ORCO_PSEUDO_TARGETS is not None:
                        t = F.normalize(ORCO_PSEUDO_TARGETS[nc].unsqueeze(0), dim=1)
                        p = F.normalize(novel_proto, dim=1)
                        novel_proto = F.normalize(
                            ORCO_BLEND * p + (1 - ORCO_BLEND) * t, dim=1
                        )
                        print(f"   ✓ Applied ORCO blending (β={ORCO_BLEND})")

                    proto_bank[nc] = novel_proto
                    novel_classes.append(nc)
                    seen_classes.append(nc)

                    # Evaluate after adding each class
                    all_seen_total, all_seen_acc = evaluate_proto_bank(
                        encoder, proto_bank, testset, seen_classes, device=DEVICE
                    )
                    novel_total, novel_acc = evaluate_proto_bank(
                        encoder, proto_bank, testset, novel_classes, device=DEVICE
                    )
                    print(
                        f"   📊 Results: All Seen={all_seen_acc:.2f}% ({all_seen_total} samples) | "
                        f"Novel Only={novel_acc:.2f}% ({novel_total} samples)"
                    )

                print(f"\n{'─'*70}")
                print(f"✅ Completed {kshot}-shot evaluation for {strategy}")
                print(f"{'─'*70}")


if __name__ == "__main__":
    main()
