#!/usr/bin/env python3
"""
Testing script for CIFAR-10 FSCIL with Baseline, OrCo and CONCM strategies.
Loads trained models and evaluates incremental learning performance.
"""

import os
import random
import argparse
from collections import OrderedDict, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from model import ResNet50Encoder, LinearClassifier
from concm_full import ConCMFramework

# ---------------- Config ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BASE_CLASSES = [0, 1, 2, 3, 4, 5, 6]
NEW_CLASSES = [7, 8, 9]
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

CONCM_ALPHA = 0.5
ORCO_BLEND = 0.5


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


def load_fixed_samples(csv_path):
    """Load fixed test sample indices from CSV file using pandas"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fixed samples file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    samples_by_class = df.groupby("class_id")["sample_idx"].apply(list).to_dict()
    return samples_by_class


def few_shot_indices(dataset, target_class, k_shot):
    """Get k-shot indices for a target class"""
    idxs = [i for i, (_, y) in enumerate(dataset) if int(y) == target_class]
    random.shuffle(idxs)
    return idxs[:k_shot]


@torch.no_grad()
def compute_prototypes(encoder, dataset, class_list, batch_size=256, device="cuda"):
    """Compute prototypes for given classes"""
    encoder.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    feat_sums, counts = defaultdict(lambda: 0), defaultdict(int)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        feats = encoder(x)
        for i in range(x.size(0)):
            c = int(y[i])
            if c in class_list:
                feat_sums[c] = (
                    feats[i]
                    if isinstance(feat_sums[c], int)
                    else feat_sums[c] + feats[i]
                )
                counts[c] += 1
    protos = OrderedDict()
    for c in class_list:
        if counts[c] > 0:
            protos[c] = F.normalize((feat_sums[c] / counts[c]).unsqueeze(0), dim=1)
    return protos


def concm_calibrate_torch(new_proto, base_proto_dict, alpha=0.5):
    """CONCM calibration: blend new prototype with mean of base prototypes"""
    base_stack = torch.cat(list(base_proto_dict.values()), dim=0)
    mean_base = base_stack.mean(0, keepdim=True)
    return F.normalize(alpha * new_proto + (1 - alpha) * mean_base, dim=1)


@torch.no_grad()
def evaluate_proto_bank(
    encoder, proto_bank, dataset, allowed_classes, device="cuda", fixed_indices=None
):
    """Evaluate using prototype-based classification

    Args:
        encoder: Feature encoder model
        proto_bank: Dictionary of class prototypes
        dataset: Test dataset
        allowed_classes: List of classes to evaluate
        device: Device to use
        fixed_indices: Optional dict mapping class_id -> list of sample indices to use
    """
    encoder.eval()
    if len(proto_bank) == 0:
        return 0, 0.0

    proto_keys = [c for c in proto_bank if c in allowed_classes]

    # If no prototypes for allowed classes (e.g., BASELINE evaluating new classes),
    # use all available prototypes (model will misclassify as known classes)
    if len(proto_keys) == 0:
        proto_keys = list(proto_bank.keys())

    proto_mat = F.normalize(
        torch.cat([proto_bank[c].to(device) for c in proto_keys], 0), dim=1
    )
    proto_labels = torch.tensor(proto_keys, device=device)

    # If fixed indices provided, create subset
    if fixed_indices is not None:
        # Collect all indices for allowed classes
        eval_indices = []
        for cls in allowed_classes:
            if cls in fixed_indices:
                eval_indices.extend(fixed_indices[cls])
        eval_dataset = Subset(dataset, eval_indices)
    else:
        eval_dataset = dataset

    loader = DataLoader(
        eval_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
    )
    total, correct = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mask = torch.tensor(
            [int(lbl) in allowed_classes for lbl in y], dtype=torch.bool, device=device
        )
        if mask.sum() == 0:
            continue
        x, y = x[mask], y[mask]
        f = F.normalize(encoder(x), dim=1)
        sims = torch.matmul(f, proto_mat.t())
        preds = proto_labels[sims.argmax(1)]
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = 100.0 * correct / total if total > 0 else 0.0
    return total, acc


@torch.no_grad()
def compute_confusion_matrix(
    encoder, proto_bank, dataset, allowed_classes, device="cuda", fixed_indices=None
):
    """Compute confusion matrix for prototype-based classification

    Args:
        encoder: Feature encoder model
        proto_bank: Dictionary of class prototypes
        dataset: Test dataset
        allowed_classes: List of classes to evaluate
        device: Device to use
        fixed_indices: Optional dict mapping class_id -> list of sample indices to use
    """
    encoder.eval()
    if len(proto_bank) == 0:
        return None, None

    proto_keys = [c for c in proto_bank if c in allowed_classes]

    # If no prototypes for allowed classes (e.g., BASELINE evaluating new classes),
    # use all available prototypes (model will misclassify as known classes)
    if len(proto_keys) == 0:
        proto_keys = list(proto_bank.keys())

    proto_mat = F.normalize(
        torch.cat([proto_bank[c].to(device) for c in proto_keys], 0), dim=1
    )
    proto_labels = torch.tensor(proto_keys, device=device)

    # If fixed indices provided, create subset
    if fixed_indices is not None:
        # Collect all indices for allowed classes
        eval_indices = []
        for cls in allowed_classes:
            if cls in fixed_indices:
                eval_indices.extend(fixed_indices[cls])
        eval_dataset = Subset(dataset, eval_indices)
    else:
        eval_dataset = dataset

    loader = DataLoader(
        eval_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
    )

    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mask = torch.tensor(
            [int(lbl) in allowed_classes for lbl in y], dtype=torch.bool, device=device
        )
        if mask.sum() == 0:
            continue
        x, y = x[mask], y[mask]
        f = F.normalize(encoder(x), dim=1)
        sims = torch.matmul(f, proto_mat.t())
        preds = proto_labels[sims.argmax(1)]

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    if len(all_labels) == 0:
        return None, None

    cm = confusion_matrix(all_labels, all_preds, labels=allowed_classes)
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12) * 100

    return cm, cm_normalized


def plot_confusion_matrix(cm_normalized, class_list, strategy, kshot, output_dir):
    """Plot and save confusion matrix"""
    display_names = [CLASS_NAMES[i] for i in class_list]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=display_names,
        yticklabels=display_names,
        cbar_kws={"label": "Percentage (%)"},
        linewidths=0.5,
        vmin=0,
        vmax=100,
    )
    plt.title(
        f"Confusion Matrix - {strategy} ({kshot}-shot)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Predicted", fontsize=12, fontweight="bold")
    plt.ylabel("True", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")

    # Highlight incremental classes
    for i, cls in enumerate(class_list):
        if cls in NEW_CLASSES:
            idx = i
            plt.gca().add_patch(
                plt.Rectangle((idx, idx), 1, 1, fill=False, edgecolor="red", lw=4)
            )

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"confusion_matrix_{strategy}_{kshot}shot.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved confusion matrix: {save_path}")


def plot_per_class_accuracy(cm, class_list, strategy, kshot, output_dir):
    """Plot per-class accuracy"""
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-12) * 100
    display_names = [CLASS_NAMES[i] for i in class_list]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(display_names))
    colors = ["#3498db" if cls in BASE_CLASSES else "#e74c3c" for cls in class_list]

    bars = ax.bar(
        x, per_class_acc, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Class", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Per-Class Accuracy - {strategy} ({kshot}-shot)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.set_ylim([0, 105])
    ax.grid(axis="y", alpha=0.3)

    # Highlight incremental classes region
    if len(class_list) > len(BASE_CLASSES):
        ax.axvspan(
            len(BASE_CLASSES) - 0.5, len(class_list) - 0.5, alpha=0.1, color="red"
        )

    plt.tight_layout()
    save_path = os.path.join(
        output_dir, f"per_class_accuracy_{strategy}_{kshot}shot.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved per-class accuracy: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Test CIFAR-10 FSCIL Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (e.g., checkpoints_cifar_resnet50/ORCO_exp1_pretrained_frozen_best.pt)",
    )
    parser.add_argument(
        "--k_shot", type=int, default=5, help="K-shot for incremental classes"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["BASELINE", "ORCO", "CONCM", "CONCM_FULL"],
        required=True,
        help="Strategy used for training (BASELINE, ORCO, CONCM, or CONCM_FULL)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Whether the encoder was pretrained on ImageNet",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Directory to save test results",
    )
    parser.add_argument(
        "--orco_targets",
        type=str,
        default=None,
        help="Path to saved ORCO pseudo targets (optional)",
    )
    parser.add_argument(
        "--fixed_samples",
        type=str,
        default=None,
        help="Path to CSV file with fixed test samples for consistent evaluation (optional)",
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("TESTING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Strategy: {args.strategy}")
    print(f"K-shot (incremental): {args.k_shot}")
    print(f"Pretrained encoder: {args.pretrained}")
    print(f"Output directory: {args.output_dir}")
    print(
        f"Fixed samples: {args.fixed_samples if args.fixed_samples else 'None (using all test data)'}"
    )
    print(f"{'='*70}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load fixed test samples if provided
    fixed_test_indices = None
    if args.fixed_samples:
        if not os.path.exists(args.fixed_samples):
            print(f"❌ Error: Fixed samples file not found at {args.fixed_samples}")
            exit(1)
        print("📥 Loading fixed test samples...")
        fixed_test_indices = load_fixed_samples(args.fixed_samples)
        total_samples = sum(len(indices) for indices in fixed_test_indices.values())
        print(
            f"✓ Loaded {total_samples} fixed samples across {len(fixed_test_indices)} classes"
        )
        for cls, indices in sorted(fixed_test_indices.items()):
            print(f"   Class {cls}: {len(indices)} samples")
        print()

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        exit(1)

    print("📥 Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)

    # Initialize encoder and classifier
    encoder = ResNet50Encoder(pretrained=False).to(DEVICE)  # Load architecture
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    print("✓ Encoder loaded successfully")

    # Load CONCM_FULL framework if needed
    concm_framework = None
    if args.strategy == "CONCM_FULL" and "concm_framework" in ckpt:
        print("📥 Loading ConCM framework...")
        concm_framework = ConCMFramework(
            feat_dim=encoder.out_dim, attr_dim=128, geom_dim=128
        ).to(DEVICE)
        concm_framework.load_state_dict(ckpt["concm_framework"])
        concm_framework.eval()
        print("✓ ConCM framework loaded successfully")

    # Setup data transforms
    mean, std = (
        ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if args.pretrained
        else ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(224) if args.pretrained else transforms.Resize(32),
            (
                transforms.CenterCrop(224)
                if args.pretrained
                else transforms.CenterCrop(32)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Load datasets
    print("\n📂 Loading CIFAR-10 dataset...")
    trainset = datasets.CIFAR10(
        "../../data", train=True, download=True, transform=transform_test
    )
    testset = datasets.CIFAR10(
        "../../data", train=False, download=True, transform=transform_test
    )

    base_idx = [i for i, (_, y) in enumerate(trainset) if int(y) in BASE_CLASSES]
    base_train = Subset(trainset, base_idx)

    print(f"✓ Base classes (7): {[CLASS_NAMES[i] for i in BASE_CLASSES]}")
    print(f"✓ Incremental classes (3): {[CLASS_NAMES[i] for i in NEW_CLASSES]}")

    # Load ORCO pseudo targets if needed
    orco_pseudo_targets = None
    if (
        args.strategy == "ORCO"
        and args.orco_targets
        and os.path.exists(args.orco_targets)
    ):
        print(f"\n📥 Loading ORCO pseudo targets from {args.orco_targets}...")
        orco_pseudo_targets = torch.load(args.orco_targets, map_location=DEVICE)
        print("✓ ORCO targets loaded")

    # ============================================================================
    # Session 0: Evaluate Base Model
    # ============================================================================
    print("\n" + "=" * 70)
    print("SESSION 0: BASE MODEL EVALUATION")
    print("=" * 70)

    base_protos = compute_prototypes(encoder, base_train, BASE_CLASSES, device=DEVICE)
    _, base_acc_s0 = evaluate_proto_bank(
        encoder,
        base_protos,
        testset,
        BASE_CLASSES,
        device=DEVICE,
        fixed_indices=fixed_test_indices,
    )
    print(f"📊 Base Classes Accuracy: {base_acc_s0:.2f}%")

    # ============================================================================
    # Session 1: Incremental Learning (or Baseline Evaluation)
    # ============================================================================
    if args.strategy == "BASELINE":
        print("\n" + "=" * 70)
        print("BASELINE EVALUATION: Testing on Unknown Classes")
        print("=" * 70)
        print(f"ℹ️  Baseline model was trained ONLY on base classes (0-6)")
        print(f"   It has NEVER seen classes 7, 8, 9 (horse, ship, truck)")
        print(f"   New classes will be misclassified as one of the known base classes")
        print(f"   This demonstrates the need for incremental learning methods")

        # For baseline, we only use base prototypes (no new class prototypes added)
        proto_bank = OrderedDict(base_protos)
        seen_classes = BASE_CLASSES + NEW_CLASSES  # Evaluate on all classes
        novel_classes = NEW_CLASSES

        print(
            f"\n📊 Using only {len(BASE_CLASSES)} base class prototypes to classify all {len(seen_classes)} classes..."
        )
    else:
        print("\n" + "=" * 70)
        print(f"SESSION 1: INCREMENTAL LEARNING ({args.k_shot}-SHOT)")
        print("=" * 70)

        proto_bank = OrderedDict(base_protos)
        seen_classes = BASE_CLASSES.copy()
        novel_classes = []

        for nc in NEW_CLASSES:
            print(
                f"\n➕ Adding class {nc} ({CLASS_NAMES[nc]}) with {args.k_shot}-shot..."
            )
            kidx = few_shot_indices(trainset, nc, args.k_shot)
            novel_set = Subset(trainset, kidx)
            novel_proto = compute_prototypes(encoder, novel_set, [nc], device=DEVICE)[
                nc
            ]

            # Apply calibration based on strategy
            if args.strategy == "CONCM":
                novel_proto = concm_calibrate_torch(
                    novel_proto, proto_bank, CONCM_ALPHA
                )
                print(f"   ✓ Applied CONCM calibration (α={CONCM_ALPHA})")
            elif args.strategy == "CONCM_FULL" and concm_framework is not None:
                novel_proto = concm_framework.calibrate_novel_class(
                    novel_proto, nc, alpha=0.6
                )
                print(f"   ✓ Applied full ConCM calibration (MPC + DSM)")
            elif args.strategy == "ORCO" and orco_pseudo_targets is not None:
                t = F.normalize(orco_pseudo_targets[nc].unsqueeze(0), dim=1)
                p = F.normalize(novel_proto, dim=1)
                novel_proto = F.normalize(ORCO_BLEND * p + (1 - ORCO_BLEND) * t, dim=1)
                print(f"   ✓ Applied ORCO blending (β={ORCO_BLEND})")

            proto_bank[nc] = novel_proto
            novel_classes.append(nc)
            seen_classes.append(nc)

    # ============================================================================
    # Final Evaluation
    # ============================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    _, base_acc_s1 = evaluate_proto_bank(
        encoder,
        proto_bank,
        testset,
        BASE_CLASSES,
        device=DEVICE,
        fixed_indices=fixed_test_indices,
    )
    _, novel_acc = evaluate_proto_bank(
        encoder,
        proto_bank,
        testset,
        NEW_CLASSES,
        device=DEVICE,
        fixed_indices=fixed_test_indices,
    )
    _, overall_acc = evaluate_proto_bank(
        encoder,
        proto_bank,
        testset,
        seen_classes,
        device=DEVICE,
        fixed_indices=fixed_test_indices,
    )

    forgetting = base_acc_s0 - base_acc_s1

    print(f"\n📊 Results Summary:")
    print(f"   Base Classes (Session 0):     {base_acc_s0:.2f}%")
    print(f"   Base Classes (After Incr.):   {base_acc_s1:.2f}%")
    print(f"   Incremental Classes:          {novel_acc:.2f}%")
    print(f"   Overall Accuracy:             {overall_acc:.2f}%")
    print(f"   Forgetting:                   {forgetting:+.2f}%", end="")

    if args.strategy == "BASELINE":
        print(" (No forgetting expected - model never learned new classes)")
        print(f"\n⚠️  BASELINE Analysis:")
        print(
            f"   - Base classes maintain high accuracy (~{base_acc_s1:.1f}%) - no forgetting"
        )
        print(
            f"   - New classes show ~{novel_acc:.1f}% accuracy (random guessing ~{100/len(BASE_CLASSES):.1f}%)"
        )
        print(f"   - Model misclassifies unknown classes as known base classes")
        print(f"   - This demonstrates the need for incremental learning methods")
    else:
        if forgetting < 0:
            print(" (Negative forgetting - base improved!)")
        elif forgetting < 3:
            print(" (Excellent - minimal forgetting)")
        elif forgetting < 5:
            print(" (Good)")
        else:
            print(" (Needs improvement)")

    # ============================================================================
    # Compute and visualize confusion matrix
    # ============================================================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    if args.strategy == "BASELINE":
        print(f"📊 Confusion matrix will show how new classes are misclassified")
        print(f"   Rows 7-9 (new classes) will have NO diagonal values")
        print(f"   Instead, they'll be distributed among base classes 0-6")

    cm, cm_normalized = compute_confusion_matrix(
        encoder,
        proto_bank,
        testset,
        seen_classes,
        device=DEVICE,
        fixed_indices=fixed_test_indices,
    )

    if cm is not None:
        # Plot confusion matrix
        plot_confusion_matrix(
            cm_normalized, seen_classes, args.strategy, args.k_shot, args.output_dir
        )

        # Plot per-class accuracy
        plot_per_class_accuracy(
            cm, seen_classes, args.strategy, args.k_shot, args.output_dir
        )

        # Per-class analysis
        per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-12) * 100
        base_accs = per_class_acc[: len(BASE_CLASSES)]
        inc_accs = per_class_acc[len(BASE_CLASSES) :]

        print(f"\n📊 Per-Class Accuracy:")
        print(f"{'Class':<15} {'Type':<12} {'Accuracy':<12}")
        print("-" * 40)
        for i, cls in enumerate(seen_classes):
            class_type = "Base" if cls in BASE_CLASSES else "Incremental"
            print(f"{CLASS_NAMES[cls]:<15} {class_type:<12} {per_class_acc[i]:>10.2f}%")
        print("-" * 40)
        print(f"{'Base Avg':<15} {'Summary':<12} {np.mean(base_accs):>10.2f}%")
        print(f"{'Inc Avg':<15} {'Summary':<12} {np.mean(inc_accs):>10.2f}%")
        print(f"{'Overall':<15} {'Summary':<12} {np.mean(per_class_acc):>10.2f}%")

    # ============================================================================
    # Save results
    # ============================================================================
    results_file = os.path.join(
        args.output_dir, f"results_{args.strategy}_{args.k_shot}shot.txt"
    )
    with open(results_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"TESTING RESULTS - {args.strategy} ({args.k_shot}-SHOT)\n")
        f.write("=" * 70 + "\n")
        f.write(f"\nCheckpoint: {args.checkpoint}\n")
        f.write(f"Strategy: {args.strategy}\n")
        f.write(f"K-shot: {args.k_shot}\n")
        f.write(f"Pretrained: {args.pretrained}\n")
        f.write(f"\nSession 0 (Base Only):\n")
        f.write(f"  Base Accuracy: {base_acc_s0:.2f}%\n")
        f.write(f"\nSession 1 (After Increment):\n")
        f.write(f"  Base Accuracy: {base_acc_s1:.2f}%\n")
        f.write(f"  Incremental Accuracy: {novel_acc:.2f}%\n")
        f.write(f"  Overall Accuracy: {overall_acc:.2f}%\n")
        f.write(f"  Forgetting: {forgetting:+.2f}%\n")
        if cm is not None:
            f.write(f"\nPer-Class Results:\n")
            f.write(f"  Base Classes Average: {np.mean(base_accs):.2f}%\n")
            f.write(f"  Incremental Classes Average: {np.mean(inc_accs):.2f}%\n")
            f.write(f"  Overall Average: {np.mean(per_class_acc):.2f}%\n")

    print(f"✓ Saved results: {results_file}")

    print("\n" + "=" * 70)
    print("TESTING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 All results saved to: {args.output_dir}/")
    print(f"   • confusion_matrix_{args.strategy}_{args.k_shot}shot.png")
    print(f"   • per_class_accuracy_{args.strategy}_{args.k_shot}shot.png")
    print(f"   • results_{args.strategy}_{args.k_shot}shot.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
