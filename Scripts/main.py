import argparse
import torch
import sys, os

# --- ensure parent directory is on path ---
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# --- import helpers (match your actual folder names) ---
from Data.CIFAR10_loader import get_cifar10_loaders
from Models.Backbone import build_backbone
from Models.Linear_head import LinearHead
from Utils.Training import train_linear, extract_features, logits_on_loader
from Utils.Scoring import compute_all_scores
from Utils.Metrics import evaluate_osr
from Utils.misc import set_seed


def main():
    print("🚀 Starting OSR training pipeline...")  # ✅ visible sanity print

    parser = argparse.ArgumentParser()
    parser.add_argument("--known-classes", type=int, default=6)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    # --- Load CIFAR-10 data ---
    print("📦 Loading CIFAR-10 dataset...")
    tr_loader, te_id_loader, te_ood_loader, known, unknown = get_cifar10_loaders(args.known_classes, args.batch_size)
    print(f"Known classes: {known}, Unknown classes: {unknown}")

    # --- Build frozen feature extractor ---
    backbone, feat_dim = build_backbone(args.backbone)
    backbone.to(device).eval()
    print(f"🧠 Loaded backbone: {args.backbone} (feature dim = {feat_dim})")

    # --- Train linear head ---
    print("🎯 Training linear head...")
    head = train_linear(backbone, feat_dim, tr_loader, len(known), device, epochs=args.epochs, lr=args.lr)

    # --- Extract logits/features for ID and OOD sets ---
    print("🔍 Extracting logits and features...")
    id_logits, id_feats = logits_on_loader(backbone, head, te_id_loader, device)
    ood_logits, ood_feats = logits_on_loader(backbone, head, te_ood_loader, device)

    # --- Compute OSR scores ---
    print("📈 Computing OSR scores (MSP, Energy, Mahalanobis, kNN)...")
    scores = compute_all_scores(backbone, tr_loader, id_logits, id_feats, ood_logits, ood_feats, device)

    # --- Evaluate ---
    print("📊 Evaluating OSR metrics...")
    evaluate_osr(scores, id_logits, te_id_loader, ood_logits)

    print("\n✅ Done! Results printed above.")


if __name__ == "__main__":
    main()