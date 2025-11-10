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
    print("🚀 Starting OSR training pipeline...")

    # ------------------------------
    # Argument parser
    # ------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--known-classes", type=int, default=6)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ------------------------------
    # Device selection (MPS / CUDA / CPU)
    # ------------------------------
    try:
        if torch.backends.mps.is_available():
            device = "mps"
            print("✅ Using Apple MPS GPU for acceleration.")
        elif torch.cuda.is_available():
            device = "cuda"
            print("✅ Using NVIDIA CUDA GPU.")
        else:
            device = "cpu"
            print("⚠️ Using CPU (no GPU backend found).")
    except Exception as e:
        print(f"⚠️ MPS initialization failed, falling back to CPU: {e}")
        device = "cpu"

    print(f"💻 Active device: {device}")
    set_seed(args.seed)

    # ------------------------------
    # Load CIFAR-10 data
    # ------------------------------
    print("\n📦 Loading CIFAR-10 dataset...")
    tr_loader, te_id_loader, te_ood_loader, known, unknown = get_cifar10_loaders(args.known_classes, args.batch_size)
    print(f"Known classes: {known}, Unknown classes: {unknown}")

    # ------------------------------
    # Build frozen feature extractor
    # ------------------------------
    backbone, feat_dim = build_backbone(args.backbone)
    backbone.to(device).eval()
    print(f"🧠 Loaded backbone: {args.backbone} (feature dim = {feat_dim})")

    # ------------------------------
    # Train linear head
    # ------------------------------
    print(f"\n🎯 Training linear head on device: {device} ...")
    head = train_linear(backbone, feat_dim, tr_loader, len(known), device, epochs=args.epochs, lr=args.lr)

    # ------------------------------
    # Extract logits/features
    # ------------------------------
    print(f"\n🔍 Extracting logits and features on device: {device} ...")
    id_logits, id_feats = logits_on_loader(backbone, head, te_id_loader, device)
    ood_logits, ood_feats = logits_on_loader(backbone, head, te_ood_loader, device)

    # ------------------------------
    # Compute OSR scores
    # ------------------------------
    print("\n📈 Computing OSR scores (MSP, Energy, Mahalanobis, kNN)...")
    scores = compute_all_scores(backbone, tr_loader, id_logits, id_feats, ood_logits, ood_feats, device)

    # ------------------------------
    # Evaluate metrics
    # ------------------------------
    print("\n📊 Evaluating OSR metrics...")
    evaluate_osr(scores, id_logits, te_id_loader, ood_logits)

    print("\n✅ Done! Results printed above.")


if __name__ == "__main__":
    main()