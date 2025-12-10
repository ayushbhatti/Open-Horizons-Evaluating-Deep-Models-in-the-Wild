# Handles feature extraction and linear head training
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

# Optional: colored output for clarity
def log_stage(msg):
    print(f"\033[96m[INFO]\033[0m {msg}")   # cyan color for stages

def log_progress(msg):
    print(f"\033[93m[PROGRESS]\033[0m {msg}")  # yellow for updates


# ------------------------------
# FEATURE EXTRACTION
# ------------------------------
@torch.no_grad()
def extract_features(backbone, loader, device):
    """
    Extract feature embeddings for an entire dataset using the frozen backbone.
    Shows a tqdm progress bar for visual confirmation.
    """
    log_stage("Starting feature extraction...")
    feats, labels = [], []

    for x, y in tqdm(loader, desc="🔍 Extracting features", ncols=90):
        x = x.to(device)
        f = backbone(x)
        if isinstance(f, (list, tuple)):
            f = f[0]
        feats.append(f.cpu())
        labels.append(y)

    feats = torch.cat(feats)
    labels = torch.cat(labels)

    log_stage(f" Feature extraction complete: {feats.shape[0]} samples, dim={feats.shape[1]}")
    return feats, labels


# ------------------------------
# LINEAR HEAD TRAINING
# ------------------------------
def train_linear(backbone, feat_dim, tr_loader, num_classes, device,
                 epochs=20, lr=1e-2, wd=1e-4):
    """
    Train a linear classifier on frozen backbone features.
    Visual confirmation for each epoch + batch.
    """
    # Cache features first
    Xtr, Ytr = extract_features(backbone, tr_loader, device)
    dataset = TensorDataset(Xtr, Ytr)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    head = nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    ce = nn.CrossEntropyLoss()

    log_stage("Starting linear head training...")

    for ep in range(epochs):
        head.train()
        losses = []

        # tqdm progress bar per epoch
        for xb, yb in tqdm(loader, desc=f"🧠 Epoch {ep+1}/{epochs}", ncols=90):
            xb, yb = xb.to(device), yb.to(device)
            loss = ce(head(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        log_progress(f"Epoch {ep+1}/{epochs} finished — avg loss: {avg_loss:.4f}")

    log_stage(" Linear head training complete.")
    return head


# ------------------------------
# LOGITS EXTRACTION
# ------------------------------
@torch.no_grad()
def logits_on_loader(backbone, head, loader, device):
    """
    Extract logits (predictions) and features for evaluation.
    Shows tqdm progress for visual feedback.
    """
    log_stage("Extracting logits and features for evaluation...")
    logits, feats = [], []

    for x, _ in tqdm(loader, desc="📊 Extracting logits", ncols=90):
        x = x.to(device)
        f = backbone(x)
        if isinstance(f, (list, tuple)):
            f = f[0]
        z = head(f)
        logits.append(z.cpu())
        feats.append(f.cpu())

    log_stage(" Logit extraction complete.")
    return torch.cat(logits), torch.cat(feats)