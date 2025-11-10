import argparse, math, os, random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms

from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.neighbors import NearestNeighbors

# --------------------
# Utils
# --------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)

@dataclass
class Scores:
    id_scores: Dict[str, np.ndarray]   # method -> scores on in-distribution
    ood_scores: Dict[str, np.ndarray]  # method -> scores on out-of-distribution

# --------------------
# Data (CIFAR-10)
# We simulate OSR by treating first K classes as known; rest as unknown
# --------------------
def get_cifar10_loaders(known_classes=6, batch_size=256, num_workers=4):
    assert 1 <= known_classes <= 9, "Keep at least one unknown class"

    normalize = transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616])
    T_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])
    T_test = transforms.Compose([transforms.ToTensor(), normalize])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=T_train)
    testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=T_test)

    known = list(range(known_classes))
    unknown = list(range(known_classes, 10))

    # indices
    tr_idx = [i for i,(_,y) in enumerate(trainset) if y in known]
    te_id_idx = [i for i,(_,y) in enumerate(testset) if y in known]
    te_ood_idx = [i for i,(_,y) in enumerate(testset) if y in unknown]

    # relabel known classes to [0..K-1]
    class_map = {c:i for i,c in enumerate(known)}
    def relabel_target(dataset, idxs):
        ds = Subset(dataset, idxs)
        old_getitem = ds.dataset.__getitem__
        def _getitem(i):
            x,y = old_getitem(ds.indices[i])
            return x, class_map[y]
        ds.dataset.__getitem__ = lambda i: _getitem(i)
        return ds

    train_known = relabel_target(trainset, tr_idx)
    test_known  = relabel_target(testset, te_id_idx)
    test_ood    = Subset(testset, te_ood_idx)  # labels unused

    tr_loader = DataLoader(train_known, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    te_id_loader = DataLoader(test_known, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    te_ood_loader = DataLoader(test_ood, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return tr_loader, te_id_loader, te_ood_loader, known, unknown

# --------------------
# Backbones (frozen) + linear head
# Proposal expects frozen pretrained encoders such as ResNet50, ConvNeXt, DINOv2, CLIP ViT-B/16
# We'll wire torchvision backbones here; CLIP/DINOv2 can be added later.
# --------------------
def build_backbone(name: str):
    name = name.lower()
    if name == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "convnext_tiny":
        m = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        feat_dim = m.classifier[2].in_features
        m.classifier[2] = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    for p in m.parameters(): p.requires_grad = False
    m.eval()
    return m, feat_dim

class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.fc(x)

# --------------------
# Train linear head on frozen features
# --------------------
@torch.no_grad()
def extract_features(backbone, loader, device):
    feats, targets = [], []
    for x,y in loader:
        x = x.to(device)
        f = backbone(x)
        if isinstance(f, (list, tuple)): f = f[0]
        feats.append(f.cpu())
        targets.append(y)
    return torch.cat(feats), torch.cat(targets)

def train_linear(backbone, feat_dim, tr_loader, num_classes, device, epochs=20, lr=1e-2, wd=1e-4):
    # cache features for efficiency
    backbone.eval()
    Xtr, Ytr = extract_features(backbone, tr_loader, device)
    dataset = torch.utils.data.TensorDataset(Xtr, Ytr)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    head = LinearHead(feat_dim, num_classes).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        head.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = head(xb)
            loss = ce(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        if (ep+1) % 5 == 0:
            print(f"Epoch {ep+1}: loss={np.mean(losses):.4f}")
    return head

# --------------------
# Scoring methods
# --------------------
@torch.no_grad()
def logits_on_loader(backbone, head, loader, device):
    all_logits, all_feats = [], []
    for x,_ in loader:
        x = x.to(device)
        f = backbone(x); 
        if isinstance(f, (list, tuple)): f = f[0]
        z = head(f)
        all_logits.append(z.cpu()); all_feats.append(f.cpu())
    return torch.cat(all_logits), torch.cat(all_feats)

def msp_scores(logits: torch.Tensor):
    probs = F.softmax(logits, dim=1).numpy()
    return probs.max(axis=1) * (-1.0)  # more unknown-like => larger score; so invert MSP

def energy_scores(logits: torch.Tensor, T=1.0):
    # Energy = -T * logsumexp(logits/T)
    E = -T * torch.logsumexp(logits/T, dim=1)
    return (-E).numpy()  # higher is more unknown-like -> negate

def mahalanobis_scores(feats_id: torch.Tensor, labels_id: torch.Tensor, feats_eval: torch.Tensor):
    # Fit class-conditional Gaussians with shared covariance
    mu = []
    classes = torch.unique(labels_id).tolist()
    for c in classes:
        mu.append(feats_id[labels_id==c].mean(0, keepdim=True))
    mu = torch.cat(mu, dim=0)  # [K, D]
    # Shared covariance
    Xc = feats_id - mu[labels_id]
    cov = EmpiricalCovariance().fit(Xc.numpy())
    inv_cov = cov.precision_
    # class-wise Mahalanobis distance -> take min over classes
    X = feats_eval.numpy()
    Mu = mu.numpy()
    dists = []
    for k in range(Mu.shape[0]):
        diff = X - Mu[k]
        # (x-μ)^T Σ^{-1} (x-μ)
        m = np.sum(diff @ inv_cov * diff, axis=1)
        dists.append(m)
    dmin = np.min(np.stack(dists, axis=1), axis=1)
    return dmin  # larger = more unknown-like

def knn_scores(feats_id: torch.Tensor, feats_eval: torch.Tensor, k=5):
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(feats_id.numpy())
    dists, _ = nn.kneighbors(feats_eval.numpy(), n_neighbors=k, return_distance=True)
    return dists.mean(axis=1)  # larger => more unknown-like

# --------------------
# Metrics: AUROC, AUPR, FPR@95TPR, OSCR
# --------------------
def fpr_at_95_tpr(id_scores, ood_scores):
    # lower score = more in-distribution; we set things so that larger = more OOD-like
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])  # 1=OOD
    y_score = np.concatenate([id_scores, ood_scores])
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # find FPR where TPR >= 0.95
    mask = tpr >= 0.95
    return fpr[mask][0] if mask.any() else 1.0

def oscr(id_logits, id_labels, ood_logits):
    # OSCR: for varying threshold on max softmax prob,
    # plot correct-ID-rate vs false-accept-rate(OOD) and integrate
    def maxprob(logits): 
        return F.softmax(torch.from_numpy(logits), dim=1).numpy().max(axis=1)
    Lid = id_labels.numpy()
    Mid = maxprob(id_logits.numpy())
    Mood = maxprob(ood_logits.numpy())
    # sweep thresholds over all unique maxprob values
    ths = np.unique(np.concatenate([Mid, Mood]))
    curv = []
    for t in ths:
        # correct ID if maxprob>=t and predicted label correct
        yhat = id_logits.argmax(dim=1).numpy()
        correct = (yhat == Lid) & (Mid >= t)
        cid_rate = correct.mean() if len(correct)>0 else 0.0
        far_ood = (Mood >= t).mean() if len(Mood)>0 else 0.0
        curv.append((far_ood, cid_rate))
    curv = sorted(curv)  # sort by FAR (x)
    # trapezoidal area under curve
    auc = 0.0
    for i in range(1, len(curv)):
        x0,y0 = curv[i-1]; x1,y1 = curv[i]
        auc += (x1-x0) * (y1+y0)/2
    return auc

# --------------------
# Main
# --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-classes", type=int, default=6)
    ap.add_argument("--backbone", type=str, default="resnet50")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_loader, te_id_loader, te_ood_loader, known, unknown = get_cifar10_loaders(
        known_classes=args.known_classes, batch_size=args.batch_size
    )
    print(f"Known classes: {known} | Unknown classes: {unknown}")

    backbone, feat_dim = build_backbone(args.backbone)
    backbone.to(device).eval()

    head = train_linear(backbone, feat_dim, tr_loader, num_classes=len(known),
                        device=device, epochs=args.epochs, lr=args.lr)

    # Collect logits & feats
    id_logits, id_feats = logits_on_loader(backbone, head, te_id_loader, device)
    ood_logits, ood_feats = logits_on_loader(backbone, head, te_ood_loader, device)

    # Scores
    scores = Scores(id_scores={}, ood_scores={})
    scores.id_scores["msp"] = msp_scores(id_logits)
    scores.ood_scores["msp"] = msp_scores(ood_logits)

    scores.id_scores["energy"] = energy_scores(id_logits)
    scores.ood_scores["energy"] = energy_scores(ood_logits)

    # Mahalanobis needs ID training features + labels
    Xtr, Ytr = extract_features(backbone, tr_loader, device)
    scores.id_scores["mahalanobis"] = mahalanobis_scores(Xtr, Ytr, id_feats)
    scores.ood_scores["mahalanobis"] = mahalanobis_scores(Xtr, Ytr, ood_feats)

    # kNN in feature space
    scores.id_scores["knn"] = knn_scores(Xtr, id_feats, k=5)
    scores.ood_scores["knn"] = knn_scores(Xtr, ood_feats, k=5)

    # Metrics
    print("\n=== OSR Metrics ===")
    for name in scores.id_scores.keys():
        id_s = scores.id_scores[name]; ood_s = scores.ood_scores[name]
        y_true = np.concatenate([np.zeros_like(id_s), np.ones_like(ood_s)])  # 1=OOD
        y_score = np.concatenate([id_s, ood_s])
        auroc = roc_auc_score(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)
        fpr95 = fpr_at_95_tpr(id_s, ood_s)
        print(f"{name:12s} | AUROC={auroc:.4f}  AUPR={aupr:.4f}  FPR@95={fpr95:.4f}")

    # OSCR (uses max-softmax)
    oscr_val = oscr(id_logits, torch.cat([y for _,y in te_id_loader.dataset]), ood_logits)
    print(f"\nOSCR (max-softmax curve area): {oscr_val:.4f}")

if __name__ == "__main__":
    main()