#Handles feature extraction and linear head training
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

@torch.no_grad()
def extract_features(backbone, loader, device):
    feats, labels = [], []
    for x,y in loader:
        x = x.to(device)
        f = backbone(x)
        if isinstance(f, (list,tuple)): f=f[0]
        feats.append(f.cpu()); labels.append(y)
    return torch.cat(feats), torch.cat(labels)

def train_linear(backbone, feat_dim, tr_loader, num_classes, device, epochs=20, lr=1e-2, wd=1e-4):
    Xtr, Ytr = extract_features(backbone, tr_loader, device)
    dataset = TensorDataset(Xtr, Ytr)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    head = nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        head.train(); losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            loss=ce(head(xb),yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if (ep+1)%5==0: print(f"Epoch {ep+1}: loss={np.mean(losses):.4f}")
    return head

@torch.no_grad()
def logits_on_loader(backbone, head, loader, device):
    logits, feats = [], []
    for x,_ in loader:
        x = x.to(device)
        f = backbone(x)
        if isinstance(f, (list,tuple)): f=f[0]
        z = head(f)
        logits.append(z.cpu()); feats.append(f.cpu())
    return torch.cat(logits), torch.cat(feats)