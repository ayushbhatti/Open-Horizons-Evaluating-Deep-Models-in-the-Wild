#!/usr/bin/env python3
"""
Model architectures and loss functions for CIFAR-10 FSCIL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.out_dim = m.fc.in_features

    def forward(self, x):
        return self.backbone(x).view(x.size(0), -1)


class LinearClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, f):
        return self.net(f)


def unfreeze_last_block(encoder):
    """Unfreeze layer4 and batchnorms for partial fine-tuning"""
    for name, param in encoder.backbone.named_parameters():
        if "layer4" in name or "bn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    print("🔓 Partially unfreezing layer4 + batchnorms")
    return encoder


def unfreeze_last_blocks(encoder):
    """Unfreeze layer3, layer4 and batchnorms for partial fine-tuning"""
    for name, param in encoder.backbone.named_parameters():
        if "layer3" in name or "layer4" in name or "bn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    print("🔓 Unfreezing layer3, layer4 + all batchnorms")
    return encoder


# ============================================================================
# Loss Functions for FSCIL Approaches
# ============================================================================


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    Supervised contrastive loss for OrCo strategy.
    Pulls together samples of the same class and pushes apart different classes.
    """
    f = F.normalize(features, dim=1)
    sim = torch.matmul(f, f.t()) / temperature
    N = f.size(0)
    labels = labels.view(-1, 1)
    mask_pos = torch.eq(labels, labels.t()).float().to(features.device)
    mask_diag = torch.eye(N, device=features.device)
    mask_pos -= mask_diag
    sim -= sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * (1 - mask_diag)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    denom = mask_pos.sum(dim=1)
    nz = denom > 0
    if nz.sum() == 0:
        return torch.tensor(0.0, device=features.device)
    loss = -(mask_pos * log_prob).sum(dim=1)[nz] / denom[nz]
    return loss.mean()


def orthogonality_loss(features, labels=None):
    """
    Orthogonality loss for OrCo strategy.
    Encourages feature orthogonality within batch or between class centroids.
    """
    device = features.device
    if labels is None:
        f = F.normalize(features, dim=1)
        G = torch.matmul(f, f.t())
        return ((G - torch.eye(G.size(0), device=device)) ** 2).mean()
    unique = torch.unique(labels)
    if unique.numel() <= 1:
        return torch.tensor(0.0, device=device)
    centroids = [features[labels == u].mean(0) for u in unique]
    C = F.normalize(torch.stack(centroids, 0), dim=1)
    G = torch.matmul(C, C.t())
    off = G - torch.diag(torch.diag(G))
    return (off**2).sum() / (C.size(0) * (C.size(0) - 1) + 1e-12)


def generate_orthogonal_targets(
    num_targets, dim, tau_o=0.07, steps=500, lr=1e-2, device="cuda"
):
    """
    Generate orthogonal pseudo-targets for OrCo strategy.
    These targets guide the model to learn orthogonal representations.
    """
    T = torch.randn(num_targets, dim, device=device, requires_grad=True)
    opt = torch.optim.SGD([T], lr=lr)
    for _ in range(steps):
        sim = torch.matmul(T, T.t()) / tau_o
        eye = torch.eye(num_targets, device=device)
        loss = ((sim * (1 - eye)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            T[:] = F.normalize(T, dim=1)
    return T.detach()


def target_contrastive_loss(
    features,
    base_indices,
    base_targets,
    all_targets,
    lambda_pert=1e-2,
    temperature=0.07,
):
    """
    Target contrastive loss for OrCo strategy.
    Aligns features with pseudo-targets using perturbed augmentation.
    """
    device = features.device
    f = F.normalize(features, dim=1)
    pert = torch.empty_like(all_targets).uniform_(-lambda_pert, lambda_pert).to(device)
    perturbed = F.normalize(all_targets + pert, dim=1)
    bank = torch.cat([all_targets, perturbed], dim=0)
    T_all = all_targets.size(0)
    sim = torch.matmul(f, bank.t()) / temperature
    N = features.size(0)
    pos_mask = torch.zeros_like(sim)
    for i in range(N):
        c = int(base_indices[i].item())
        pos_mask[i, c] = 1
        pos_mask[i, c + T_all] = 1
    log_prob = sim - torch.log(torch.exp(sim).sum(dim=1, keepdim=True) + 1e-12)
    pos_log_prob = (pos_mask * log_prob).sum(dim=1)
    num_pos = pos_mask.sum(dim=1).clamp_min(1.0)
    return -(pos_log_prob / num_pos).mean()
