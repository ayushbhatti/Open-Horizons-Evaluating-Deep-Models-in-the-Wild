# Implements MSP, Energy, Mahalanobis and kNN scoring

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

from .Training import extract_features


# -----------------------------
# Simple scores: MSP & Energy
# -----------------------------
def msp_scores(logits: torch.Tensor) -> np.ndarray:
    """
    Maximum Softmax Probability (MSP).
    Returns negative max prob so that larger values = more OOD-like.
    """
    probs = F.softmax(logits, dim=1)
    maxp = probs.max(dim=1)[0].cpu().numpy()
    return -maxp


def energy_scores(logits: torch.Tensor, T: float = 1.0) -> np.ndarray:
    """
    Energy score: E(x) = -T * logsumexp(logits / T).
    Larger E(x) corresponds to more OOD-like samples.
    """
    E = -T * torch.logsumexp(logits / T, dim=1)
    return E.cpu().numpy()


# -----------------------------
# Feature preprocessing
# -----------------------------
def center_and_normalize_with_mean(
    feats: torch.Tensor, mean: torch.Tensor | None = None
):
    """
    Center features by a shared mean vector and apply L2 normalization.
    If mean is None, compute it from feats and also return it.
    This stabilizes distance-based OSR methods, especially for modern backbones.
    """
    if mean is None:
        mean = feats.mean(0, keepdim=True)
    feats = feats - mean
    feats = F.normalize(feats, dim=1)
    return feats, mean


# -----------------------------
# Mahalanobis distance (with LedoitWolf)
# -----------------------------
def mahalanobis_scores(
    feats_id: torch.Tensor,
    labels_id: torch.Tensor,
    feats_eval: torch.Tensor,
) -> np.ndarray:
    """
    Class-conditional Mahalanobis distance with shared covariance.
    Assumes feats_id and feats_eval are already centered + L2-normalized.
    Returns the minimum distance to any class mean (larger = more OOD-like).
    """
    classes = torch.unique(labels_id).tolist()

    # Class means in feature space
    mu_list = []
    for c in classes:
        mu_list.append(feats_id[labels_id == c].mean(0, keepdim=True))
    mu = torch.cat(mu_list, dim=0)  # [K, D]

    # Fit shared covariance on centered features using shrinkage
    Xc = feats_id - mu[labels_id]  # [N, D]
    cov = LedoitWolf().fit(Xc.cpu().numpy())
    inv_cov = cov.precision_  # [D, D]

    X = feats_eval.cpu().numpy()  # [N_eval, D]
    Mu = mu.cpu().numpy()        # [K, D]

    dists = []
    for k in range(Mu.shape[0]):
        diff = X - Mu[k]  # [N_eval, D]
        m = np.sum(diff @ inv_cov * diff, axis=1)  # Mahalanobis distance
        dists.append(m)

    dmin = np.min(np.stack(dists, axis=1), axis=1)
    return dmin  # larger = more OOD-like


# -----------------------------
# kNN in feature space
# -----------------------------
def knn_scores(
    feats_id: torch.Tensor,
    feats_eval: torch.Tensor,
    k: int = 5,
) -> np.ndarray:
    """
    k-Nearest Neighbour distance in feature space.
    Assumes feats_id and feats_eval are already centered + L2-normalized.
    """
    X_ref = feats_id.cpu().numpy()
    X_eval = feats_eval.cpu().numpy()

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X_ref)
    dists, _ = nn.kneighbors(X_eval, n_neighbors=k, return_distance=True)
    return dists.mean(axis=1)  # larger => more OOD-like


# -----------------------------
# Wrapper to compute all scores
# -----------------------------
def compute_all_scores(
    backbone,
    tr_loader,
    id_logits: torch.Tensor,
    id_feats: torch.Tensor,
    ood_logits: torch.Tensor,
    ood_feats: torch.Tensor,
    device: str,
):

    # Extract training features (for distance-based methods)
    Xtr, Ytr = extract_features(backbone, tr_loader, device)

    # Preprocess features for Mahalanobis & kNN using a shared training mean
    Xtr_proc, global_mean    = center_and_normalize_with_mean(Xtr)
    id_feats_proc, _         = center_and_normalize_with_mean(id_feats, global_mean)
    ood_feats_proc, _        = center_and_normalize_with_mean(ood_feats, global_mean)

    scores = {
        "msp": (
            msp_scores(id_logits),
            msp_scores(ood_logits),
        ),
        "energy": (
            energy_scores(id_logits),
            energy_scores(ood_logits),
        ),
        "mahalanobis": (
            mahalanobis_scores(Xtr_proc, Ytr, id_feats_proc),
            mahalanobis_scores(Xtr_proc, Ytr, ood_feats_proc),
        ),
        "knn": (
            knn_scores(Xtr_proc, id_feats_proc),
            knn_scores(Xtr_proc, ood_feats_proc),
        ),
    }

    return scores