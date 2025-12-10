# Utils/Visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import torch
import torch.nn.functional as F


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_score_histograms(scores: dict, save_dir: str = "plots", bins: int = 50):
    """
    Plot histograms of ID vs OOD scores for each OSR method.

    scores: dict[name] -> (id_scores, ood_scores)
    """
    _ensure_dir(save_dir)

    for name, (id_s, ood_s) in scores.items():
        id_s = np.asarray(id_s)
        ood_s = np.asarray(ood_s)

        plt.figure(figsize=(6, 4))
        plt.hist(id_s, bins=bins, alpha=0.6, density=True,
                 label="ID", edgecolor="none")
        plt.hist(ood_s, bins=bins, alpha=0.6, density=True,
                 label="OOD", edgecolor="none")
        plt.title(f"Score histogram: {name}")
        plt.xlabel("Score (larger = more OOD-like)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"hist_{name}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_roc_curves(scores: dict, save_dir: str = "plots"):
    """
    Plot ROC curve (OOD vs ID) for each OSR method on one figure.

    scores: dict[name] -> (id_scores, ood_scores)
    """
    _ensure_dir(save_dir)

    plt.figure(figsize=(6, 5))

    for name, (id_s, ood_s) in scores.items():
        id_s = np.asarray(id_s)
        ood_s = np.asarray(ood_s)

        y_true = np.concatenate(
            [np.zeros_like(id_s), np.ones_like(ood_s)]
        )  # 0=ID, 1=OOD
        y_score = np.concatenate([id_s, ood_s])
        fpr, tpr, _ = roc_curve(y_true, y_score)

        plt.plot(fpr, tpr, label=name)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)  # random baseline
    plt.xlabel("False Positive Rate (ID misclassified as OOD)")
    plt.ylabel("True Positive Rate (OOD correctly detected)")
    plt.title("ROC curves for OSR scores")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    out_path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_oscr_curve(
    id_logits: torch.Tensor,
    id_labels: torch.Tensor,
    ood_logits: torch.Tensor,
    save_dir: str = "plots",
):
    """
    Plot OSCR curve: Correct Classification Rate (CCR) vs False Acceptance Rate (FAR).
    Uses max-softmax probability as confidence.
    """
    _ensure_dir(save_dir)

    # Softmax probabilities
    id_probs = F.softmax(id_logits, dim=1).cpu().numpy()
    ood_probs = F.softmax(ood_logits, dim=1).cpu().numpy()

    # Max-softmax probability per sample
    Mid = id_probs.max(axis=1)      # [N_id]
    Mood = ood_probs.max(axis=1)    # [N_ood]

    # Correctness for ID samples
    yhat = id_logits.argmax(dim=1).cpu().numpy()
    Lid = id_labels.cpu().numpy()
    correct_id = (yhat == Lid)

    # Thresholds over all max-softmax values
    ths = np.unique(np.concatenate([Mid, Mood]))
    CCR = []   # Correct classification rate (ID correct & confident)
    FAR = []   # False acceptance rate (OOD mistaken as known)

    for t in ths:
        ccr = (correct_id & (Mid >= t)).mean()   # ID correct & above threshold
        far = (Mood >= t).mean()                 # OOD above threshold
        CCR.append(ccr)
        FAR.append(far)

    # Plot OSCR curve
    plt.figure(figsize=(6, 5))
    plt.plot(FAR, CCR, label="OSCR Curve")
    plt.xlabel("FAR (Unknown accepted as Known)")
    plt.ylabel("CCR (Correct Known Classification)")
    plt.title("OSCR Curve")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(save_dir, "oscr_curve.png")
    plt.savefig(out_path, dpi=200)
    plt.close()