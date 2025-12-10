# Evaluation Metrics (AUROC, AUPR, FPR@95, OSCR) + Known/Unknown decisions

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch.nn.functional as F


def fpr_at_95_tpr(id_s, ood_s):
    """
    Compute FPR when TPR (for the OOD class) is at least 0.95.
    Assumes larger scores = more OOD-like.
    """
    y_true = np.concatenate([np.zeros_like(id_s), np.ones_like(ood_s)])  # 0=ID, 1=OOD
    y_score = np.concatenate([id_s, ood_s])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = tpr >= 0.95
    return fpr[mask][0] if mask.any() else 1.0


def threshold_for_target_tpr(id_s, ood_s, target_tpr=0.95):
    """
    Find the score threshold that achieves at least target_tpr
    for the OOD class (treated as positive).
    Returns (threshold, tpr_at_thr, fpr_at_thr).
    """
    y_true = np.concatenate([np.zeros_like(id_s), np.ones_like(ood_s)])  # 0=ID, 1=OOD
    y_score = np.concatenate([id_s, ood_s])
    fpr, tpr, thr = roc_curve(y_true, y_score)

    idx_candidates = np.where(tpr >= target_tpr)[0]
    if len(idx_candidates) == 0:
        idx = len(thr) - 1  # fallback: last threshold
    else:
        idx = idx_candidates[0]

    return thr[idx], tpr[idx], fpr[idx]


def oscr(id_logits, id_labels, ood_logits):
    """Compute Open Set Classification Rate (OSCR) curve area."""

    def maxprob(x):
        # Accept either torch.Tensor or numpy.ndarray
        if isinstance(x, torch.Tensor):
            probs = F.softmax(x, dim=1).cpu().numpy()
        else:
            probs = F.softmax(torch.from_numpy(x), dim=1).numpy()
        # Return the vector of max probabilities for each sample
        return np.max(probs, axis=1)

    Lid = id_labels.numpy()
    Mid = maxprob(id_logits)     # vector of shape [N_id]
    Mood = maxprob(ood_logits)   # vector of shape [N_ood]

    # Now both are 1-D arrays → safe to concatenate
    ths = np.unique(np.concatenate([Mid, Mood]))
    curv = []
    for t in ths:
        yhat = id_logits.argmax(1).numpy()
        correct = (yhat == Lid) & (Mid >= t)
        cid = correct.mean()
        far = (Mood >= t).mean()
        curv.append((far, cid))

    curv = sorted(curv)
    auc = 0.0
    for i in range(1, len(curv)):
        x0, y0 = curv[i - 1]
        x1, y1 = curv[i]
        auc += (x1 - x0) * (y1 + y0) / 2
    return auc


def evaluate_osr(scores, id_logits, te_id_loader, ood_logits):
    """
    Evaluate OSR metrics (AUROC, AUPR, FPR@95, OSCR) and
    also print hard Known/Unknown decisions per scoring method
    using a threshold chosen at target TPR for the OOD class.
    """
    target_tpr = 0.95  # you can change this if you want
    print("\n=== OSR Metrics ===")

    thresholds = {}  # method -> (thr, tpr_at_thr, fpr_at_thr)

    # 1) Standard OSR metrics for each scoring method
    for name, (id_s, ood_s) in scores.items():
        id_s = np.asarray(id_s)
        ood_s = np.asarray(ood_s)

        y_true = np.concatenate([np.zeros_like(id_s), np.ones_like(ood_s)])  # 0=ID, 1=OOD
        y_score = np.concatenate([id_s, ood_s])
        auroc = roc_auc_score(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)
        fpr95 = fpr_at_95_tpr(id_s, ood_s)

        print(f"{name:12s} | AUROC={auroc:.4f}  AUPR={aupr:.4f}  FPR@95={fpr95:.4f}")

        thr, tpr_val, fpr_val = threshold_for_target_tpr(id_s, ood_s, target_tpr=target_tpr)
        thresholds[name] = (thr, tpr_val, fpr_val)

    # 2) OSCR (needs ID labels)
    all_labels = []
    for _, labels in te_id_loader:
        all_labels.append(labels)
    id_labels = torch.cat(all_labels)

    oscr_val = oscr(id_logits, id_labels, ood_logits)
    print(f"\nOSCR (max-softmax curve area): {oscr_val:.4f}")

    # 3) Hard Known/Unknown decisions using thresholds
    print(f"\n=== Known vs Unknown decisions (thresholded at TPR>={target_tpr:.2f}) ===")

    for name, (id_s, ood_s) in scores.items():
        id_s = np.asarray(id_s)
        ood_s = np.asarray(ood_s)
        thr, tpr_val, fpr_val = thresholds[name]

        # score >= thr => predict OOD (Unknown)
        id_pred_unknown = id_s >= thr
        ood_pred_unknown = ood_s >= thr

        n_id = len(id_s)
        n_ood = len(ood_s)

        id_correct_known = (~id_pred_unknown).sum()
        id_misflag_unknown = id_pred_unknown.sum()

        ood_correct_unknown = ood_pred_unknown.sum()
        ood_missed_as_known = (~ood_pred_unknown).sum()

        print(f"\n[{name}]")
        print(f"  Threshold: {thr:.4f}  (TPR_OOD={tpr_val:.3f}, FPR_ID={fpr_val:.3f})")
        print(f"  ID samples   : {id_correct_known}/{n_id} kept as Known, "
              f"{id_misflag_unknown} flagged as Unknown")
        print(f"  OOD samples  : {ood_correct_unknown}/{n_ood} correctly flagged Unknown, "
              f"{ood_missed_as_known} misclassified as Known")