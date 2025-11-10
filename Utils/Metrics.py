# Evaluation Metrics(AUROC, AUPR, FPR@95, OSCR)

import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch.nn.functional as F

def fpr_at_95_tpr(id_s, ood_s):
    y_true=np.concatenate([np.zeros_like(id_s),np.ones_like(ood_s)])
    y_score=np.concatenate([id_s,ood_s])
    fpr,tpr,_=roc_curve(y_true,y_score)
    mask=tpr>=0.95
    return fpr[mask][0] if mask.any() else 1.0

def oscr(id_logits,id_labels,ood_logits):
    def maxprob(x): return F.softmax(torch.from_numpy(x),dim=1).numpy().max(1)
    Lid=id_labels.numpy()
    Mid=maxprob(id_logits)[0]; Mood=maxprob(ood_logits)[0]
    ths=np.unique(np.concatenate([Mid,Mood]))
    curv=[]
    for t in ths:
        yhat=id_logits.argmax(1).numpy()
        correct=(yhat==Lid)&(Mid>=t)
        cid=correct.mean(); far=(Mood>=t).mean()
        curv.append((far,cid))
    curv=sorted(curv); auc=0
    for i in range(1,len(curv)):
        x0,y0=curv[i-1]; x1,y1=curv[i]
        auc+=(x1-x0)*(y1+y0)/2
    return auc

def evaluate_osr(scores,id_logits,te_id_loader,ood_logits):
    print("\n=== OSR Metrics ===")
    for name,(id_s,ood_s) in scores.items():
        y_true=np.concatenate([np.zeros_like(id_s),np.ones_like(ood_s)])
        y_score=np.concatenate([id_s,ood_s])
        auroc=roc_auc_score(y_true,y_score)
        aupr=average_precision_score(y_true,y_score)
        fpr95=fpr_at_95_tpr(id_s,ood_s)
        print(f"{name:12s} | AUROC={auroc:.4f}  AUPR={aupr:.4f}  FPR@95={fpr95:.4f}")

    oscr_val=oscr(id_logits,torch.cat([y for _,y in te_id_loader.dataset]),ood_logits)
    print(f"\nOSCR (max-softmax curve area): {oscr_val:.4f}")