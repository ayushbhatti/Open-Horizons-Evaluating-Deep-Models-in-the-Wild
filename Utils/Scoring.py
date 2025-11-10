#Implements MSP, Energy, Mahalanobis and KNN scoring
import torch, numpy as np
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import NearestNeighbors
from .training import extract_features

def msp_scores(logits): return -F.softmax(logits,dim=1).max(1)[0].numpy()
def energy_scores(logits,T=1.0): return torch.logsumexp(logits/T,dim=1).numpy()
def mahalanobis_scores(feats_id, labels_id, feats_eval):
    mu=[]
    for c in torch.unique(labels_id):
        mu.append(feats_id[labels_id==c].mean(0,keepdim=True))
    mu=torch.cat(mu)
    cov = EmpiricalCovariance().fit((feats_id - mu[labels_id]).numpy())
    inv = cov.precision_
    X,Mu = feats_eval.numpy(), mu.numpy()
    dists=[np.sum((X-Mu[k])@inv*(X-Mu[k]),axis=1) for k in range(Mu.shape[0])]
    return np.min(np.stack(dists,1),1)
def knn_scores(feats_id, feats_eval, k=5):
    nn=NearestNeighbors(n_neighbors=k).fit(feats_id.numpy())
    dists,_=nn.kneighbors(feats_eval.numpy(),n_neighbors=k)
    return dists.mean(1)

def compute_all_scores(backbone, tr_loader, id_logits, id_feats, ood_logits, ood_feats, device):
    Xtr,Ytr = extract_features(backbone, tr_loader, device)
    return {
        "msp": (msp_scores(id_logits), msp_scores(ood_logits)),
        "energy": (energy_scores(id_logits), energy_scores(ood_logits)),
        "mahalanobis": (mahalanobis_scores(Xtr,Ytr,id_feats),
                        mahalanobis_scores(Xtr,Ytr,ood_feats)),
        "knn": (knn_scores(Xtr,id_feats), knn_scores(Xtr,ood_feats))
    }