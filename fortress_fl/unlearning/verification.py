"""Verification of unlearning beyond proxy metrics."""

from typing import Dict, List, Optional

import numpy as np

from .models import sigmoid as _sigmoid


def predict_proba(theta, X, model=None):
    z = X @ theta if model is None else model.logit(theta, X)
    return _sigmoid(z)


def accuracy(theta, data, model=None) -> float:
    return float(np.mean((predict_proba(theta, data["X"], model) > 0.5) == data["y"]))


def per_sample_loss(theta, X, y, model=None):
    p = np.clip(predict_proba(theta, X, model), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _auc(pos_scores, neg_scores) -> float:
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney)."""
    s = np.concatenate([pos_scores, neg_scores])
    ranks = s.argsort().argsort() + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = len(pos_scores), len(neg_scores)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mia_auc(theta, forget: Dict, nonmember: Dict) -> float:
    """Loss-threshold membership inference (Yeom et al.): score = -loss."""
    lf = per_sample_loss(theta, forget["X"], forget["y"])
    ln = per_sample_loss(theta, nonmember["X"], nonmember["y"])
    return _auc(-lf, -ln)


def backdoor_asr(theta, bd: Dict, model=None) -> float:
    return float(np.mean((predict_proba(theta, bd["X"], model) > 0.5) == bd["y_target"]))


def param_gap(theta_u, theta_ref, theta_ref2) -> float:
    """Distance to a retrained reference, normalised by the distance between
    two independent retrainings (the noise floor of retraining itself)."""
    floor = np.linalg.norm(theta_ref - theta_ref2) + 1e-12
    return float(np.linalg.norm(theta_u - theta_ref) / floor)


def prediction_disagreement(theta_a, theta_b, X) -> float:
    return float(np.mean((predict_proba(theta_a, X) > 0.5) != (predict_proba(theta_b, X) > 0.5)))


def forget_loss_gap(theta_u, theta_ref, forget) -> float:
    """Mean loss on forgotten data relative to the retrained model (1.0 = same)."""
    lu = per_sample_loss(theta_u, forget["X"], forget["y"]).mean()
    lr = per_sample_loss(theta_ref, forget["X"], forget["y"]).mean()
    return float(lu / (lr + 1e-12))
