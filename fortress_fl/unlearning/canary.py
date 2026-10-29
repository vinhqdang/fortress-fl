"""
Paired ghost canaries for federated unlearning audits.

Each client privately draws m twin pairs of exchangeable random-label records
before training and commits to the manifest. A fair coin decides which twin
joins the client's training data; the other (the ghost) is never used. After a
revocation wave the verifier compares how the released model scores the in-twin
and its ghost. If the revoked clients' influence has been removed, the sign of
that difference is a fair coin, which is the exact null used by VOUCH
(Dang, Machine Learning, 2026) and by the e-process verifier called below.
"""

import hashlib
import json
from typing import Dict, List, Sequence

import numpy as np

from .models import sigmoid as _sigmoid


def _twin(rng, dim, scale):
    x = rng.normal(0.0, scale, size=dim)
    x[-1] = 1.0                      # bias column of the feature map
    return x, float(rng.random() < 0.5)


def _twin_crossover(rng, X):
    """Features mixed per coordinate from two random real records, random label:
    in-distribution canaries that an outlier detector cannot single out."""
    a, b = X[rng.integers(len(X))], X[rng.integers(len(X))]
    x = np.where(rng.random(len(a)) < 0.5, a, b).astype(float)
    x[-1] = 1.0
    return x, float(rng.random() < 0.5)


def plant_canaries(clients: List[Dict], pairs_per_client: int, dim: int,
                   seed: int = 0, scale: float = 1.0, kind: str = "gaussian") -> Dict[str, Dict]:
    """
    Append the in-twin of every pair to its client's data and return the
    per-client manifests (kept by the client; only their digest is published).
    """
    manifests = {}
    for c in clients:
        rng = np.random.default_rng([seed, int(c["id"][1:])])
        pairs = []
        Xin, yin = [], []
        real_X = c["X"].copy()
        for _ in range(pairs_per_client):
            if kind == "crossover":
                a = _twin_crossover(rng, real_X)
                b = _twin_crossover(rng, real_X)
            else:
                a = _twin(rng, dim, scale)
                b = _twin(rng, dim, scale)
            coin = int(rng.random() < 0.5)
            inn, gho = (a, b) if coin == 0 else (b, a)
            pairs.append({"in": (inn[0], inn[1]), "ghost": (gho[0], gho[1])})
            Xin.append(inn[0]); yin.append(inn[1])
        c["X"] = np.vstack([c["X"], np.array(Xin)])
        c["y"] = np.concatenate([c["y"], np.array(yin)])
        if "latent" in c:
            c["latent"] = np.concatenate([c["latent"], -np.ones(len(Xin), int)])
        c["n_canary"] = pairs_per_client
        digest = hashlib.sha256(json.dumps(
            [[p["in"][0].round(6).tolist(), p["in"][1], p["ghost"][0].round(6).tolist(),
              p["ghost"][1]] for p in pairs]).encode()).hexdigest()
        manifests[c["id"]] = {"pairs": pairs, "sha256": digest}
    return manifests


def commit_manifests(manifests: Dict[str, Dict]) -> Dict[str, str]:
    return {cid: m["sha256"] for cid, m in manifests.items()}


def _logit(theta, x, model=None):
    return float(x @ theta) if model is None else float(model.logit(theta, x[None, :])[0])


def pair_scores(theta: np.ndarray, pair: Dict, model=None) -> Dict[str, float]:
    """
    Score differences D = s(in) - s(ghost) for the declared score class. The 'loss'
    score is the logistic loss (lower for a memorised record, so D<0 signals leakage)
    and 'margin' is the signed logit margin (higher for a memorised record, D>0). The
    verifier tests both directions, so the sign convention does not affect validity.
    """
    out = {}
    for name in ("loss", "margin"):
        s = []
        for key in ("in", "ghost"):
            x, y = pair[key]
            z = _logit(theta, x, model)
            if name == "loss":
                p = float(np.clip(_sigmoid(np.array(z)), 1e-12, 1 - 1e-12))
                s.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
            else:
                s.append((2 * y - 1) * z)
        out[name] = s[0] - s[1]
    return out


def cohort_diffs(theta: np.ndarray, manifests: Dict[str, Dict],
                 client_ids: Sequence[str], model=None,
                 revealed: Dict[str, Sequence[int]] = None, missing: str = "worst"
                 ) -> List[Dict[str, float]]:
    """
    Score differences for the committed cohort. ``revealed`` lists, per client, the
    indices of pairs the client actually opened. A committed pair that is not opened
    is scored as the worst case (the in-twin looks memorised, D>0) when
    ``missing="worst"``, so withholding ghosts can only hurt the revoking client; with
    ``missing="drop"`` it is skipped, which is exploitable (see the cherry-pick attack).
    """
    out = []
    for cid in client_ids:
        pairs = manifests[cid]["pairs"]
        idx = range(len(pairs)) if revealed is None else revealed.get(cid, [])
        shown = set(idx)
        for i, p in enumerate(pairs):
            if i in shown:
                out.append(pair_scores(theta, p, model))
            elif missing == "worst":
                out.append({"loss": 1e6, "margin": 1e6})
    return out
