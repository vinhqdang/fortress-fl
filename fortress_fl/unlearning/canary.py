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

from .system import _sigmoid


def _twin(rng, dim, scale):
    x = rng.normal(0.0, scale, size=dim)
    x[-1] = 1.0                      # bias column of the feature map
    return x, float(rng.random() < 0.5)


def plant_canaries(clients: List[Dict], pairs_per_client: int, dim: int,
                   seed: int = 0, scale: float = 1.0) -> Dict[str, Dict]:
    """
    Append the in-twin of every pair to its client's data and return the
    per-client manifests (kept by the client; only their digest is published).
    """
    manifests = {}
    for c in clients:
        rng = np.random.default_rng([seed, int(c["id"][1:])])
        pairs = []
        Xin, yin = [], []
        for _ in range(pairs_per_client):
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


def _logit(theta, x):
    return float(x @ theta)


def pair_scores(theta: np.ndarray, pair: Dict) -> Dict[str, float]:
    """Score differences D = s(in) - s(ghost) for the declared score class."""
    out = {}
    for name in ("loss", "margin"):
        s = []
        for key in ("in", "ghost"):
            x, y = pair[key]
            z = _logit(theta, x)
            if name == "loss":
                p = float(np.clip(_sigmoid(np.array(z)), 1e-12, 1 - 1e-12))
                s.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
            else:
                s.append((2 * y - 1) * z)
        out[name] = s[0] - s[1]
    return out


def cohort_diffs(theta: np.ndarray, manifests: Dict[str, Dict],
                 client_ids: Sequence[str]) -> List[Dict[str, float]]:
    return [pair_scores(theta, p) for cid in client_ids
            for p in manifests[cid]["pairs"]]
