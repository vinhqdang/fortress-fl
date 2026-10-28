"""
Paired real-record twins.

Instead of planting synthetic canaries, a client pairs records of its own data at
random. A fair coin decides which record of each pair is trained on; the other, the
ghost, is withheld. The twins are exchangeable by construction of the random
matching, so the same sign test applies, but now the audit measures what the model
retained about real records of the revoked clients. Cost: ghosts are not trained on.
"""

import hashlib
import json
from typing import Dict, List

import numpy as np


def plant_real_twins(clients: List[Dict], pair_fraction: float = 0.5, seed: int = 0) -> Dict[str, Dict]:
    """
    Pair ``pair_fraction`` of each client's records at random, keep one twin per pair
    in its training data and withhold the other. Returns manifests with the same
    layout as ``plant_canaries`` so that the rest of the pipeline is unchanged.
    """
    manifests = {}
    for c in clients:
        rng = np.random.default_rng([seed, 7919, int(c["id"][1:])])
        n = len(c["y"])
        n_pairs = int(pair_fraction * n) // 2
        perm = rng.permutation(n)
        a_idx, b_idx = perm[:n_pairs], perm[n_pairs:2 * n_pairs]
        coins = rng.random(n_pairs) < 0.5
        in_idx = np.where(coins, a_idx, b_idx)
        gh_idx = np.where(coins, b_idx, a_idx)
        pairs = [{"in": (c["X"][i].copy(), float(c["y"][i])),
                  "ghost": (c["X"][g].copy(), float(c["y"][g]))}
                 for i, g in zip(in_idx, gh_idx)]
        keep = np.ones(n, bool)
        keep[gh_idx] = False
        c["X"], c["y"] = c["X"][keep], c["y"][keep]
        if "latent" in c:
            c["latent"] = c["latent"][keep]
        c["n_canary"] = 0
        digest = hashlib.sha256(json.dumps(
            [[p["in"][0].round(5).tolist(), p["in"][1], p["ghost"][0].round(5).tolist(),
              p["ghost"][1]] for p in pairs]).encode()).hexdigest()
        manifests[c["id"]] = {"pairs": pairs, "sha256": digest}
    return manifests
