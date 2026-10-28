"""Federated data partitions with latent classes for unlearning experiments."""

from typing import Dict, List, Tuple

import numpy as np


def synthetic_latent(n_total: int, dim: int, n_latent: int = 10, sep: float = 1.2,
                     label_noise: float = 0.08, seed: int = 0):
    """Ten Gaussian blobs; the binary label is the parity of the blob index."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, size=(n_latent, dim))
    k = rng.integers(0, n_latent, n_total)
    X = centers[k] + rng.normal(0, 1.0, size=(n_total, dim))
    y = (k % 2).astype(float)
    flip = rng.random(n_total) < label_noise
    return X, y, k, flip


def digits_latent(seed: int = 0):
    """scikit-learn digits (bundled with the library); label = digit parity."""
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.data / 16.0
    k = d.target
    return X, (k % 2).astype(float), k


def dirichlet_partition(latent: np.ndarray, n_clients: int, alpha: float,
                        seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n_clients)]
    for k in np.unique(latent):
        idx = np.where(latent == k)[0]
        rng.shuffle(idx)
        p = rng.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for c, chunk in enumerate(np.split(idx, cuts)):
            parts[c].extend(chunk.tolist())
    return [np.array(sorted(p), dtype=int) for p in parts]


def build_federation(dataset: str = "synthetic", n_clients: int = 20,
                     n_per_client: int = 50, dim: int = 60, alpha: float = 1.0,
                     n_test: int = 2000, seed: int = 0,
                     byz_clients: Tuple[int, ...] = (), attack: str = "sign_flip",
                     min_samples: int = 8):
    """
    Returns (clients, test, info). Each client dict has id, X, y, latent and the
    bias column is appended to X. Test data is clean and disjoint from training.
    """
    if dataset == "synthetic":
        n_total = n_clients * n_per_client + n_test
        X, y, k, flip = synthetic_latent(n_total, dim, seed=seed)
        ytr = np.where(flip, 1 - y, y)
        tr = slice(0, n_clients * n_per_client)
        te = slice(n_clients * n_per_client, n_total)
        Xtr, ytr, ktr = X[tr], ytr[tr], k[tr]
        Xte, yte, kte = X[te], y[te], k[te]
    else:
        X, y, k = digits_latent(seed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y))
        n_te = 400
        te_idx, tr_idx = perm[:n_te], perm[n_te:]
        Xtr, ytr, ktr = X[tr_idx], y[tr_idx], k[tr_idx]
        Xte, yte, kte = X[te_idx], y[te_idx], k[te_idx]
        dim = X.shape[1]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    add_bias = lambda A: np.hstack([A, np.ones((len(A), 1))])
    Xtr, Xte = add_bias(Xtr), add_bias(Xte)

    parts = dirichlet_partition(ktr, n_clients, alpha, seed)
    # make sure nobody is (almost) empty: move samples from the largest client
    for c in range(n_clients):
        while len(parts[c]) < min_samples:
            big = int(np.argmax([len(p) for p in parts]))
            parts[c] = np.append(parts[c], parts[big][-1])
            parts[big] = parts[big][:-1]
    clients = []
    for c, idx in enumerate(parts):
        clients.append({"id": f"C{c:02d}", "X": Xtr[idx], "y": ytr[idx],
                        "latent": ktr[idx], "byzantine": c in byz_clients,
                        "attack": attack if c in byz_clients else None})
    test = {"X": Xte, "y": yte, "latent": kte}
    return clients, test, {"dim": Xtr.shape[1]}


def add_backdoor(client: Dict, fraction: float, trigger_value: float = 4.0,
                 target: float = 1.0, n_trigger: int = 3, seed: int = 0):
    """Data-poisoning backdoor: stamp a trigger on features, relabel to target."""
    rng = np.random.default_rng(seed)
    n = len(client["y"])
    m = max(1, int(fraction * n))
    idx = rng.choice(n, m, replace=False)
    client["X"] = client["X"].copy()
    client["y"] = client["y"].copy()
    client["X"][idx, :n_trigger] = trigger_value
    client["y"][idx] = target
    client["poison_idx"] = idx
    return client


def backdoor_test_set(test: Dict, trigger_value: float = 4.0, target: float = 1.0,
                      n_trigger: int = 3):
    keep = test["y"] != target
    X = test["X"][keep].copy()
    X[:, :n_trigger] = trigger_value
    return {"X": X, "y_target": np.full(len(X), target)}
