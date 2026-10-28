"""Parametric models trained by the federated unlearning system.

Parameters are one flat vector so that updates, commitments, ledger digests and
rollback treat every model identically. Both models are binary classifiers with
the logistic loss; the last input feature is a constant bias column.
"""

import numpy as np


def sigmoid(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


class LinearModel:
    """Logistic regression (a linear head on fixed features)."""

    def __init__(self, d_in: int):
        self.d_in = d_in
        self.n_params = d_in
        self.flops_per_sample = 4.0 * d_in

    def init(self, seed: int = 0) -> np.ndarray:
        return np.zeros(self.n_params)

    def logit(self, theta, X):
        return X @ theta

    def grad(self, theta, X, y):
        return X.T @ (sigmoid(X @ theta) - y) / len(y)


class MLPModel:
    """One hidden tanh layer, scalar logit."""

    def __init__(self, d_in: int, hidden: int = 32):
        self.d_in, self.h = d_in, hidden
        self.n_params = d_in * hidden + hidden + hidden + 1
        self.flops_per_sample = 6.0 * self.n_params     # fwd + bwd

    def _unpack(self, theta):
        d, h = self.d_in, self.h
        W1 = theta[:d * h].reshape(d, h)
        b1 = theta[d * h:d * h + h]
        w2 = theta[d * h + h:d * h + 2 * h]
        b2 = theta[-1]
        return W1, b1, w2, b2

    def init(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(10_000 + seed)
        d, h = self.d_in, self.h
        return np.concatenate([rng.normal(0, 1 / np.sqrt(d), d * h), np.zeros(h),
                               rng.normal(0, 1 / np.sqrt(h), h), np.zeros(1)])

    def logit(self, theta, X):
        W1, b1, w2, b2 = self._unpack(theta)
        return np.tanh(X @ W1 + b1) @ w2 + b2

    def grad(self, theta, X, y):
        W1, b1, w2, b2 = self._unpack(theta)
        Z = np.tanh(X @ W1 + b1)
        delta = (sigmoid(Z @ w2 + b2) - y) / len(y)
        gw2 = Z.T @ delta
        dZ = np.outer(delta, w2) * (1.0 - Z ** 2)
        return np.concatenate([(X.T @ dZ).ravel(), dZ.sum(0), gw2, [delta.sum()]])
