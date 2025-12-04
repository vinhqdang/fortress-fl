"""
Baseline Aggregation Algorithms for Comparison

This module implements standard robust aggregation algorithms to serve as
baselines for evaluating FORTRESS-FL performance.

Baselines included:
1. FedAvg (Standard non-robust aggregation)
2. Krum (Blanchard et al., NeurIPS 2017)
3. Coordinate-wise Median (Yin et al., ICML 2018)
4. Trimmed Mean (Yin et al., ICML 2018)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def aggregate_fedavg(gradients: List[np.ndarray]) -> np.ndarray:
    """
    Standard Federated Averaging (mean aggregation).
    Not robust to Byzantine attacks.
    """
    if not gradients:
        raise ValueError("Cannot aggregate empty gradient list")
    
    return np.mean(gradients, axis=0)


def aggregate_krum(gradients: List[np.ndarray], f: int) -> np.ndarray:
    """
    Krum aggregation algorithm.
    Selects the gradient that minimizes the sum of squared distances to its
    n - f - 2 closest neighbors.
    
    Args:
        gradients: List of gradient vectors
        f: Number of Byzantine workers to tolerate
        
    Returns:
        Selected gradient (one of the input gradients)
    """
    n = len(gradients)
    if n <= 2 * f + 2:
        # Krum requires n > 2f + 2
        # Fallback to mean if condition not met (or warn)
        return np.mean(gradients, axis=0)
        
    scores = []
    
    # Compute pairwise distances
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(gradients[i] - gradients[j]) ** 2
            dists.append(dist)
            
        # Sum of distances to n - f - 2 closest neighbors
        dists.sort()
        # We want n - f - 2 neighbors, so take slice [:n-f-2]
        # Note: dists has length n-1 (excludes self)
        k = n - f - 2
        score = sum(dists[:k])
        scores.append(score)
        
    # Select gradient with minimum score
    best_idx = np.argmin(scores)
    return gradients[best_idx]


def aggregate_median(gradients: List[np.ndarray]) -> np.ndarray:
    """
    Coordinate-wise Median aggregation.
    Robust to Byzantine attacks but may lose statistical efficiency.
    
    Args:
        gradients: List of gradient vectors
        
    Returns:
        Coordinate-wise median vector
    """
    if not gradients:
        raise ValueError("Cannot aggregate empty gradient list")
        
    # Stack gradients: shape (n, d)
    stacked_grads = np.stack(gradients)
    
    # Compute median along operator axis (axis 0)
    return np.median(stacked_grads, axis=0)


def aggregate_trimmed_mean(gradients: List[np.ndarray], beta: float = 0.1) -> np.ndarray:
    """
    Coordinate-wise Trimmed Mean aggregation.
    Removes the largest and smallest beta fraction of values in each coordinate.
    
    Args:
        gradients: List of gradient vectors
        beta: Fraction of values to trim from each end (0 < beta < 0.5)
        
    Returns:
        Coordinate-wise trimmed mean vector
    """
    if not gradients:
        raise ValueError("Cannot aggregate empty gradient list")
        
    n = len(gradients)
    k = int(n * beta)
    
    if k == 0:
        return np.mean(gradients, axis=0)
        
    # Stack gradients: shape (n, d)
    stacked_grads = np.stack(gradients)
    
    # Sort along operator axis
    sorted_grads = np.sort(stacked_grads, axis=0)
    
    # Trim k smallest and k largest
    trimmed = sorted_grads[k:n-k, :]
    
    # Compute mean of remaining
    return np.mean(trimmed, axis=0)


class BaselineAggregator:
    """Wrapper for baseline aggregation algorithms."""
    
    def __init__(self, method: str = 'fedavg', f: int = 0, beta: float = 0.1):
        self.method = method.lower()
        self.f = f
        self.beta = beta
        
    def aggregate(self, gradients: List[np.ndarray]) -> np.ndarray:
        """Perform aggregation using selected method."""
        if self.method == 'fedavg':
            return aggregate_fedavg(gradients)
        elif self.method == 'krum':
            return aggregate_krum(gradients, self.f)
        elif self.method == 'median':
            return aggregate_median(gradients)
        elif self.method == 'trimmed_mean':
            return aggregate_trimmed_mean(gradients, self.beta)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
