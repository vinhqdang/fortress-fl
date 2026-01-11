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
        elif self.method == 'centered_clipping':
            return aggregate_centered_clipping(gradients)
        elif self.method == 'rfa':
            return aggregate_geometric_median(gradients)
        elif self.method == 'foundationfl_median':
            return aggregate_foundation_fl(gradients, base_method='median')
        elif self.method == 'foundationfl_trimmed_mean':
            return aggregate_foundation_fl(gradients, base_method='trimmed_mean')
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")


def aggregate_foundation_fl(gradients: List[np.ndarray], base_method: str = 'median', scalar_m: float = 0.5) -> np.ndarray:
    """
    FoundationFL Aggregation (Fang et al., NDSS 2025).
    
    1. Identify 'extreme' updates (max and min per dimension).
    2. Score each update based on distance to extremes (Equation 4).
       Score = min( ||g - g_max||, ||g - g_min|| )
       Higher score = further from extremes = more central.
    3. Select best update i* with max score.
    4. Generate m synthetic updates = g_i*.
    5. Aggregate all (n + m) updates using base_method.
    
    Args:
        gradients: List of client gradients
        base_method: 'median' or 'trimmed_mean'
        scalar_m: Fraction of n to use for m (m = scalar_m * n)
    """
    if not gradients:
        raise ValueError("Cannot aggregate empty gradient list")
        
    n = len(gradients)
    m = int(n * scalar_m)
    stacked_grads = np.stack(gradients) # (n, d)
    
    # 1. Calc Extremes
    g_max = np.max(stacked_grads, axis=0) # (d,)
    g_min = np.min(stacked_grads, axis=0) # (d,)
    
    # 2. Score Updates
    scores = []
    for g in gradients:
        # Distance to max and min
        dist_max = np.linalg.norm(g - g_max)
        dist_min = np.linalg.norm(g - g_min)
        # Score is min distance (we want to be far from BOTH extremes)
        # Wait, if we are close to g_max, dist_max is small.
        # We want to maximize score, so we want to maximize the MIN distance.
        # This effectively finds the "center" of the bounding box defined by g_max, g_min?
        # Yes, standard Maximin strategy.
        scores.append(min(dist_max, dist_min))
        
    # 3. Select Best
    best_idx = np.argmax(scores)
    g_star = gradients[best_idx]
    
    # 4. Generate Synthetic Updates
    synthetic_updates = [g_star for _ in range(m)]
    
    # 5. Aggregate
    combined_gradients = gradients + synthetic_updates
    
    if base_method == 'median':
        return aggregate_median(combined_gradients)
    elif base_method == 'trimmed_mean':
        return aggregate_trimmed_mean(combined_gradients)
    else:
        raise ValueError(f"Unknown base method for FoundationFL: {base_method}")



def aggregate_centered_clipping(gradients: List[np.ndarray], tau: float = 1.0) -> np.ndarray:
    """
    Centered Clipping Aggregation.
    Clip updates that are too far from the median.
    """
    if not gradients:
        raise ValueError("Cannot aggregate empty gradient list")
        
    stacked_grads = np.stack(gradients)
    
    # 1. Compute reference point (coordinate-wise median)
    median_vect = np.median(stacked_grads, axis=0)
    
    # 2. Clip updates
    clipped_grads = []
    for grad in gradients:
        diff = grad - median_vect
        norm = np.linalg.norm(diff)
        if norm > tau:
            clipped = median_vect + diff * (tau / norm)
        else:
            clipped = grad
        clipped_grads.append(clipped)
        
    # 3. Average
    return np.mean(clipped_grads, axis=0)


def aggregate_geometric_median(gradients: List[np.ndarray], max_iter: int = 100, tol: float = 1e-5) -> np.ndarray:
    """
    Robust Federated Aggregation (RFA) using Geometric Median (Weiszfeld's algorithm).
    Minimizes sum of Euclidean distances to all points.
    """
    if not gradients:
        raise ValueError("Cannot aggregate empty gradient list")
        
    points = np.stack(gradients)
    
    # Initial guess: coordinate-wise median
    estimate = np.median(points, axis=0)
    
    for _ in range(max_iter):
        distances = np.linalg.norm(points - estimate, axis=1)
        
        # Avoid division by zero
        zeros = distances < 1e-9
        distances[zeros] = 1e-9
        
        weights = 1.0 / distances
        weights = weights / weights.sum()
        
        new_estimate = np.sum(points * weights[:, None], axis=0)
        
        if np.linalg.norm(new_estimate - estimate) < tol:
            estimate = new_estimate
            break
            
        estimate = new_estimate
        
    return estimate
