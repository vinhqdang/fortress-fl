"""
FABA - Fast Adaptive Byzantine-robust Aggregation
Based on: Li et al. "RSA: Byzantine-Robust Stochastic Aggregation Methods for Distributed Learning"
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class FABA:
    """FABA (Fast Adaptive Byzantine-robust Aggregation) method."""

    def __init__(self, n_byzantine: int = 0, alpha: float = 0.5):
        self.name = "FABA"
        self.is_byzantine_robust = True
        self.n_byzantine = n_byzantine
        self.alpha = alpha  # Filtering parameter

    def compute_pairwise_distances(self, gradients: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise distances between gradients.

        Args:
            gradients: List of gradient arrays

        Returns:
            Distance matrix
        """
        n = len(gradients)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(gradients[i] - gradients[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def filter_gradients(self, gradients: List[np.ndarray]) -> List[int]:
        """
        Filter gradients based on pairwise distances and clustering.

        Args:
            gradients: List of gradient arrays

        Returns:
            List of selected indices
        """
        n = len(gradients)
        if n <= 2 * self.n_byzantine:
            return list(range(n))

        # Compute pairwise distances
        distances = self.compute_pairwise_distances(gradients)

        # For each gradient, find its k nearest neighbors
        k = n - self.n_byzantine - 1
        selected_indices = []

        for i in range(n):
            # Get distances to other gradients
            dists_to_i = distances[i]
            # Find k nearest neighbors (excluding self)
            neighbor_indices = np.argsort(dists_to_i)[1:k+1]

            # Compute average distance to k nearest neighbors
            avg_dist = np.mean(dists_to_i[neighbor_indices])
            selected_indices.append((i, avg_dist))

        # Select gradients with smallest average distances to neighbors
        selected_indices.sort(key=lambda x: x[1])
        n_select = n - self.n_byzantine
        selected = [idx for idx, _ in selected_indices[:n_select]]

        return selected

    def aggregate(self, gradients: List[np.ndarray],
                  operator_weights: Optional[List[float]] = None,
                  **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aggregate gradients using FABA.

        Args:
            gradients: List of gradient arrays from operators
            operator_weights: Optional weights for selected operators

        Returns:
            Tuple of (aggregated_gradient, info_dict)
        """
        if len(gradients) == 0:
            raise ValueError("No gradients provided")

        n = len(gradients)

        # Filter gradients to remove Byzantine ones
        selected_indices = self.filter_gradients(gradients)
        selected_gradients = [gradients[i] for i in selected_indices]

        # Weighted average of selected gradients
        if operator_weights is not None:
            selected_weights = [operator_weights[i] for i in selected_indices]
            total_weight = sum(selected_weights)
            if total_weight > 0:
                selected_weights = [w / total_weight for w in selected_weights]
            else:
                selected_weights = [1.0 / len(selected_gradients)] * len(selected_gradients)
        else:
            selected_weights = [1.0 / len(selected_gradients)] * len(selected_gradients)

        # Compute weighted average
        aggregated = np.zeros_like(gradients[0])
        for i, gradient in enumerate(selected_gradients):
            aggregated += selected_weights[i] * gradient

        # Identify Byzantine operators
        byzantine_detected = [i for i in range(n) if i not in selected_indices]

        info = {
            'method': 'FABA',
            'n_operators': n,
            'selected_operators': selected_indices,
            'byzantine_detected': byzantine_detected,
            'detection_accuracy': None,  # Will be computed externally
            'aggregation_weights': selected_weights,
            'alpha': self.alpha
        }

        return aggregated, info

    def train_round(self, operators_data: List[Dict],
                   global_model: np.ndarray,
                   learning_rate: float = 0.01,
                   **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute one training round with FABA.

        Args:
            operators_data: List of operator data dictionaries
            global_model: Current global model parameters
            learning_rate: Learning rate for gradient computation

        Returns:
            Tuple of (updated_model, round_info)
        """
        from ..core.training import compute_local_gradient, generate_byzantine_gradient

        gradients = []
        operator_info = []

        for i, operator in enumerate(operators_data):
            if operator.get('is_byzantine', False):
                # Generate Byzantine gradient based on attack type
                attack_type = operator.get('attack_type', 'sign_flip')
                honest_gradient = compute_local_gradient(
                    operator['dataset'], global_model, learning_rate
                )
                gradient = generate_byzantine_gradient(
                    honest_gradient, attack_type
                )
            else:
                # Compute honest gradient
                gradient = compute_local_gradient(
                    operator['dataset'], global_model, learning_rate
                )

            gradients.append(gradient)
            operator_info.append({
                'operator_id': operator['id'],
                'is_byzantine': operator.get('is_byzantine', False),
                'attack_type': operator.get('attack_type', None)
            })

        # Aggregate gradients
        aggregated_gradient, agg_info = self.aggregate(gradients)

        # Update global model
        updated_model = global_model + aggregated_gradient

        round_info = {
            'operators': operator_info,
            'aggregation': agg_info,
            'gradient_norm': np.linalg.norm(aggregated_gradient),
            'model_norm': np.linalg.norm(updated_model)
        }

        return updated_model, round_info