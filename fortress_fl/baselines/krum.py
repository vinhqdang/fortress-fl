"""
Krum - Byzantine-robust aggregation method
Based on: Blanchard et al. "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

class Krum:
    """Krum Byzantine-robust aggregation method."""

    def __init__(self, n_byzantine: int = 0):
        self.name = "Krum"
        self.is_byzantine_robust = True
        self.n_byzantine = n_byzantine

    def compute_krum_scores(self, gradients: List[np.ndarray]) -> List[float]:
        """
        Compute Krum scores for each gradient.

        Args:
            gradients: List of gradient arrays

        Returns:
            List of Krum scores (lower is better)
        """
        n = len(gradients)
        if n <= 2 * self.n_byzantine:
            raise ValueError(f"Need more than 2f operators: got {n}, f={self.n_byzantine}")

        scores = []
        k = n - self.n_byzantine - 2  # Number of closest neighbors to consider

        for i in range(n):
            # Compute distances to all other gradients
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(gradients[i] - gradients[j]) ** 2
                    distances.append(dist)

            # Sort distances and sum the k closest
            distances.sort()
            score = sum(distances[:k])
            scores.append(score)

        return scores

    def aggregate(self, gradients: List[np.ndarray],
                  operator_weights: Optional[List[float]] = None,
                  **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aggregate gradients using Krum.

        Args:
            gradients: List of gradient arrays from operators
            operator_weights: Ignored for Krum

        Returns:
            Tuple of (aggregated_gradient, info_dict)
        """
        if len(gradients) == 0:
            raise ValueError("No gradients provided")

        if len(gradients) <= 2 * self.n_byzantine:
            # Fall back to simple averaging if not enough operators
            logging.warning(f"Krum: Not enough operators ({len(gradients)}) for f={self.n_byzantine}, falling back to averaging")
            aggregated = np.mean(gradients, axis=0)
            selected_indices = list(range(len(gradients)))
        else:
            # Compute Krum scores
            scores = self.compute_krum_scores(gradients)

            # Select gradient with minimum score
            selected_idx = np.argmin(scores)
            aggregated = gradients[selected_idx]
            selected_indices = [selected_idx]

        # Identify Byzantine operators (those not selected)
        byzantine_detected = [i for i in range(len(gradients)) if i not in selected_indices]

        info = {
            'method': 'Krum',
            'n_operators': len(gradients),
            'selected_operators': selected_indices,
            'byzantine_detected': byzantine_detected,
            'detection_accuracy': None,  # Will be computed externally
            'krum_scores': scores if len(gradients) > 2 * self.n_byzantine else None
        }

        return aggregated, info

    def train_round(self, operators_data: List[Dict],
                   global_model: np.ndarray,
                   learning_rate: float = 0.01,
                   **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute one training round with Krum.

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