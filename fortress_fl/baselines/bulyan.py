"""
Bulyan - Advanced Byzantine-robust aggregation method
Based on: Guerraoui et al. "The Hidden Vulnerability of Distributed Learning in Byzantium"
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

class Bulyan:
    """Bulyan Byzantine-robust aggregation method."""

    def __init__(self, n_byzantine: int = 0):
        self.name = "Bulyan"
        self.is_byzantine_robust = True
        self.n_byzantine = n_byzantine

    def krum_selection(self, gradients: List[np.ndarray], k: int) -> List[int]:
        """
        Select k gradients using Krum scores.

        Args:
            gradients: List of gradient arrays
            k: Number of gradients to select

        Returns:
            List of selected indices
        """
        n = len(gradients)
        if n <= 2 * self.n_byzantine:
            return list(range(min(k, n)))

        scores = []
        nb_closest = n - self.n_byzantine - 2

        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(gradients[i] - gradients[j]) ** 2
                    distances.append(dist)

            distances.sort()
            score = sum(distances[:nb_closest])
            scores.append(score)

        # Select k gradients with lowest Krum scores
        selected_indices = np.argsort(scores)[:k]
        return selected_indices.tolist()

    def trimmed_mean(self, gradients: List[np.ndarray], trim_count: int) -> np.ndarray:
        """
        Compute coordinate-wise trimmed mean.

        Args:
            gradients: List of gradient arrays
            trim_count: Number of values to trim from each end

        Returns:
            Trimmed mean gradient
        """
        if len(gradients) <= 2 * trim_count:
            return np.mean(gradients, axis=0)

        gradient_matrix = np.stack(gradients, axis=0)
        aggregated = np.zeros_like(gradients[0])

        for coord_idx in range(gradient_matrix.shape[1]):
            coord_values = gradient_matrix[:, coord_idx]
            sorted_values = np.sort(coord_values)

            # Remove trim_count smallest and largest
            trimmed_values = sorted_values[trim_count:-trim_count]
            aggregated[coord_idx] = np.mean(trimmed_values)

        return aggregated

    def aggregate(self, gradients: List[np.ndarray],
                  operator_weights: Optional[List[float]] = None,
                  **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aggregate gradients using Bulyan.

        Args:
            gradients: List of gradient arrays from operators
            operator_weights: Ignored for Bulyan

        Returns:
            Tuple of (aggregated_gradient, info_dict)
        """
        if len(gradients) == 0:
            raise ValueError("No gradients provided")

        n = len(gradients)

        # Bulyan requires n >= 4f + 3
        if n < 4 * self.n_byzantine + 3:
            logging.warning(f"Bulyan: Not enough operators ({n}) for f={self.n_byzantine}, falling back to median")
            gradient_matrix = np.stack(gradients, axis=0)
            aggregated = np.median(gradient_matrix, axis=0)
            selected_indices = list(range(n))
        else:
            # Step 1: Select n-2f gradients using Krum
            k = n - 2 * self.n_byzantine
            selected_indices = self.krum_selection(gradients, k)
            selected_gradients = [gradients[i] for i in selected_indices]

            # Step 2: Apply coordinate-wise trimmed mean
            aggregated = self.trimmed_mean(selected_gradients, self.n_byzantine)

        # Identify Byzantine operators
        byzantine_detected = [i for i in range(n) if i not in selected_indices]

        info = {
            'method': 'Bulyan',
            'n_operators': n,
            'selected_operators': selected_indices,
            'byzantine_detected': byzantine_detected,
            'detection_accuracy': None,  # Will be computed externally
            'krum_selected': len(selected_indices) if n >= 4 * self.n_byzantine + 3 else None
        }

        return aggregated, info

    def train_round(self, operators_data: List[Dict],
                   global_model: np.ndarray,
                   learning_rate: float = 0.01,
                   **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute one training round with Bulyan.

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