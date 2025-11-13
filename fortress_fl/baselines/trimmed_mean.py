"""
Trimmed Mean - Byzantine-robust aggregation method
Based on: Yin et al. "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

class TrimmedMean:
    """Trimmed Mean Byzantine-robust aggregation method."""

    def __init__(self, n_byzantine: int = 0):
        self.name = "TrimmedMean"
        self.is_byzantine_robust = True
        self.n_byzantine = n_byzantine

    def aggregate(self, gradients: List[np.ndarray],
                  operator_weights: Optional[List[float]] = None,
                  **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aggregate gradients using coordinate-wise trimmed mean.

        Args:
            gradients: List of gradient arrays from operators
            operator_weights: Ignored for Trimmed Mean

        Returns:
            Tuple of (aggregated_gradient, info_dict)
        """
        if len(gradients) == 0:
            raise ValueError("No gradients provided")

        n = len(gradients)
        if n <= 2 * self.n_byzantine:
            logging.warning(f"TrimmedMean: Not enough operators ({n}) for f={self.n_byzantine}, using all")
            trim_count = 0
        else:
            trim_count = self.n_byzantine

        # Stack gradients for coordinate-wise operations
        gradient_matrix = np.stack(gradients, axis=0)  # Shape: (n_operators, gradient_dim)

        # Apply trimmed mean coordinate-wise
        aggregated = np.zeros_like(gradients[0])
        removed_operators = set()

        for coord_idx in range(gradient_matrix.shape[1]):
            coord_values = gradient_matrix[:, coord_idx]

            if trim_count > 0:
                # Sort coordinate values and trim extremes
                sorted_indices = np.argsort(coord_values)

                # Remove trim_count smallest and largest values
                valid_indices = sorted_indices[trim_count:-trim_count]

                # Track which operators were removed
                removed_indices = set(sorted_indices[:trim_count]) | set(sorted_indices[-trim_count:])
                removed_operators.update(removed_indices)

                # Compute mean of remaining values
                aggregated[coord_idx] = np.mean(coord_values[valid_indices])
            else:
                # No trimming needed
                aggregated[coord_idx] = np.mean(coord_values)

        # Identify Byzantine operators (those frequently removed)
        # An operator is considered Byzantine if it was trimmed in more than 50% of coordinates
        byzantine_detected = []
        coord_count = gradient_matrix.shape[1]

        for op_idx in range(n):
            times_removed = 0
            for coord_idx in range(coord_count):
                coord_values = gradient_matrix[:, coord_idx]
                if trim_count > 0:
                    sorted_indices = np.argsort(coord_values)
                    removed_indices = set(sorted_indices[:trim_count]) | set(sorted_indices[-trim_count:])
                    if op_idx in removed_indices:
                        times_removed += 1

            # If removed in more than 50% of coordinates, consider Byzantine
            if times_removed > coord_count * 0.5:
                byzantine_detected.append(op_idx)

        info = {
            'method': 'TrimmedMean',
            'n_operators': n,
            'trim_count': trim_count,
            'byzantine_detected': byzantine_detected,
            'detection_accuracy': None,  # Will be computed externally
            'removed_per_coordinate': trim_count * 2 if trim_count > 0 else 0
        }

        return aggregated, info

    def train_round(self, operators_data: List[Dict],
                   global_model: np.ndarray,
                   learning_rate: float = 0.01,
                   **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute one training round with Trimmed Mean.

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