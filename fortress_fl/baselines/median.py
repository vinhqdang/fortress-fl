"""
Coordinate-wise Median - Simple Byzantine-robust aggregation method
Based on: Yin et al. "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class CoordinateWiseMedian:
    """Coordinate-wise median Byzantine-robust aggregation method."""

    def __init__(self):
        self.name = "CoordinateWiseMedian"
        self.is_byzantine_robust = True

    def aggregate(self, gradients: List[np.ndarray],
                  operator_weights: Optional[List[float]] = None,
                  **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aggregate gradients using coordinate-wise median.

        Args:
            gradients: List of gradient arrays from operators
            operator_weights: Ignored for coordinate-wise median

        Returns:
            Tuple of (aggregated_gradient, info_dict)
        """
        if len(gradients) == 0:
            raise ValueError("No gradients provided")

        # Stack gradients for coordinate-wise operations
        gradient_matrix = np.stack(gradients, axis=0)  # Shape: (n_operators, gradient_dim)

        # Compute coordinate-wise median
        aggregated = np.median(gradient_matrix, axis=0)

        # Simple Byzantine detection: operators far from median
        byzantine_detected = []
        median_distances = []

        for i, gradient in enumerate(gradients):
            distance = np.linalg.norm(gradient - aggregated)
            median_distances.append(distance)

        # Consider operators with distance > 2 * median distance as Byzantine
        if len(median_distances) > 0:
            median_dist_threshold = 2.0 * np.median(median_distances)
            for i, dist in enumerate(median_distances):
                if dist > median_dist_threshold:
                    byzantine_detected.append(i)

        info = {
            'method': 'CoordinateWiseMedian',
            'n_operators': len(gradients),
            'byzantine_detected': byzantine_detected,
            'detection_accuracy': None,  # Will be computed externally
            'median_distances': median_distances,
            'detection_threshold': median_dist_threshold if len(median_distances) > 0 else None
        }

        return aggregated, info

    def train_round(self, operators_data: List[Dict],
                   global_model: np.ndarray,
                   learning_rate: float = 0.01,
                   **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute one training round with Coordinate-wise Median.

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