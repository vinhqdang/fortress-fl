"""
FedAvg - Standard Federated Averaging (no Byzantine robustness)
Based on: McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data"
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class FedAvg:
    """Standard FedAvg aggregation (vulnerable to Byzantine attacks)."""

    def __init__(self):
        self.name = "FedAvg"
        self.is_byzantine_robust = False

    def aggregate(self, gradients: List[np.ndarray],
                  operator_weights: Optional[List[float]] = None,
                  **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Standard weighted averaging of gradients.

        Args:
            gradients: List of gradient arrays from operators
            operator_weights: Optional weights for each operator (uniform if None)

        Returns:
            Tuple of (aggregated_gradient, info_dict)
        """
        if len(gradients) == 0:
            raise ValueError("No gradients provided")

        # Use uniform weights if not provided
        if operator_weights is None:
            operator_weights = [1.0 / len(gradients)] * len(gradients)
        else:
            # Normalize weights
            total_weight = sum(operator_weights)
            operator_weights = [w / total_weight for w in operator_weights]

        # Weighted average
        aggregated = np.zeros_like(gradients[0])
        for i, gradient in enumerate(gradients):
            aggregated += operator_weights[i] * gradient

        info = {
            'method': 'FedAvg',
            'n_operators': len(gradients),
            'byzantine_detected': [],
            'detection_accuracy': None,
            'aggregation_weights': operator_weights
        }

        return aggregated, info

    def train_round(self, operators_data: List[Dict],
                   global_model: np.ndarray,
                   learning_rate: float = 0.01,
                   **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute one training round with FedAvg.

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
                    global_model, operator['dataset'], learning_rate
                )
                gradient = generate_byzantine_gradient(
                    honest_gradient, attack_type
                )
            else:
                # Compute honest gradient
                gradient = compute_local_gradient(
                    global_model, operator['dataset'], learning_rate
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