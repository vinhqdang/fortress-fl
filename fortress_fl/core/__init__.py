"""Core FORTRESS-FL implementation."""

from .fortress_fl import FortressFL
from .training import (
    train_fortress_fl,
    create_federated_datasets,
    generate_synthetic_data,
    compute_local_gradient,
    generate_byzantine_gradient
)

__all__ = [
    'FortressFL',
    'train_fortress_fl',
    'create_federated_datasets',
    'generate_synthetic_data',
    'compute_local_gradient',
    'generate_byzantine_gradient'
]