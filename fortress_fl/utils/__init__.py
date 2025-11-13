"""Utility functions for FORTRESS-FL."""

from .attacks import (
    ByzantineAttack,
    SignFlipAttack,
    RandomAttack,
    ModelPoisoningAttack,
    CoordinatedAttack,
    AdaptiveAttack
)

from .evaluation import (
    MetricsTracker,
    evaluate_convergence,
    evaluate_byzantine_robustness,
    evaluate_privacy_utility_tradeoff,
    plot_training_metrics,
    generate_performance_report
)

__all__ = [
    'ByzantineAttack',
    'SignFlipAttack',
    'RandomAttack',
    'ModelPoisoningAttack',
    'CoordinatedAttack',
    'AdaptiveAttack',
    'MetricsTracker',
    'evaluate_convergence',
    'evaluate_byzantine_robustness',
    'evaluate_privacy_utility_tradeoff',
    'plot_training_metrics',
    'generate_performance_report'
]