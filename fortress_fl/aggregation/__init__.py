"""Aggregation algorithms for FORTRESS-FL."""

from .spectral_clustering import (
    compute_similarity_matrix,
    spectral_clustering_byzantine_detection,
    identify_byzantine_cluster,
    filter_byzantine_gradients
)

from .reputation import (
    compute_quality_score,
    update_reputations,
    select_operators_by_reputation
)

from .trustchain import trustchain_aggregation

__all__ = [
    'compute_similarity_matrix',
    'spectral_clustering_byzantine_detection',
    'identify_byzantine_cluster',
    'filter_byzantine_gradients',
    'compute_quality_score',
    'update_reputations',
    'select_operators_by_reputation',
    'trustchain_aggregation'
]