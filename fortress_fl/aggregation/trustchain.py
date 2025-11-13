"""
TrustChain Aggregation Algorithm for FORTRESS-FL

Complete aggregation pipeline combining cryptographic verification, spectral clustering,
reputation weighting, and differential privacy.
"""

import numpy as np
from typing import List, Dict, Tuple

from ..crypto.pedersen import verify_commitment
from .spectral_clustering import spectral_clustering_byzantine_detection
from .reputation import update_reputations, compute_reputation_weights


def trustchain_aggregation(gradients: List[np.ndarray], commitments: List[int],
                          openings: List[int], reputations: Dict[str, float],
                          operator_ids: List[str], comm_params: Tuple[int, int, int, int],
                          dp_params: Dict[str, float]) -> Dict:
    """
    TrustChain aggregation with cryptographic verification, spectral clustering,
    reputation weighting, and differential privacy.

    Args:
        gradients: List of n gradient vectors [g_1, ..., g_n]
        commitments: List of n commitments [C_1, ..., C_n]
        openings: List of n opening randomness [r_1, ..., r_n]
        reputations: Dict {operator_id -> reputation score}
        operator_ids: List of n operator IDs
        comm_params: Tuple (p, q, g, h) for Pedersen commitments
        dp_params: Dict {'sigma': float, 'epsilon': float} for DP noise

    Returns:
        result: Dict {
            'aggregated_gradient': numpy array,
            'updated_reputations': dict,
            'honest_indices': list,
            'byzantine_indices': list,
            'honest_operator_ids': list,
            'weights': numpy array,
            'verification_results': list,
            'clustering_quality': dict
        }
    """
    n = len(gradients)
    p, q, g_gen, h_gen = comm_params
    sigma = dp_params['sigma']

    print(f"[TrustChain] Starting aggregation for {n} operators...")

    # ===== PHASE 1: CRYPTOGRAPHIC VERIFICATION =====
    print(f"[TrustChain] Phase 1: Verifying {n} commitments...")
    verified_indices = []
    invalid_indices = []
    verification_results = []

    for i in range(n):
        is_valid = verify_commitment(gradients[i], commitments[i],
                                   openings[i], p, q, g_gen, h_gen)
        verification_results.append(is_valid)

        if is_valid:
            verified_indices.append(i)
        else:
            invalid_indices.append(i)
            print(f"[TrustChain] WARNING: Operator {operator_ids[i]} failed commitment verification!")

    # Penalize invalid operators immediately
    updated_reputations = reputations.copy()
    for i in invalid_indices:
        op_id = operator_ids[i]
        updated_reputations[op_id] = max(0.0, updated_reputations[op_id] - 0.2)

    # Filter to verified gradients only
    verified_gradients = [gradients[i] for i in verified_indices]
    verified_operator_ids = [operator_ids[i] for i in verified_indices]
    n_verified = len(verified_gradients)

    if n_verified == 0:
        raise ValueError("[TrustChain] ERROR: No valid gradients after verification!")

    print(f"[TrustChain] {n_verified}/{n} operators passed verification")

    # ===== PHASE 2: SPECTRAL CLUSTERING =====
    print(f"[TrustChain] Phase 2: Spectral clustering for Byzantine detection...")

    if n_verified < 3:
        # Too few operators for clustering, skip Byzantine detection
        print(f"[TrustChain] WARNING: Too few operators ({n_verified}) for clustering, "
              f"skipping Byzantine detection")
        honest_indices_rel = list(range(n_verified))
        byzantine_indices_rel = []
        clustering_quality = {'n_clusters': 1, 'silhouette_score': -1.0}
    else:
        try:
            cluster_labels, byzantine_cluster_id = spectral_clustering_byzantine_detection(
                verified_gradients, k=2
            )

            # Separate honest and Byzantine
            byzantine_indices_rel = np.where(cluster_labels == byzantine_cluster_id)[0].tolist()
            honest_indices_rel = np.where(cluster_labels != byzantine_cluster_id)[0].tolist()

            print(f"[TrustChain] Detected {len(byzantine_indices_rel)} Byzantine operators")

            # Analyze clustering quality
            from .spectral_clustering import analyze_clustering_quality
            clustering_quality = analyze_clustering_quality(verified_gradients, cluster_labels)

        except Exception as e:
            print(f"[TrustChain] WARNING: Spectral clustering failed: {e}")
            # Fallback: assume all verified gradients are honest
            honest_indices_rel = list(range(n_verified))
            byzantine_indices_rel = []
            clustering_quality = {'error': str(e)}

    # Map back to original indices
    honest_indices = [verified_indices[i] for i in honest_indices_rel]
    byzantine_indices = [verified_indices[i] for i in byzantine_indices_rel]
    byzantine_indices.extend(invalid_indices)  # Add invalid commitment operators

    # Filter to honest gradients
    honest_gradients = [verified_gradients[i] for i in honest_indices_rel]
    honest_operator_ids = [verified_operator_ids[i] for i in honest_indices_rel]

    if len(honest_gradients) == 0:
        raise ValueError("[TrustChain] ERROR: No honest operators remaining after Byzantine detection!")

    # ===== PHASE 3: REPUTATION-WEIGHTED AGGREGATION =====
    print(f"[TrustChain] Phase 3: Reputation-weighted aggregation...")

    # Extract reputation scores for honest operators
    honest_reputations = np.array([updated_reputations[op_id] for op_id in honest_operator_ids])

    # Normalize reputations to sum to 1 (for weighted average)
    rep_sum = honest_reputations.sum()
    if rep_sum == 0:
        # All have zero reputation (e.g., all new operators), use uniform weights
        weights = np.ones(len(honest_gradients)) / len(honest_gradients)
        print("[TrustChain] WARNING: All operators have zero reputation, using uniform weights")
    else:
        weights = honest_reputations / rep_sum

    # Weighted average of honest gradients
    aggregated_gradient = np.zeros_like(honest_gradients[0])
    for i, gradient in enumerate(honest_gradients):
        aggregated_gradient += weights[i] * gradient

    print(f"[TrustChain] Aggregated {len(honest_gradients)} honest gradients")
    print(f"[TrustChain] Reputation weights: min={weights.min():.3f}, "
          f"max={weights.max():.3f}, mean={weights.mean():.3f}")

    # ===== PHASE 4: DIFFERENTIAL PRIVACY =====
    print(f"[TrustChain] Phase 4: Adding DP noise (σ={sigma})...")

    # Add Gaussian noise for differential privacy
    dp_noise = np.random.normal(0, sigma, size=aggregated_gradient.shape)
    aggregated_gradient_dp = aggregated_gradient + dp_noise

    # ===== PHASE 5: REPUTATION UPDATE =====
    print(f"[TrustChain] Phase 5: Updating reputations...")

    updated_reputations = update_reputations(
        operator_ids, gradients, aggregated_gradient,
        honest_indices, byzantine_indices,
        updated_reputations, lambda_param=0.1, penalty=0.2
    )

    # ===== RETURN RESULTS =====
    result = {
        'aggregated_gradient': aggregated_gradient_dp,
        'updated_reputations': updated_reputations,
        'honest_indices': honest_indices,
        'byzantine_indices': byzantine_indices,
        'honest_operator_ids': honest_operator_ids,
        'weights': weights,
        'verification_results': verification_results,
        'clustering_quality': clustering_quality,
        'n_verified': n_verified,
        'n_honest': len(honest_indices),
        'n_byzantine': len(byzantine_indices)
    }

    print(f"[TrustChain] Aggregation complete: {len(honest_indices)} honest, "
          f"{len(byzantine_indices)} Byzantine, {len(invalid_indices)} invalid")

    return result


def robust_mean_aggregation(gradients: List[np.ndarray], weights: np.ndarray = None) -> np.ndarray:
    """
    Compute weighted mean with robustness checks.

    Args:
        gradients: List of gradient vectors
        weights: Optional weights (uniform if None)

    Returns:
        aggregated_gradient: Weighted mean gradient
    """
    if not gradients:
        raise ValueError("Cannot aggregate empty gradient list")

    n = len(gradients)
    if weights is None:
        weights = np.ones(n) / n
    else:
        # Normalize weights
        weights = weights / weights.sum()

    # Check for consistent dimensions
    gradient_dim = gradients[0].shape
    for i, grad in enumerate(gradients):
        if grad.shape != gradient_dim:
            raise ValueError(f"Gradient {i} has inconsistent shape: {grad.shape} vs {gradient_dim}")

    # Compute weighted average
    aggregated_gradient = np.zeros(gradient_dim)
    for i, gradient in enumerate(gradients):
        aggregated_gradient += weights[i] * gradient

    return aggregated_gradient


def adaptive_noise_calibration(gradients: List[np.ndarray], epsilon: float,
                              delta: float = 1e-6) -> float:
    """
    Adaptively calibrate DP noise based on gradient sensitivity.

    Args:
        gradients: List of gradients
        epsilon: Privacy budget
        delta: DP parameter

    Returns:
        sigma: Calibrated noise standard deviation
    """
    if not gradients:
        return 0.1  # Default noise

    # Estimate gradient sensitivity (L2 norm of gradients)
    gradient_norms = [np.linalg.norm(grad) for grad in gradients]
    max_norm = np.max(gradient_norms)
    median_norm = np.median(gradient_norms)

    # Use median as more robust estimate of typical gradient magnitude
    sensitivity = max(median_norm, 1.0)  # Avoid zero sensitivity

    # Gaussian mechanism: sigma = sqrt(2 * log(1.25/delta)) * sensitivity / epsilon
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon

    return sigma


def secure_aggregation_with_dropouts(gradients: List[np.ndarray],
                                   operator_ids: List[str],
                                   reputations: Dict[str, float],
                                   dropout_rate: float = 0.1) -> Dict:
    """
    Perform secure aggregation with random dropouts for robustness.

    Args:
        gradients: List of gradients
        operator_ids: List of operator IDs
        reputations: Reputation scores
        dropout_rate: Probability of dropping each operator

    Returns:
        result: Aggregation result with dropout information
    """
    n = len(gradients)

    # Random dropout (but keep at least 2 operators)
    keep_mask = np.random.rand(n) > dropout_rate
    if np.sum(keep_mask) < 2:
        # Keep at least 2 operators with highest reputation
        rep_scores = np.array([reputations.get(op_id, 0.5) for op_id in operator_ids])
        top_indices = np.argsort(rep_scores)[-2:]
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[top_indices] = True

    # Filter to active operators
    active_indices = np.where(keep_mask)[0]
    active_gradients = [gradients[i] for i in active_indices]
    active_operator_ids = [operator_ids[i] for i in active_indices]

    # Compute reputation weights for active operators
    active_reputations = {op_id: reputations.get(op_id, 0.5) for op_id in active_operator_ids}
    weights = compute_reputation_weights(active_operator_ids, active_reputations)

    # Aggregate
    aggregated_gradient = robust_mean_aggregation(active_gradients, weights)

    result = {
        'aggregated_gradient': aggregated_gradient,
        'active_indices': active_indices.tolist(),
        'active_operator_ids': active_operator_ids,
        'weights': weights,
        'dropout_rate': dropout_rate,
        'n_active': len(active_indices)
    }

    return result


class TrustChainAggregator:
    """
    Stateful TrustChain aggregator with history tracking and adaptive parameters.
    """

    def __init__(self, operator_ids: List[str], comm_params: Tuple[int, int, int, int],
                 lambda_param: float = 0.1, penalty: float = 0.2, base_sigma: float = 0.1):
        """
        Initialize TrustChain aggregator.

        Args:
            operator_ids: List of operator IDs
            comm_params: Commitment parameters (p, q, g, h)
            lambda_param: Reputation update rate
            penalty: Byzantine penalty
            base_sigma: Base DP noise level
        """
        self.operator_ids = operator_ids
        self.comm_params = comm_params
        self.lambda_param = lambda_param
        self.penalty = penalty
        self.base_sigma = base_sigma

        # Initialize reputations
        self.reputations = {op_id: 0.5 for op_id in operator_ids}

        # History tracking
        self.aggregation_history = []
        self.reputation_history = []
        self.byzantine_detection_history = []

    def aggregate(self, gradients: List[np.ndarray], commitments: List[int],
                 openings: List[int], epsilon: float = 0.1) -> Dict:
        """
        Perform one round of TrustChain aggregation.

        Args:
            gradients: List of gradients
            commitments: List of commitments
            openings: List of opening randomness
            epsilon: Privacy budget for this round

        Returns:
            result: Aggregation result
        """
        # Adaptive noise calibration
        sigma = adaptive_noise_calibration(gradients, epsilon)
        sigma = max(sigma, self.base_sigma)  # Minimum noise level

        dp_params = {'sigma': sigma, 'epsilon': epsilon}

        # Perform aggregation
        result = trustchain_aggregation(
            gradients, commitments, openings,
            self.reputations, self.operator_ids,
            self.comm_params, dp_params
        )

        # Update internal state
        self.reputations = result['updated_reputations']

        # Track history
        self.aggregation_history.append(result)
        self.reputation_history.append(self.reputations.copy())
        self.byzantine_detection_history.append(result['byzantine_indices'])

        return result

    def get_reputation_statistics(self) -> Dict[str, float]:
        """Get current reputation statistics."""
        from .reputation import get_reputation_statistics
        return get_reputation_statistics(self.reputations)

    def get_top_operators(self, k: int) -> List[str]:
        """Get top-k operators by reputation."""
        from .reputation import select_operators_by_reputation
        return select_operators_by_reputation(self.operator_ids, self.reputations, k)

    def plot_aggregation_stats(self, save_path: str = None) -> None:
        """
        Plot aggregation statistics over time.

        Args:
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt

        n_rounds = len(self.aggregation_history)
        if n_rounds == 0:
            print("No aggregation history to plot")
            return

        # Extract statistics
        n_honest = [result['n_honest'] for result in self.aggregation_history]
        n_byzantine = [result['n_byzantine'] for result in self.aggregation_history]
        n_verified = [result['n_verified'] for result in self.aggregation_history]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot operator counts
        rounds = range(1, n_rounds + 1)
        ax1.plot(rounds, n_verified, label='Verified', marker='o')
        ax1.plot(rounds, n_honest, label='Honest', marker='s')
        ax1.plot(rounds, n_byzantine, label='Byzantine', marker='^')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Number of Operators')
        ax1.set_title('Operator Classification Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot reputation evolution for each operator
        for op_id in self.operator_ids:
            reputation_evolution = [round_reps[op_id] for round_reps in self.reputation_history]
            ax2.plot(range(n_rounds + 1), reputation_evolution, label=op_id, marker='o', markersize=3)

        ax2.set_xlabel('Round')
        ax2.set_ylabel('Reputation Score')
        ax2.set_title('Reputation Evolution')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("Testing TrustChain Aggregation...")

    # Test with py313 environment
    import sys
    print(f"Python version: {sys.version}")

    # Setup test parameters
    from ..crypto.pedersen import setup_pedersen_commitment, commit_gradient

    operator_ids = [f"Operator_{i}" for i in range(5)]
    n_rounds = 5
    gradient_dim = 6

    # Setup cryptographic parameters (small for testing)
    comm_params = setup_pedersen_commitment(1024)

    # Initialize aggregator
    aggregator = TrustChainAggregator(operator_ids, comm_params)

    # Simulate training rounds
    for round_idx in range(n_rounds):
        print(f"\n{'='*50}")
        print(f"Round {round_idx + 1}")
        print(f"{'='*50}")

        # Generate test gradients
        gradients = []
        commitments = []
        openings = []

        for i, op_id in enumerate(operator_ids):
            if i < 3:  # Honest operators
                gradient = np.random.randn(gradient_dim) * 0.1
            else:  # Byzantine operators (sign flip attack)
                gradient = -np.random.randn(gradient_dim) * 0.2

            # Commit to gradient
            commitment, opening = commit_gradient(gradient, *comm_params)

            gradients.append(gradient)
            commitments.append(commitment)
            openings.append(opening)

        # Perform aggregation
        result = aggregator.aggregate(gradients, commitments, openings, epsilon=0.1)

        # Print results
        print(f"Aggregation completed:")
        print(f"  - Verified: {result['n_verified']}")
        print(f"  - Honest: {result['n_honest']}")
        print(f"  - Byzantine: {result['n_byzantine']}")
        print(f"  - Aggregated gradient norm: {np.linalg.norm(result['aggregated_gradient']):.4f}")

        # Print reputation statistics
        stats = aggregator.get_reputation_statistics()
        print(f"Reputation stats: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    print("\nTrustChain aggregation test completed!")