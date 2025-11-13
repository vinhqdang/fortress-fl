"""
Reputation System for FORTRESS-FL

Tracks operator quality based on gradient consistency and penalizes Byzantine behavior.
Uses exponential moving average for reputation updates.
"""

import numpy as np
from typing import List, Dict


def compute_quality_score(gradient: np.ndarray, aggregated_gradient: np.ndarray,
                         honest_gradients: List[np.ndarray]) -> float:
    """
    Compute quality score for gradient based on deviation from aggregated gradient.

    Args:
        gradient: Gradient from operator
        aggregated_gradient: Current aggregated gradient (consensus)
        honest_gradients: List of gradients from honest cluster

    Returns:
        quality_score: Quality score in [0, 1]
    """
    # Distance to aggregated gradient (consensus)
    distance_to_consensus = np.linalg.norm(gradient - aggregated_gradient)

    # Normalize by average distance of honest gradients
    if len(honest_gradients) > 0:
        honest_distances = [np.linalg.norm(g - aggregated_gradient) for g in honest_gradients]
        avg_honest_distance = np.mean(honest_distances)
    else:
        avg_honest_distance = 1.0

    # Prevent division by zero
    avg_honest_distance = max(avg_honest_distance, 1e-10)

    # Quality score: Higher for smaller distances
    # Q_i = exp(-distance / avg_distance)
    quality_score = np.exp(-distance_to_consensus / avg_honest_distance)

    # Clip to [0, 1]
    quality_score = np.clip(quality_score, 0.0, 1.0)

    return quality_score


def update_reputations(operator_ids: List[str], gradients: List[np.ndarray],
                      aggregated_gradient: np.ndarray, honest_indices: List[int],
                      byzantine_indices: List[int], reputations: Dict[str, float],
                      lambda_param: float = 0.1, penalty: float = 0.2) -> Dict[str, float]:
    """
    Update reputation scores for all operators.

    Args:
        operator_ids: List of operator IDs (length n)
        gradients: List of gradients (length n)
        aggregated_gradient: Current aggregated gradient
        honest_indices: Indices of honest operators
        byzantine_indices: Indices of detected Byzantine operators
        reputations: Current reputation scores (dict: operator_id -> reputation)
        lambda_param: Reputation update rate (EMA parameter)
        penalty: Penalty for Byzantine behavior

    Returns:
        updated_reputations: New reputation scores (dict)
    """
    updated_reputations = {}
    honest_gradients = [gradients[i] for i in honest_indices]

    for idx, op_id in enumerate(operator_ids):
        current_reputation = reputations.get(op_id, 0.5)  # Default reputation for new operators

        if idx in byzantine_indices:
            # Penalize detected Byzantine operators
            new_reputation = max(0.0, current_reputation - penalty)
        else:
            # Update based on quality score
            quality_score = compute_quality_score(gradients[idx], aggregated_gradient, honest_gradients)

            # Exponential moving average update
            new_reputation = (1 - lambda_param) * current_reputation + lambda_param * quality_score
            new_reputation = np.clip(new_reputation, 0.0, 1.0)

        updated_reputations[op_id] = new_reputation

    return updated_reputations


def select_operators_by_reputation(operator_ids: List[str], reputations: Dict[str, float],
                                  k: int) -> List[str]:
    """
    Select top-k operators based on reputation scores.

    Args:
        operator_ids: List of all operator IDs
        reputations: Dict mapping operator_id -> reputation
        k: Number of operators to select

    Returns:
        selected_ids: List of k selected operator IDs
    """
    # Sort operators by reputation (descending)
    sorted_ops = sorted(operator_ids, key=lambda op: reputations.get(op, 0.0), reverse=True)

    # Select top-k
    selected_ids = sorted_ops[:k]

    return selected_ids


def compute_reputation_weights(operator_ids: List[str], reputations: Dict[str, float]) -> np.ndarray:
    """
    Compute normalized weights from reputation scores.

    Args:
        operator_ids: List of operator IDs
        reputations: Dict mapping operator_id -> reputation

    Returns:
        weights: Normalized weights array (sums to 1)
    """
    reputation_scores = np.array([reputations.get(op_id, 0.5) for op_id in operator_ids])

    # Normalize to sum to 1
    total_reputation = reputation_scores.sum()
    if total_reputation == 0:
        # All have zero reputation, use uniform weights
        weights = np.ones(len(operator_ids)) / len(operator_ids)
    else:
        weights = reputation_scores / total_reputation

    return weights


def get_reputation_statistics(reputations: Dict[str, float]) -> Dict[str, float]:
    """
    Compute statistics about current reputation distribution.

    Args:
        reputations: Dict mapping operator_id -> reputation

    Returns:
        stats: Dict with reputation statistics
    """
    if not reputations:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }

    reputation_values = np.array(list(reputations.values()))

    stats = {
        'mean': np.mean(reputation_values),
        'std': np.std(reputation_values),
        'min': np.min(reputation_values),
        'max': np.max(reputation_values),
        'median': np.median(reputation_values),
        'count': len(reputation_values)
    }

    return stats


def initialize_reputations(operator_ids: List[str], initial_value: float = 0.5) -> Dict[str, float]:
    """
    Initialize reputation scores for a list of operators.

    Args:
        operator_ids: List of operator IDs
        initial_value: Initial reputation value (default: 0.5)

    Returns:
        reputations: Dict mapping operator_id -> initial reputation
    """
    reputations = {op_id: initial_value for op_id in operator_ids}
    return reputations


def reputation_decay(reputations: Dict[str, float], decay_rate: float = 0.01) -> Dict[str, float]:
    """
    Apply time-based reputation decay to prevent stale high reputations.

    Args:
        reputations: Current reputation scores
        decay_rate: Decay rate per time step

    Returns:
        decayed_reputations: Updated reputation scores after decay
    """
    decayed_reputations = {}

    for op_id, reputation in reputations.items():
        # Apply exponential decay
        new_reputation = reputation * (1 - decay_rate)
        decayed_reputations[op_id] = max(0.0, new_reputation)

    return decayed_reputations


def detect_reputation_manipulation(reputations: Dict[str, float],
                                 reputation_history: List[Dict[str, float]],
                                 threshold: float = 0.3) -> List[str]:
    """
    Detect potential reputation manipulation attacks.

    Args:
        reputations: Current reputation scores
        reputation_history: List of historical reputation states
        threshold: Threshold for detecting suspicious changes

    Returns:
        suspicious_operators: List of operator IDs with suspicious reputation changes
    """
    if len(reputation_history) < 2:
        return []

    suspicious_operators = []
    prev_reputations = reputation_history[-2]

    for op_id in reputations:
        if op_id in prev_reputations:
            current_rep = reputations[op_id]
            prev_rep = prev_reputations[op_id]

            # Check for sudden reputation spikes (potential manipulation)
            reputation_change = current_rep - prev_rep
            if reputation_change > threshold:
                suspicious_operators.append(op_id)

    return suspicious_operators


class ReputationTracker:
    """
    Class to track reputation evolution over time with additional analytics.
    """

    def __init__(self, operator_ids: List[str], initial_reputation: float = 0.5,
                 lambda_param: float = 0.1, penalty: float = 0.2):
        """
        Initialize reputation tracker.

        Args:
            operator_ids: List of operator IDs
            initial_reputation: Initial reputation value
            lambda_param: Reputation update rate
            penalty: Byzantine penalty
        """
        self.operator_ids = operator_ids
        self.lambda_param = lambda_param
        self.penalty = penalty

        # Initialize reputations
        self.reputations = initialize_reputations(operator_ids, initial_reputation)

        # History tracking
        self.reputation_history = [self.reputations.copy()]
        self.quality_scores_history = []
        self.byzantine_detections = []

    def update(self, gradients: List[np.ndarray], aggregated_gradient: np.ndarray,
              honest_indices: List[int], byzantine_indices: List[int]) -> None:
        """
        Update reputations and tracking history.

        Args:
            gradients: List of gradients
            aggregated_gradient: Aggregated gradient
            honest_indices: Indices of honest operators
            byzantine_indices: Indices of Byzantine operators
        """
        # Update reputations
        self.reputations = update_reputations(
            self.operator_ids, gradients, aggregated_gradient,
            honest_indices, byzantine_indices, self.reputations,
            self.lambda_param, self.penalty
        )

        # Track history
        self.reputation_history.append(self.reputations.copy())
        self.byzantine_detections.append(byzantine_indices.copy())

        # Compute quality scores for history
        honest_gradients = [gradients[i] for i in honest_indices]
        quality_scores = {}
        for idx, op_id in enumerate(self.operator_ids):
            if idx not in byzantine_indices:
                quality_scores[op_id] = compute_quality_score(
                    gradients[idx], aggregated_gradient, honest_gradients
                )
            else:
                quality_scores[op_id] = 0.0

        self.quality_scores_history.append(quality_scores)

    def get_weights(self) -> np.ndarray:
        """Get normalized reputation weights."""
        return compute_reputation_weights(self.operator_ids, self.reputations)

    def get_statistics(self) -> Dict[str, float]:
        """Get current reputation statistics."""
        return get_reputation_statistics(self.reputations)

    def plot_reputation_evolution(self, save_path: str = None) -> None:
        """
        Plot reputation evolution over time.

        Args:
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt

        n_rounds = len(self.reputation_history)

        plt.figure(figsize=(12, 8))

        for op_id in self.operator_ids:
            reputation_evolution = [round_reps[op_id] for round_reps in self.reputation_history]
            plt.plot(range(n_rounds), reputation_evolution, label=op_id, marker='o', markersize=4)

        plt.xlabel('Round')
        plt.ylabel('Reputation Score')
        plt.title('Reputation Evolution Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Reputation System...")

    # Setup test data
    operator_ids = [f"Operator_{i}" for i in range(5)]
    n_rounds = 10
    gradient_dim = 8

    # Initialize reputation tracker
    tracker = ReputationTracker(operator_ids, initial_reputation=0.5)

    # Simulate training rounds
    for round_idx in range(n_rounds):
        print(f"\nRound {round_idx + 1}:")

        # Generate test gradients
        # Operators 0, 1, 2 are honest
        # Operators 3, 4 are Byzantine
        gradients = []
        honest_base = np.random.randn(gradient_dim)

        for i, op_id in enumerate(operator_ids):
            if i < 3:  # Honest
                gradient = honest_base + 0.1 * np.random.randn(gradient_dim)
            else:  # Byzantine
                gradient = -honest_base + 0.2 * np.random.randn(gradient_dim)
            gradients.append(gradient)

        # Simulate aggregated gradient (average of honest gradients)
        honest_indices = [0, 1, 2]
        byzantine_indices = [3, 4]
        honest_gradients = [gradients[i] for i in honest_indices]
        aggregated_gradient = np.mean(honest_gradients, axis=0)

        # Update reputations
        tracker.update(gradients, aggregated_gradient, honest_indices, byzantine_indices)

        # Print current reputations
        current_stats = tracker.get_statistics()
        print(f"Reputation stats: mean={current_stats['mean']:.3f}, "
              f"std={current_stats['std']:.3f}")

        for op_id in operator_ids:
            rep = tracker.reputations[op_id]
            status = "Byzantine" if op_id in ["Operator_3", "Operator_4"] else "Honest"
            print(f"  {op_id}: {rep:.3f} ({status})")

    # Final analysis
    print(f"\nFinal reputation statistics:")
    final_stats = tracker.get_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value:.3f}")

    print("Reputation system test completed!")