"""
Evaluation and Analysis Tools for FORTRESS-FL

Comprehensive evaluation metrics and visualization tools for analyzing
FORTRESS-FL performance, Byzantine robustness, and privacy-utility tradeoffs.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    round_number: int
    global_model_norm: float
    test_loss: float
    test_accuracy: float
    privacy_budget_used: float
    n_honest: int
    n_byzantine: int
    byzantine_detection_accuracy: float
    reputation_stats: Dict[str, float]


class MetricsTracker:
    """Comprehensive metrics tracking for FORTRESS-FL."""

    def __init__(self):
        self.metrics_history = []
        self.reputation_history = []
        self.attack_history = []

    def record_round(self, metrics: TrainingMetrics) -> None:
        """Record metrics for a training round."""
        self.metrics_history.append(metrics)

    def record_reputations(self, reputations: Dict[str, float]) -> None:
        """Record reputation scores."""
        self.reputation_history.append(reputations.copy())

    def record_attack_info(self, attack_info: Dict) -> None:
        """Record attack information."""
        self.attack_history.append(attack_info)

    def get_convergence_metrics(self) -> Dict:
        """Calculate model convergence metrics."""
        if not self.metrics_history:
            return {}

        model_norms = [m.global_model_norm for m in self.metrics_history]
        test_losses = [m.test_loss for m in self.metrics_history if m.test_loss is not None]

        convergence_metrics = {
            'final_model_norm': model_norms[-1] if model_norms else 0,
            'model_norm_change': model_norms[-1] - model_norms[0] if len(model_norms) > 1 else 0,
            'model_norm_variance': np.var(model_norms) if len(model_norms) > 1 else 0,
            'convergence_rate': self._calculate_convergence_rate(model_norms),
            'final_test_loss': test_losses[-1] if test_losses else None,
            'test_loss_improvement': test_losses[0] - test_losses[-1] if len(test_losses) > 1 else None
        }

        return convergence_metrics

    def get_byzantine_robustness_metrics(self) -> Dict:
        """Calculate Byzantine robustness metrics."""
        if not self.metrics_history:
            return {}

        detection_accuracies = [m.byzantine_detection_accuracy for m in self.metrics_history
                              if m.byzantine_detection_accuracy is not None]
        n_byzantine_per_round = [m.n_byzantine for m in self.metrics_history]
        n_honest_per_round = [m.n_honest for m in self.metrics_history]

        robustness_metrics = {
            'avg_detection_accuracy': np.mean(detection_accuracies) if detection_accuracies else None,
            'detection_consistency': np.std(detection_accuracies) if len(detection_accuracies) > 1 else None,
            'avg_byzantine_detected': np.mean(n_byzantine_per_round),
            'avg_honest_preserved': np.mean(n_honest_per_round),
            'false_positive_rate': self._calculate_false_positive_rate(),
            'system_resilience': self._calculate_system_resilience()
        }

        return robustness_metrics

    def get_privacy_metrics(self) -> Dict:
        """Calculate privacy-related metrics."""
        if not self.metrics_history:
            return {}

        privacy_budgets = [m.privacy_budget_used for m in self.metrics_history
                          if m.privacy_budget_used is not None]
        test_losses = [m.test_loss for m in self.metrics_history if m.test_loss is not None]

        privacy_metrics = {
            'total_privacy_budget': sum(privacy_budgets),
            'avg_privacy_per_round': np.mean(privacy_budgets) if privacy_budgets else 0,
            'privacy_efficiency': self._calculate_privacy_efficiency(privacy_budgets, test_losses),
            'utility_loss': self._calculate_utility_loss()
        }

        return privacy_metrics

    def _calculate_convergence_rate(self, model_norms: List[float]) -> float:
        """Calculate model convergence rate."""
        if len(model_norms) < 3:
            return 0.0

        # Fit exponential decay: ||theta_t|| = A * exp(-r * t)
        # Use linear regression on log scale: log(||theta||) = log(A) - r * t
        t = np.arange(len(model_norms))
        log_norms = np.log(np.array(model_norms) + 1e-10)  # Avoid log(0)

        try:
            coeffs = np.polyfit(t, log_norms, 1)
            convergence_rate = -coeffs[0]  # Negative slope indicates convergence
            return max(0, convergence_rate)  # Return 0 if diverging
        except:
            return 0.0

    def _calculate_false_positive_rate(self) -> Optional[float]:
        """Calculate false positive rate in Byzantine detection."""
        # This requires ground truth information
        # For now, return None as we don't have ground truth in tracker
        return None

    def _calculate_system_resilience(self) -> float:
        """Calculate overall system resilience score."""
        if not self.metrics_history:
            return 0.0

        # Resilience based on maintaining high honest participation
        honest_rates = [m.n_honest / (m.n_honest + m.n_byzantine) if (m.n_honest + m.n_byzantine) > 0 else 0
                       for m in self.metrics_history]

        return np.mean(honest_rates) if honest_rates else 0.0

    def _calculate_privacy_efficiency(self, privacy_budgets: List[float],
                                    test_losses: List[float]) -> float:
        """Calculate privacy-utility efficiency."""
        if not privacy_budgets or not test_losses:
            return 0.0

        total_privacy = sum(privacy_budgets)
        final_loss = test_losses[-1] if test_losses else float('inf')

        # Efficiency = 1 / (privacy_cost * utility_loss)
        efficiency = 1.0 / (total_privacy * (final_loss + 1e-6))
        return efficiency

    def _calculate_utility_loss(self) -> Optional[float]:
        """Calculate utility loss due to privacy."""
        # This would require comparison with non-private baseline
        return None


def evaluate_convergence(fortress_fl, baseline_model: np.ndarray = None) -> Dict:
    """
    Evaluate model convergence properties.

    Args:
        fortress_fl: Trained FortressFL instance
        baseline_model: Optional baseline model for comparison

    Returns:
        convergence_eval: Dict with convergence metrics
    """
    history = fortress_fl.get_training_history()
    final_model = fortress_fl.get_global_model()

    # Basic convergence metrics
    model_norms = history['global_model_norms']
    convergence_eval = {
        'final_model_norm': np.linalg.norm(final_model),
        'model_stability': np.std(model_norms[-5:]) if len(model_norms) >= 5 else np.std(model_norms),
        'convergence_achieved': len(model_norms) > 1 and abs(model_norms[-1] - model_norms[-2]) < 0.01
    }

    # Compare with baseline if provided
    if baseline_model is not None:
        model_error = np.linalg.norm(final_model - baseline_model)
        convergence_eval['baseline_error'] = model_error
        convergence_eval['relative_error'] = model_error / np.linalg.norm(baseline_model)

    # Convergence rate estimation
    if len(model_norms) > 3:
        t = np.arange(len(model_norms))
        try:
            # Fit exponential: norm(t) = A * exp(-rate * t) + C
            log_norms = np.log(np.array(model_norms) + 1e-10)
            coeffs = np.polyfit(t, log_norms, 1)
            convergence_eval['convergence_rate'] = -coeffs[0]
        except:
            convergence_eval['convergence_rate'] = 0.0

    return convergence_eval


def evaluate_byzantine_robustness(fortress_fl, ground_truth_byzantine: List[int]) -> Dict:
    """
    Evaluate Byzantine robustness and detection accuracy.

    Args:
        fortress_fl: Trained FortressFL instance
        ground_truth_byzantine: Ground truth Byzantine operator indices

    Returns:
        robustness_eval: Dict with robustness metrics
    """
    history = fortress_fl.get_training_history()
    round_results = history['round_results']

    if not round_results or not ground_truth_byzantine:
        return {'error': 'Insufficient data for robustness evaluation'}

    # Detection accuracy across rounds
    detection_accuracies = []
    false_positives = []
    false_negatives = []
    precision_scores = []
    recall_scores = []

    for result in round_results:
        detected = set(result['byzantine_indices'])
        expected = set(ground_truth_byzantine)

        # True positives, false positives, false negatives
        tp = len(detected & expected)
        fp = len(detected - expected)
        fn = len(expected - detected)

        # Metrics
        if len(expected) > 0:
            recall = tp / len(expected)
            recall_scores.append(recall)
        else:
            recall_scores.append(1.0)

        if len(detected) > 0:
            precision = tp / len(detected)
            precision_scores.append(precision)
        else:
            precision_scores.append(1.0 if len(expected) == 0 else 0.0)

        detection_accuracies.append(recall)
        false_positives.append(fp)
        false_negatives.append(fn)

    robustness_eval = {
        'avg_detection_accuracy': np.mean(detection_accuracies),
        'final_detection_accuracy': detection_accuracies[-1],
        'detection_consistency': 1.0 - np.std(detection_accuracies),  # Higher is better
        'avg_precision': np.mean(precision_scores),
        'avg_recall': np.mean(recall_scores),
        'f1_score': 2 * np.mean(precision_scores) * np.mean(recall_scores) / (np.mean(precision_scores) + np.mean(recall_scores)),
        'avg_false_positives': np.mean(false_positives),
        'avg_false_negatives': np.mean(false_negatives),
        'false_positive_rate': np.mean(false_positives) / (fortress_fl.n_operators - len(ground_truth_byzantine)),
        'system_resilience': np.mean([result['n_honest'] / fortress_fl.n_operators for result in round_results])
    }

    return robustness_eval


def evaluate_privacy_utility_tradeoff(fortress_fl, test_data: Dict,
                                     baseline_performance: float = None) -> Dict:
    """
    Evaluate privacy-utility tradeoff.

    Args:
        fortress_fl: Trained FortressFL instance
        test_data: Test dataset
        baseline_performance: Baseline performance without privacy

    Returns:
        privacy_eval: Dict with privacy-utility metrics
    """
    # Model performance
    performance = fortress_fl.evaluate_model(test_data)
    current_loss = performance['loss']

    # Privacy budget consumption
    total_privacy_budget = fortress_fl.total_privacy_budget
    n_rounds = fortress_fl.round_number

    privacy_eval = {
        'total_privacy_budget': total_privacy_budget,
        'privacy_budget_per_round': total_privacy_budget / n_rounds if n_rounds > 0 else 0,
        'final_test_loss': current_loss,
        'model_utility': 1.0 / (1.0 + current_loss),  # Higher utility = lower loss
    }

    # Compare with baseline if provided
    if baseline_performance is not None:
        utility_loss = current_loss - baseline_performance
        privacy_eval['utility_degradation'] = utility_loss
        privacy_eval['privacy_efficiency'] = utility_loss / total_privacy_budget if total_privacy_budget > 0 else 0
    else:
        privacy_eval['utility_degradation'] = None
        privacy_eval['privacy_efficiency'] = None

    # Privacy-utility ratio (lower is better)
    privacy_eval['privacy_utility_ratio'] = total_privacy_budget * current_loss

    return privacy_eval


def plot_training_metrics(fortress_fl, save_path: str = None, figsize: Tuple[int, int] = (15, 12)) -> None:
    """
    Create comprehensive training metrics visualization.

    Args:
        fortress_fl: Trained FortressFL instance
        save_path: Optional path to save plot
        figsize: Figure size
    """
    history = fortress_fl.get_training_history()

    if not history['round_results']:
        print("No training history to plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('FORTRESS-FL Training Metrics', fontsize=16, fontweight='bold')

    rounds = range(1, len(history['round_results']) + 1)

    # 1. Model convergence
    ax1 = axes[0, 0]
    ax1.plot(rounds, history['global_model_norms'], 'b-', linewidth=2, marker='o')
    ax1.set_title('Model Convergence')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Global Model L2 Norm')
    ax1.grid(True, alpha=0.3)

    # 2. Byzantine detection
    ax2 = axes[0, 1]
    byzantine_counts = [result['n_byzantine'] for result in history['round_results']]
    honest_counts = [result['n_honest'] for result in history['round_results']]
    ax2.plot(rounds, honest_counts, 'g-', label='Honest', linewidth=2, marker='s')
    ax2.plot(rounds, byzantine_counts, 'r-', label='Byzantine', linewidth=2, marker='^')
    ax2.set_title('Byzantine Detection')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Number of Operators')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Privacy budget
    ax3 = axes[0, 2]
    ax3.plot(rounds, history['privacy_budgets'], 'purple', linewidth=2, marker='d')
    ax3.set_title('Privacy Budget Consumption')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Cumulative Privacy Budget')
    ax3.grid(True, alpha=0.3)

    # 4. Test performance (if available)
    ax4 = axes[1, 0]
    if 'test_losses' in history and history['test_losses']:
        test_rounds = range(1, len(history['test_losses']) + 1)
        ax4.plot(test_rounds, history['test_losses'], 'orange', linewidth=2, marker='o')
        ax4.set_title('Test Loss')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Test Loss')
    else:
        ax4.text(0.5, 0.5, 'No test data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Test Loss (N/A)')
    ax4.grid(True, alpha=0.3)

    # 5. Reputation evolution
    ax5 = axes[1, 1]
    rep_tracker = fortress_fl.reputation_tracker
    if rep_tracker.reputation_history:
        for op_id in fortress_fl.operator_ids:
            rep_evolution = [reps[op_id] for reps in rep_tracker.reputation_history]
            ax5.plot(range(len(rep_evolution)), rep_evolution, label=op_id, marker='o', markersize=3)
        ax5.set_title('Reputation Evolution')
        ax5.set_xlabel('Round')
        ax5.set_ylabel('Reputation Score')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.set_ylim(0, 1)
    else:
        ax5.text(0.5, 0.5, 'No reputation data', ha='center', va='center', transform=ax5.transAxes)
    ax5.grid(True, alpha=0.3)

    # 6. System efficiency
    ax6 = axes[1, 2]
    verification_rates = [result['n_verified'] / fortress_fl.n_operators for result in history['round_results']]
    ax6.plot(rounds, verification_rates, 'teal', linewidth=2, marker='x')
    ax6.set_title('System Efficiency')
    ax6.set_xlabel('Round')
    ax6.set_ylabel('Verification Rate')
    ax6.set_ylim(0, 1.1)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def generate_performance_report(fortress_fl, test_data: Dict = None,
                              ground_truth_byzantine: List[int] = None,
                              baseline_performance: float = None,
                              save_path: str = None) -> Dict:
    """
    Generate comprehensive performance report.

    Args:
        fortress_fl: Trained FortressFL instance
        test_data: Optional test dataset
        ground_truth_byzantine: Optional ground truth Byzantine operators
        baseline_performance: Optional baseline performance
        save_path: Optional path to save report

    Returns:
        report: Comprehensive performance report
    """
    report = {
        'system_info': {
            'n_operators': fortress_fl.n_operators,
            'model_dimension': fortress_fl.model_dim,
            'training_rounds': fortress_fl.round_number,
            'operator_ids': fortress_fl.operator_ids
        },
        'hyperparameters': {
            'lambda_reputation': fortress_fl.lambda_rep,
            'sigma_dp': fortress_fl.sigma_dp,
            'epsilon_dp': fortress_fl.epsilon_dp,
            'initial_reputation': fortress_fl.initial_reputation
        }
    }

    # Model performance
    if test_data is not None:
        model_performance = fortress_fl.evaluate_model(test_data)
        report['model_performance'] = model_performance

    # Convergence analysis
    convergence_metrics = evaluate_convergence(fortress_fl)
    report['convergence_analysis'] = convergence_metrics

    # Byzantine robustness (if ground truth available)
    if ground_truth_byzantine is not None:
        robustness_metrics = evaluate_byzantine_robustness(fortress_fl, ground_truth_byzantine)
        report['byzantine_robustness'] = robustness_metrics

    # Privacy analysis
    if test_data is not None:
        privacy_metrics = evaluate_privacy_utility_tradeoff(fortress_fl, test_data, baseline_performance)
        report['privacy_analysis'] = privacy_metrics

    # System statistics
    system_stats = fortress_fl.get_system_statistics()
    report['system_statistics'] = system_stats

    # Reputation analysis
    final_reputations = fortress_fl.get_reputations()
    rep_stats = fortress_fl.aggregator.get_reputation_statistics()
    report['reputation_analysis'] = {
        'final_reputations': final_reputations,
        'reputation_statistics': rep_stats
    }

    # Generate summary
    report['executive_summary'] = _generate_executive_summary(report)

    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Performance report saved to {save_path}")

    return report


def _generate_executive_summary(report: Dict) -> Dict:
    """Generate executive summary from detailed report."""
    summary = {}

    # Training success
    if 'convergence_analysis' in report:
        conv = report['convergence_analysis']
        summary['convergence_achieved'] = conv.get('convergence_achieved', False)
        summary['final_model_norm'] = conv.get('final_model_norm', 0)

    # Byzantine defense effectiveness
    if 'byzantine_robustness' in report:
        byz = report['byzantine_robustness']
        summary['avg_detection_accuracy'] = byz.get('avg_detection_accuracy', 0)
        summary['system_resilience'] = byz.get('system_resilience', 0)

    # Privacy preservation
    if 'privacy_analysis' in report:
        priv = report['privacy_analysis']
        summary['total_privacy_budget'] = priv.get('total_privacy_budget', 0)
        summary['privacy_efficiency'] = priv.get('privacy_efficiency', None)

    # Overall assessment
    if summary.get('avg_detection_accuracy', 0) > 0.8 and summary.get('system_resilience', 0) > 0.7:
        summary['overall_assessment'] = 'EXCELLENT'
    elif summary.get('avg_detection_accuracy', 0) > 0.6 and summary.get('system_resilience', 0) > 0.5:
        summary['overall_assessment'] = 'GOOD'
    else:
        summary['overall_assessment'] = 'NEEDS_IMPROVEMENT'

    return summary


# Example usage and testing
if __name__ == "__main__":
    print("Testing FORTRESS-FL Evaluation Tools...")

    # Create mock data for testing
    from ..core.fortress_fl import FortressFL
    from ..core.training import create_federated_datasets

    # Test setup
    n_operators = 5
    byzantine_operators = [3, 4]
    model_dim = 6
    n_rounds = 3

    # Create test data
    operators_data = create_federated_datasets(
        n_operators, 50, model_dim, byzantine_operators, task_type='regression'
    )

    # Initialize and run FORTRESS-FL (quick test)
    fortress_fl = FortressFL(
        n_operators, model_dim, security_param=512,  # Small for testing
        lambda_rep=0.1, sigma_dp=0.05, epsilon_dp=0.1
    )

    # Simulate a few rounds
    for round_idx in range(n_rounds):
        local_gradients = []
        for i, op_data in enumerate(operators_data):
            if op_data['is_byzantine']:
                gradient = np.random.randn(model_dim) * -1.0  # Sign flip
            else:
                gradient = np.random.randn(model_dim) * 0.1
            local_gradients.append(gradient)

        fortress_fl.train_round(local_gradients)

    # Test evaluation functions
    print("\n📊 Testing Evaluation Functions:")

    # Create test data
    X_test = np.random.randn(100, model_dim)
    y_test = X_test @ np.random.randn(model_dim)
    test_data = {'X': X_test, 'y': y_test}

    # Test convergence evaluation
    conv_eval = evaluate_convergence(fortress_fl)
    print(f"Convergence evaluation: {conv_eval}")

    # Test Byzantine robustness evaluation
    rob_eval = evaluate_byzantine_robustness(fortress_fl, byzantine_operators)
    print(f"Robustness evaluation: {rob_eval}")

    # Test privacy evaluation
    priv_eval = evaluate_privacy_utility_tradeoff(fortress_fl, test_data)
    print(f"Privacy evaluation: {priv_eval}")

    # Generate comprehensive report
    report = generate_performance_report(
        fortress_fl, test_data, byzantine_operators
    )

    print(f"\n📋 Executive Summary:")
    for key, value in report['executive_summary'].items():
        print(f"  {key}: {value}")

    print("\nFORTRESS-FL evaluation tools test completed!")