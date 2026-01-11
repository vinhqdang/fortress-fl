"""
FORTRESS-FL: Complete Implementation

Main class for the Federated Operator Resilient Trustworthy Resource Efficient
Secure Slice Learning system.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from ..crypto.pedersen import setup_pedersen_commitment, commit_gradient
from ..aggregation.trustchain import TrustChainAggregator
from ..aggregation.reputation import ReputationTracker


class FortressFL:
    """
    Complete FORTRESS-FL implementation for multi-operator Byzantine-robust federated learning.
    """

    def __init__(self, n_operators: int, model_dim: int, operator_ids: List[str] = None,
                 security_param: int = 2048, lambda_rep: float = 0.1, sigma_dp: float = 0.1,
                 epsilon_dp: float = 0.1, initial_reputation: float = 0.5):
        """
        Initialize FORTRESS-FL system.

        Args:
            n_operators: Number of participating operators
            model_dim: Dimension of model parameters
            operator_ids: List of operator IDs (auto-generated if None)
            security_param: Bit length for Pedersen commitment (default: 2048)
            lambda_rep: Reputation update rate (default: 0.1)
            sigma_dp: DP noise standard deviation (default: 0.1)
            epsilon_dp: Per-round privacy budget (default: 0.1)
            initial_reputation: Initial reputation for new operators (default: 0.5)
        """
        self.n_operators = n_operators
        self.model_dim = model_dim
        self.lambda_rep = lambda_rep
        self.sigma_dp = sigma_dp
        self.epsilon_dp = epsilon_dp
        self.initial_reputation = initial_reputation

        # Generate operator IDs if not provided
        if operator_ids is None:
            self.operator_ids = [f"Operator_{i}" for i in range(n_operators)]
        else:
            assert len(operator_ids) == n_operators, "Operator IDs length mismatch"
            self.operator_ids = operator_ids

        # Setup cryptographic parameters (one-time)
        print(f"[FORTRESS-FL] Setting up Pedersen commitment (security={security_param} bits)...")
        self.comm_params = setup_pedersen_commitment(security_param)

        # Initialize TrustChain aggregator
        self.aggregator = TrustChainAggregator(
            self.operator_ids, self.comm_params,
            lambda_param=lambda_rep, penalty=0.2, base_sigma=sigma_dp
        )

        # Initialize reputation tracker for additional analytics
        self.reputation_tracker = ReputationTracker(
            self.operator_ids, initial_reputation, lambda_rep, penalty=0.2
        )

        # Initialize global model
        self.global_model = np.zeros(model_dim)

        # Training metadata
        self.round_number = 0
        self.total_privacy_budget = 0.0

        # Training history
        self.history = {
            'global_model_norms': [],
            'privacy_budgets': [],
            'round_results': []
        }

        print(f"[FORTRESS-FL] Initialized with {n_operators} operators, model dim={model_dim}")
        print(f"[FORTRESS-FL] Operator IDs: {self.operator_ids}")

    def train_round(self, local_gradients: List[np.ndarray],
                   learning_rate: float = 0.01, max_grad_norm: float = 1.0) -> Dict:
        """
        Execute one round of FORTRESS-FL training.

        Args:
            local_gradients: List of n gradient vectors from operators
            learning_rate: Learning rate for global model update

        Returns:
            result: Dict with aggregated gradient and training statistics
        """
        self.round_number += 1

        print(f"\n{'='*70}")
        print(f"[FORTRESS-FL] Round {self.round_number}")
        print(f"{'='*70}")

        n = len(local_gradients)
        assert n == self.n_operators, f"Expected {self.n_operators} gradients, got {n}"

        # Validate gradient dimensions
        for i, grad in enumerate(local_gradients):
            assert grad.shape == (self.model_dim,), \
                f"Gradient {i} has wrong shape: {grad.shape} vs ({self.model_dim},)"

        # ===== STEP 1: COMMITMENT PHASE =====
        print(f"[FORTRESS-FL] Step 1: Operators commit to gradients...")
        commitments = []
        openings = []

        for i, gradient in enumerate(local_gradients):
            commitment, opening = commit_gradient(gradient, *self.comm_params)
            commitments.append(commitment)
            openings.append(opening)

        print(f"[FORTRESS-FL] Received {n} commitments")

        # ===== STEP 2: REVEAL PHASE =====
        print(f"[FORTRESS-FL] Step 2: Operators reveal gradients...")
        # In real system, operators send (g_i, r_i) in second message
        # For simulation, we already have them

        # ===== STEP 3: TRUSTCHAIN AGGREGATION =====
        print(f"[FORTRESS-FL] Step 3: TrustChain aggregation...")

        # Update privacy budget for this round
        current_epsilon = self.epsilon_dp
        self.total_privacy_budget += current_epsilon

        result = self.aggregator.aggregate(
            local_gradients, commitments, openings, epsilon=current_epsilon,
            max_grad_norm=max_grad_norm
        )

        # ===== STEP 4: UPDATE GLOBAL MODEL =====
        print(f"[FORTRESS-FL] Step 4: Updating global model...")
        aggregated_gradient = result['aggregated_gradient']

        # Apply gradient to global model
        self.global_model -= learning_rate * aggregated_gradient  # Gradient descent

        # ===== STEP 5: UPDATE REPUTATION TRACKER =====
        # Update the separate reputation tracker for analytics
        self.reputation_tracker.update(
            local_gradients, aggregated_gradient,
            result['honest_indices'], result['byzantine_indices']
        )

        # ===== STEP 6: LOG HISTORY =====
        global_model_norm = np.linalg.norm(self.global_model)
        self.history['global_model_norms'].append(global_model_norm)
        self.history['privacy_budgets'].append(self.total_privacy_budget)
        self.history['round_results'].append(result)

        # ===== STEP 7: PRINT SUMMARY =====
        print(f"\n[FORTRESS-FL] Round {self.round_number} Summary:")
        print(f"  - Verified operators: {result['n_verified']}")
        print(f"  - Honest operators: {result['n_honest']}")
        print(f"  - Byzantine detected: {result['n_byzantine']}")
        print(f"  - Aggregated gradient norm: {np.linalg.norm(aggregated_gradient):.4f}")
        print(f"  - Global model norm: {global_model_norm:.4f}")
        print(f"  - Privacy budget used: {current_epsilon:.3f} (total: {self.total_privacy_budget:.3f})")

        # Print current reputations
        rep_stats = self.aggregator.get_reputation_statistics()
        print(f"  - Reputation stats: mean={rep_stats['mean']:.3f}, std={rep_stats['std']:.3f}")

        # Enhanced result with additional metadata
        enhanced_result = result.copy()
        enhanced_result.update({
            'round_number': self.round_number,
            'global_model': self.global_model.copy(),
            'global_model_norm': global_model_norm,
            'learning_rate': learning_rate,
            'privacy_budget_used': current_epsilon,
            'total_privacy_budget': self.total_privacy_budget,
            'reputation_stats': rep_stats
        })

        return enhanced_result

    def get_global_model(self) -> np.ndarray:
        """Return current global model parameters."""
        return self.global_model.copy()

    def get_reputations(self) -> Dict[str, float]:
        """Return current reputation scores."""
        return self.aggregator.reputations.copy()

    def get_training_history(self) -> Dict:
        """Return complete training history."""
        return self.history.copy()

    def evaluate_model(self, test_data: Dict, loss_function=None, loss_type='mse') -> Dict:
        """
        Evaluate the global model on test data.

        Args:
            test_data: Dict with 'X' (features) and 'y' (targets)
            loss_function: Custom loss function (uses MSE if None)
            loss_type: 'mse' or 'logistic' (used if loss_function is None)

        Returns:
            evaluation: Dict with evaluation metrics
        """
        X = test_data['X']
        y = test_data['y']

        # Make predictions
        if loss_type == 'logistic':
            # Logistic regression prediction (sigmoid)
            logits = X @ self.global_model
            predictions = np.where(logits >= 0, 
                                  1 / (1 + np.exp(-logits)), 
                                  np.exp(logits) / (1 + np.exp(logits)))
        else:
            # Linear regression prediction
            predictions = X @ self.global_model

        # Compute loss
        if loss_function is None:
            if loss_type == 'logistic':
                # Log loss (binary cross entropy)
                epsilon = 1e-15
                pred_clipped = np.clip(predictions, epsilon, 1 - epsilon)
                loss = -np.mean(y * np.log(pred_clipped) + (1 - y) * np.log(1 - pred_clipped))
            else:
                # Default to mean squared error for regression
                mse = np.mean((predictions - y) ** 2)
                loss = mse
        else:
            loss = loss_function(y, predictions)

        # Compute additional metrics
        mae = np.mean(np.abs(predictions - y))
        
        evaluation = {
            'loss': loss,
            'mae': mae,
            'predictions': predictions
        }

        if loss_type == 'logistic':
            # Classification metrics
            pred_labels = (predictions > 0.5).astype(int)
            accuracy = np.mean(pred_labels == y)
            evaluation['accuracy'] = accuracy
        else:
            # Regression metrics
            r2 = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
            evaluation['r2_score'] = r2
            evaluation['mse'] = np.mean((predictions - y) ** 2)

        return evaluation

    def save_model(self, filepath: str) -> None:
        """
        Save the global model and training metadata.

        Args:
            filepath: Path to save the model
        """
        import pickle

        model_data = {
            'global_model': self.global_model,
            'model_dim': self.model_dim,
            'operator_ids': self.operator_ids,
            'reputations': self.get_reputations(),
            'round_number': self.round_number,
            'total_privacy_budget': self.total_privacy_budget,
            'history': self.history,
            'hyperparameters': {
                'lambda_rep': self.lambda_rep,
                'sigma_dp': self.sigma_dp,
                'epsilon_dp': self.epsilon_dp,
                'initial_reputation': self.initial_reputation
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"[FORTRESS-FL] Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'FortressFL':
        """
        Load a saved FORTRESS-FL model.

        Args:
            filepath: Path to the saved model

        Returns:
            fortress_fl: Loaded FortressFL instance
        """
        import pickle

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create new instance
        hyperparams = model_data['hyperparameters']
        fortress_fl = cls(
            n_operators=len(model_data['operator_ids']),
            model_dim=model_data['model_dim'],
            operator_ids=model_data['operator_ids'],
            lambda_rep=hyperparams['lambda_rep'],
            sigma_dp=hyperparams['sigma_dp'],
            epsilon_dp=hyperparams['epsilon_dp'],
            initial_reputation=hyperparams['initial_reputation']
        )

        # Restore state
        fortress_fl.global_model = model_data['global_model']
        fortress_fl.round_number = model_data['round_number']
        fortress_fl.total_privacy_budget = model_data['total_privacy_budget']
        fortress_fl.history = model_data['history']
        fortress_fl.aggregator.reputations = model_data['reputations']

        print(f"[FORTRESS-FL] Model loaded from {filepath}")

        return fortress_fl

    def plot_training_progress(self, save_path: str = None) -> None:
        """
        Plot training progress including model convergence and reputation evolution.

        Args:
            save_path: Optional path to save plots
        """
        import matplotlib.pyplot as plt

        if not self.history['global_model_norms']:
            print("No training history to plot")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        rounds = range(1, len(self.history['global_model_norms']) + 1)

        # 1. Global model convergence
        ax1.plot(rounds, self.history['global_model_norms'], 'b-', linewidth=2)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Global Model Norm')
        ax1.set_title('Global Model Convergence')
        ax1.grid(True, alpha=0.3)

        # 2. Privacy budget consumption
        ax2.plot(rounds, self.history['privacy_budgets'], 'r-', linewidth=2)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Cumulative Privacy Budget')
        ax2.set_title('Privacy Budget Consumption')
        ax2.grid(True, alpha=0.3)

        # 3. Byzantine detection over time
        byzantine_counts = [len(result['byzantine_indices']) for result in self.history['round_results']]
        honest_counts = [result['n_honest'] for result in self.history['round_results']]

        ax3.plot(rounds, honest_counts, 'g-', label='Honest', linewidth=2)
        ax3.plot(rounds, byzantine_counts, 'r-', label='Byzantine', linewidth=2)
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Number of Operators')
        ax3.set_title('Byzantine Detection Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Reputation evolution
        for op_id in self.operator_ids:
            reputation_evolution = [round_reps[op_id] for round_reps in self.reputation_tracker.reputation_history]
            ax4.plot(range(len(reputation_evolution)), reputation_evolution,
                    label=op_id, marker='o', markersize=3)

        ax4.set_xlabel('Round')
        ax4.set_ylabel('Reputation Score')
        ax4.set_title('Reputation Evolution')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def get_system_statistics(self) -> Dict:
        """
        Get comprehensive system statistics.

        Returns:
            stats: Dict with system performance metrics
        """
        if not self.history['round_results']:
            return {'error': 'No training history available'}

        # Aggregate statistics across all rounds
        total_rounds = len(self.history['round_results'])

        # Byzantine detection accuracy (if ground truth available)
        total_byzantine_detected = sum(len(result['byzantine_indices'])
                                     for result in self.history['round_results'])

        # Privacy budget efficiency
        avg_privacy_per_round = self.total_privacy_budget / total_rounds if total_rounds > 0 else 0

        # Reputation statistics
        rep_stats = self.aggregator.get_reputation_statistics()

        # Model convergence
        model_norm_change = (self.history['global_model_norms'][-1] -
                           self.history['global_model_norms'][0]) if len(self.history['global_model_norms']) > 1 else 0

        stats = {
            'training_rounds': total_rounds,
            'total_byzantine_detected': total_byzantine_detected,
            'avg_byzantine_per_round': total_byzantine_detected / total_rounds if total_rounds > 0 else 0,
            'total_privacy_budget': self.total_privacy_budget,
            'avg_privacy_per_round': avg_privacy_per_round,
            'model_norm_change': model_norm_change,
            'final_model_norm': self.history['global_model_norms'][-1] if self.history['global_model_norms'] else 0,
            'reputation_statistics': rep_stats,
            'system_efficiency': {
                'avg_honest_operators': np.mean([result['n_honest'] for result in self.history['round_results']]),
                'avg_verification_rate': np.mean([result['n_verified'] / self.n_operators for result in self.history['round_results']]),
            }
        }

        return stats

    def __repr__(self) -> str:
        """String representation of FortressFL instance."""
        return (f"FortressFL(operators={self.n_operators}, model_dim={self.model_dim}, "
                f"rounds={self.round_number}, privacy_budget={self.total_privacy_budget:.3f})")


# Example usage and testing
if __name__ == "__main__":
    print("Testing FORTRESS-FL Main Class...")

    # Test parameters
    n_operators = 5
    model_dim = 8
    n_rounds = 3

    # Initialize FORTRESS-FL (small security parameter for testing)
    fortress_fl = FortressFL(
        n_operators=n_operators,
        model_dim=model_dim,
        security_param=1024,  # Small for testing
        lambda_rep=0.1,
        sigma_dp=0.05,
        epsilon_dp=0.1
    )

    # Simulate training rounds
    for round_idx in range(n_rounds):
        print(f"\n🔄 Simulating Round {round_idx + 1}")

        # Generate synthetic gradients
        local_gradients = []
        for i in range(n_operators):
            if i < 3:  # Honest operators
                gradient = np.random.randn(model_dim) * 0.1
            else:  # Byzantine operators
                gradient = -np.random.randn(model_dim) * 0.2

            local_gradients.append(gradient)

        # Execute training round
        result = fortress_fl.train_round(local_gradients, learning_rate=0.01)

    # Final system statistics
    print(f"\n{'='*70}")
    print("FINAL SYSTEM STATISTICS")
    print(f"{'='*70}")

    stats = fortress_fl.get_system_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

    print(f"\nFORTRESS-FL instance: {fortress_fl}")
    print("FORTRESS-FL main class test completed!")