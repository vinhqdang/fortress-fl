"""
Training Loop Functions for FORTRESS-FL

High-level training functions and utilities for multi-round federated learning.
"""

import numpy as np
from typing import List, Dict, Callable, Optional, Tuple

from .fortress_fl import FortressFL


def train_fortress_fl(operators_data: List[Dict], n_rounds: int, model_dim: int,
                     learning_rate: float = 0.01, security_param: int = 2048,
                     lambda_rep: float = 0.1, sigma_dp: float = 0.1, epsilon_dp: float = 0.1,
                     test_data: Dict = None, verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Complete multi-round FORTRESS-FL training.

    Args:
        operators_data: List of dicts, each with {'id': str, 'dataset': data, 'is_byzantine': bool}
        n_rounds: Number of training rounds
        model_dim: Model parameter dimension
        learning_rate: Global model learning rate
        security_param: Cryptographic security parameter
        lambda_rep: Reputation update rate
        sigma_dp: DP noise standard deviation
        epsilon_dp: Per-round privacy budget
        test_data: Optional test data for evaluation
        verbose: Whether to print detailed progress

    Returns:
        (final_model, history): Final model and complete training history
    """
    n_operators = len(operators_data)
    operator_ids = [op['id'] for op in operators_data]

    if verbose:
        print(f"🚀 Starting FORTRESS-FL Training")
        print(f"{'='*70}")
        print(f"Operators: {n_operators}")
        print(f"Rounds: {n_rounds}")
        print(f"Model dimension: {model_dim}")
        print(f"Learning rate: {learning_rate}")
        print(f"Security parameter: {security_param}")
        print(f"Reputation λ: {lambda_rep}")
        print(f"DP noise σ: {sigma_dp}")
        print(f"Privacy budget ε: {epsilon_dp}")
        print(f"{'='*70}")

    # Initialize FORTRESS-FL
    fortress_fl = FortressFL(
        n_operators=n_operators,
        model_dim=model_dim,
        operator_ids=operator_ids,
        security_param=security_param,
        lambda_rep=lambda_rep,
        sigma_dp=sigma_dp,
        epsilon_dp=epsilon_dp
    )

    # Training history
    training_history = {
        'test_losses': [],
        'test_accuracies': [],
        'byzantine_detection_accuracy': []
    }

    # Training loop
    for round_idx in range(n_rounds):
        if verbose:
            print(f"\n{'#'*70}")
            print(f"# TRAINING ROUND {round_idx + 1}/{n_rounds}")
            print(f"{'#'*70}")

        # ===== LOCAL TRAINING =====
        # Each operator computes gradient on local data
        local_gradients = []
        for op in operators_data:
            if op['is_byzantine']:
                # Byzantine operator: Generate malicious gradient
                gradient = generate_byzantine_gradient(
                    model_dim,
                    attack_type=op.get('attack_type', 'sign_flip'),
                    current_model=fortress_fl.get_global_model()
                )
            else:
                # Honest operator: Compute true gradient
                gradient = compute_local_gradient(
                    fortress_fl.get_global_model(),
                    op['dataset'],
                    learning_rate=learning_rate
                )
            local_gradients.append(gradient)

        # ===== FORTRESS-FL AGGREGATION =====
        result = fortress_fl.train_round(local_gradients, learning_rate)

        # ===== EVALUATION =====
        if test_data is not None:
            evaluation = fortress_fl.evaluate_model(test_data)
            training_history['test_losses'].append(evaluation['loss'])
            training_history['test_accuracies'].append(evaluation.get('accuracy', evaluation['r2_score']))

            if verbose:
                print(f"Test evaluation: loss={evaluation['loss']:.4f}, "
                      f"R²={evaluation['r2_score']:.4f}")

        # ===== BYZANTINE DETECTION ACCURACY =====
        # Compute detection accuracy if ground truth is available
        expected_byzantine = [i for i, op in enumerate(operators_data) if op['is_byzantine']]
        detected_byzantine = result['byzantine_indices']

        if expected_byzantine:
            detection_accuracy = len(set(expected_byzantine) & set(detected_byzantine)) / len(expected_byzantine)
            training_history['byzantine_detection_accuracy'].append(detection_accuracy)

            if verbose:
                print(f"Byzantine detection: expected={expected_byzantine}, "
                      f"detected={detected_byzantine}, accuracy={detection_accuracy:.2%}")

    # Return final model and complete history
    final_model = fortress_fl.get_global_model()
    complete_history = fortress_fl.get_training_history()
    complete_history.update(training_history)

    if verbose:
        print(f"\n🎉 FORTRESS-FL Training Complete!")
        print(f"Final model norm: {np.linalg.norm(final_model):.4f}")
        print(f"Total privacy budget used: {fortress_fl.total_privacy_budget:.3f}")

    return final_model, complete_history


def compute_local_gradient(global_model: np.ndarray, local_dataset: Dict,
                          learning_rate: float = 0.01, loss_type: str = 'mse') -> np.ndarray:
    """
    Compute local gradient for honest operator.

    Args:
        global_model: Current global model parameters
        local_dataset: Local training data with 'X' and 'y'
        learning_rate: Local SGD learning rate
        loss_type: Type of loss function ('mse', 'logistic')

    Returns:
        gradient: Local gradient vector
    """
    X = local_dataset['X']
    y = local_dataset['y']

    if loss_type == 'mse':
        # Linear regression: loss = 0.5 * ||y - X @ theta||^2
        # gradient = X^T @ (X @ theta - y) / n
        predictions = X @ global_model
        errors = predictions - y
        gradient = (X.T @ errors) / len(y)

    elif loss_type == 'logistic':
        # Logistic regression: loss = -y * log(sigmoid(X @ theta)) - (1-y) * log(1-sigmoid(X @ theta))
        # gradient = X^T @ (sigmoid(X @ theta) - y) / n
        predictions = sigmoid(X @ global_model)
        errors = predictions - y
        gradient = (X.T @ errors) / len(y)

    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return gradient


def generate_byzantine_gradient(model_dim: int, attack_type: str = 'sign_flip',
                               current_model: np.ndarray = None,
                               attack_strength: float = 1.0) -> np.ndarray:
    """
    Generate malicious gradient for Byzantine operator.

    Args:
        model_dim: Dimension of gradient
        attack_type: Type of attack
        current_model: Current global model (for adaptive attacks)
        attack_strength: Strength of the attack

    Returns:
        byzantine_gradient: Malicious gradient vector
    """
    if attack_type == 'sign_flip':
        # Flip sign of typical honest gradient direction
        honest_direction = -np.random.randn(model_dim)  # Assume minimization
        byzantine_gradient = -honest_direction * attack_strength

    elif attack_type == 'random':
        # Random noise attack
        byzantine_gradient = np.random.randn(model_dim) * attack_strength * 10.0

    elif attack_type == 'zero':
        # Free-riding: contribute nothing
        byzantine_gradient = np.zeros(model_dim)

    elif attack_type == 'gaussian':
        # Gaussian noise with high variance
        byzantine_gradient = np.random.normal(0, attack_strength, model_dim)

    elif attack_type == 'model_poisoning' and current_model is not None:
        # Target specific model parameters
        target_direction = np.random.randn(model_dim)
        target_direction = target_direction / np.linalg.norm(target_direction)
        byzantine_gradient = target_direction * attack_strength

    elif attack_type == 'coordinated' and current_model is not None:
        # Coordinated attack targeting model divergence
        # Push model in specific malicious direction
        malicious_direction = np.ones(model_dim) / np.sqrt(model_dim)
        byzantine_gradient = malicious_direction * attack_strength

    else:
        # Default to sign flip
        honest_direction = -np.random.randn(model_dim)
        byzantine_gradient = -honest_direction * attack_strength

    return byzantine_gradient


def generate_synthetic_data(n_samples: int, n_features: int, noise_std: float = 0.1,
                          task_type: str = 'regression') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset for testing.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise_std: Standard deviation of noise
        task_type: 'regression' or 'classification'

    Returns:
        (X, y): Feature matrix and target vector
    """
    # Generate random features
    X = np.random.randn(n_samples, n_features)

    if task_type == 'regression':
        # Linear regression: y = X @ true_theta + noise
        true_theta = np.random.randn(n_features)
        y = X @ true_theta + np.random.normal(0, noise_std, n_samples)

    elif task_type == 'classification':
        # Logistic regression: y = sigmoid(X @ true_theta) + noise > 0.5
        true_theta = np.random.randn(n_features)
        logits = X @ true_theta
        probabilities = sigmoid(logits)
        y = (probabilities + np.random.normal(0, noise_std, n_samples)) > 0.5
        y = y.astype(float)

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return X, y


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid function."""
    return np.where(x >= 0,
                   1 / (1 + np.exp(-x)),
                   np.exp(x) / (1 + np.exp(x)))


def create_federated_datasets(n_operators: int, n_samples_per_operator: int,
                             n_features: int, byzantine_operators: List[int] = None,
                             data_heterogeneity: float = 0.0,
                             task_type: str = 'regression') -> List[Dict]:
    """
    Create federated datasets for multiple operators.

    Args:
        n_operators: Number of operators
        n_samples_per_operator: Samples per operator
        n_features: Number of features
        byzantine_operators: List of Byzantine operator indices
        data_heterogeneity: Degree of data heterogeneity (0=homogeneous, 1=fully heterogeneous)
        task_type: 'regression' or 'classification'

    Returns:
        operators_data: List of operator data dictionaries
    """
    if byzantine_operators is None:
        byzantine_operators = []

    # Generate global true model
    global_true_theta = np.random.randn(n_features)

    operators_data = []

    for i in range(n_operators):
        # Create local data distribution (with potential heterogeneity)
        if data_heterogeneity > 0:
            # Add local bias to features
            local_bias = np.random.randn(n_features) * data_heterogeneity
            X = np.random.randn(n_samples_per_operator, n_features) + local_bias
        else:
            # Homogeneous data distribution
            X = np.random.randn(n_samples_per_operator, n_features)

        # Generate labels using global model
        if task_type == 'regression':
            y = X @ global_true_theta + np.random.normal(0, 0.1, n_samples_per_operator)
        else:  # classification
            logits = X @ global_true_theta
            probabilities = sigmoid(logits)
            y = (probabilities > 0.5).astype(float)

        # Create operator data
        operator_data = {
            'id': f'Operator_{i}',
            'dataset': {'X': X, 'y': y},
            'is_byzantine': i in byzantine_operators,
            'attack_type': 'sign_flip' if i in byzantine_operators else None
        }

        operators_data.append(operator_data)

    return operators_data


def evaluate_fortress_fl_performance(fortress_fl: FortressFL, test_data: Dict,
                                   ground_truth_byzantine: List[int] = None) -> Dict:
    """
    Comprehensive evaluation of FORTRESS-FL performance.

    Args:
        fortress_fl: Trained FortressFL instance
        test_data: Test dataset
        ground_truth_byzantine: Ground truth Byzantine operator indices

    Returns:
        evaluation: Dict with comprehensive performance metrics
    """
    # Model performance
    model_eval = fortress_fl.evaluate_model(test_data)

    # System statistics
    system_stats = fortress_fl.get_system_statistics()

    # Byzantine detection analysis
    detection_analysis = {}
    if ground_truth_byzantine is not None and fortress_fl.history['round_results']:
        # Compute detection accuracy across all rounds
        detection_accuracies = []
        false_positives = []
        false_negatives = []

        for result in fortress_fl.history['round_results']:
            detected = set(result['byzantine_indices'])
            expected = set(ground_truth_byzantine)

            if expected:  # Only compute if there are Byzantine operators
                true_positives = len(detected & expected)
                accuracy = true_positives / len(expected)
                detection_accuracies.append(accuracy)

                fp = len(detected - expected)
                fn = len(expected - detected)
                false_positives.append(fp)
                false_negatives.append(fn)

        if detection_accuracies:
            detection_analysis = {
                'avg_detection_accuracy': np.mean(detection_accuracies),
                'final_detection_accuracy': detection_accuracies[-1],
                'avg_false_positives': np.mean(false_positives),
                'avg_false_negatives': np.mean(false_negatives),
                'detection_consistency': np.std(detection_accuracies)
            }

    # Privacy analysis
    privacy_efficiency = {
        'total_budget_used': fortress_fl.total_privacy_budget,
        'budget_per_round': fortress_fl.total_privacy_budget / fortress_fl.round_number if fortress_fl.round_number > 0 else 0,
        'privacy_utility_tradeoff': model_eval['loss'] * fortress_fl.total_privacy_budget  # Simple metric
    }

    # Combine all evaluations
    comprehensive_eval = {
        'model_performance': model_eval,
        'system_statistics': system_stats,
        'byzantine_detection': detection_analysis,
        'privacy_analysis': privacy_efficiency,
        'fortress_fl_summary': str(fortress_fl)
    }

    return comprehensive_eval


# Example usage and testing
if __name__ == "__main__":
    print("Testing FORTRESS-FL Training Functions...")

    # Test parameters
    n_operators = 6
    byzantine_operators = [4, 5]  # Last 2 operators are Byzantine
    model_dim = 10
    n_rounds = 5
    n_samples_per_op = 100

    print(f"Setting up federated learning scenario:")
    print(f"  - {n_operators} operators")
    print(f"  - Byzantine operators: {byzantine_operators}")
    print(f"  - Model dimension: {model_dim}")
    print(f"  - Training rounds: {n_rounds}")

    # Create federated datasets
    operators_data = create_federated_datasets(
        n_operators=n_operators,
        n_samples_per_operator=n_samples_per_op,
        n_features=model_dim,
        byzantine_operators=byzantine_operators,
        data_heterogeneity=0.1,  # Slight heterogeneity
        task_type='regression'
    )

    # Create test data
    X_test, y_test = generate_synthetic_data(200, model_dim, task_type='regression')
    test_data = {'X': X_test, 'y': y_test}

    # Run FORTRESS-FL training
    final_model, history = train_fortress_fl(
        operators_data=operators_data,
        n_rounds=n_rounds,
        model_dim=model_dim,
        learning_rate=0.01,
        security_param=1024,  # Small for testing
        lambda_rep=0.15,
        sigma_dp=0.05,
        epsilon_dp=0.1,
        test_data=test_data,
        verbose=True
    )

    print(f"\n🎯 TRAINING RESULTS")
    print(f"{'='*50}")
    print(f"Final model norm: {np.linalg.norm(final_model):.4f}")

    if history['test_losses']:
        print(f"Test loss progression: {history['test_losses'][0]:.4f} → {history['test_losses'][-1]:.4f}")

    if history['byzantine_detection_accuracy']:
        avg_detection = np.mean(history['byzantine_detection_accuracy'])
        print(f"Average Byzantine detection accuracy: {avg_detection:.2%}")

    print("\nFORTRESS-FL training functions test completed!")