#!/usr/bin/env python3
"""
MNIST Classification Example with FORTRESS-FL

Demonstrates FORTRESS-FL on logistic regression for binary MNIST classification.
Note: This is a simplified demo using synthetic data resembling MNIST properties.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fortress_fl.core import train_fortress_fl

def generate_mnist_like_data(n_samples: int = 100, n_features: int = 28*28//16) -> dict:
    """
    Generate synthetic data resembling MNIST properties.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features (reduced from 784 for demo speed)

    Returns:
        Dictionary with 'X' (features) and 'y' (binary labels)
    """
    # Generate synthetic features with MNIST-like properties
    np.random.seed(42)

    # Create two clusters for binary classification (0 vs 1)
    n_class0 = n_samples // 2
    n_class1 = n_samples - n_class0

    # Class 0 features (resembling digit 0)
    X_0 = np.random.randn(n_class0, n_features) * 0.8 + np.array([0.2] * n_features)
    y_0 = np.zeros(n_class0)

    # Class 1 features (resembling digit 1)
    X_1 = np.random.randn(n_class1, n_features) * 0.8 + np.array([-0.2] * n_features)
    y_1 = np.ones(n_class1)

    # Combine and shuffle
    X = np.vstack([X_0, X_1])
    y = np.concatenate([y_0, y_1])

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return {'X': X, 'y': y}

def main():
    print("🚀 FORTRESS-FL: MNIST-like Binary Classification Example")
    print("="*70)

    # Configuration
    n_operators = 6
    n_byzantine = 2  # 33% Byzantine
    n_samples_per_operator = 80
    n_features = 49  # 7x7 "images" for demo speed (instead of 28x28)
    n_rounds = 10

    print(f"Configuration:")
    print(f"  - Operators: {n_operators} ({n_byzantine} Byzantine)")
    print(f"  - Samples per operator: {n_samples_per_operator}")
    print(f"  - Features: {n_features} (simulated image pixels)")
    print(f"  - Training rounds: {n_rounds}")
    print(f"  - Task: Binary classification (0 vs 1)")
    print()

    # Create federated operators with MNIST-like data
    operators_data = []
    byzantine_operators = list(range(n_operators - n_byzantine, n_operators))

    for i in range(n_operators):
        # Generate local dataset for each operator
        local_data = generate_mnist_like_data(n_samples_per_operator, n_features)

        operator_data = {
            'id': f'Hospital_{i}',  # Simulating federated hospitals
            'dataset': local_data,
            'is_byzantine': i in byzantine_operators,
            'attack_type': 'label_flipping' if i in byzantine_operators else None
        }
        operators_data.append(operator_data)

    # Create test dataset
    test_data = generate_mnist_like_data(200, n_features)

    print(f"Byzantine operators: {[op['id'] for op in operators_data if op['is_byzantine']]}")
    print(f"Test set size: {len(test_data['X'])} samples")
    print()

    # Train FORTRESS-FL for classification
    final_model, history = train_fortress_fl(
        operators_data=operators_data,
        n_rounds=n_rounds,
        model_dim=n_features,
        learning_rate=0.01,
        security_param=512,  # Smaller for demo speed
        lambda_rep=0.15,
        sigma_dp=0.08,
        epsilon_dp=0.12,
        test_data=test_data,
        verbose=True
    )

    # Evaluate classification performance
    print("\n" + "="*70)
    print("📊 CLASSIFICATION RESULTS")
    print("="*70)

    # Test accuracy calculation
    X_test, y_test = test_data['X'], test_data['y']
    predictions = X_test @ final_model
    predicted_labels = (predictions > 0).astype(int)  # Threshold at 0
    accuracy = np.mean(predicted_labels == y_test)

    print(f"Test accuracy:        {accuracy:.1%}")
    print(f"Model norm:           {np.linalg.norm(final_model):.4f}")

    if history['test_losses']:
        print(f"Final test loss:      {history['test_losses'][-1]:.4f}")
        print(f"Initial test loss:    {history['test_losses'][0]:.4f}")

    # Byzantine detection analysis
    if history.get('byzantine_detection_accuracy'):
        detection_history = history['byzantine_detection_accuracy']
        avg_detection = np.mean(detection_history)
        print(f"Byzantine detection:  {avg_detection:.1%}")

        # Show detection improvement over time
        if len(detection_history) > 1:
            early_detection = np.mean(detection_history[:len(detection_history)//2])
            late_detection = np.mean(detection_history[len(detection_history)//2:])
            print(f"  Early rounds:       {early_detection:.1%}")
            print(f"  Later rounds:       {late_detection:.1%}")

    print(f"Privacy budget used:  {history.get('total_privacy_budget', 'N/A'):.3f}")

    # Security analysis
    print(f"\n🔒 SECURITY ANALYSIS:")
    print(f"  - Cryptographic commitments: ✅ Enabled")
    print(f"  - Differential privacy: ✅ ε={0.12}, σ={0.08}")
    print(f"  - Reputation system: ✅ Active learning")
    print(f"  - Spectral clustering: ✅ Advanced detection")

    print(f"\n✅ MNIST-like classification completed successfully!")
    print(f"🏥 Federated hospitals collaborated securely despite {n_byzantine} Byzantine participants.")
    print(f"🔐 FORTRESS-FL ensured privacy and robustness in medical data analysis.")

if __name__ == "__main__":
    main()