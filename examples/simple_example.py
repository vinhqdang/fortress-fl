#!/usr/bin/env python3
"""
Simple FORTRESS-FL Example

A minimal example demonstrating FORTRESS-FL for linear regression
with Byzantine-robust federated learning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fortress_fl.core import train_fortress_fl, create_federated_datasets

def main():
    print("🚀 Simple FORTRESS-FL Example")
    print("="*40)

    # Parameters
    n_operators = 5
    byzantine_operators = [3, 4]  # Last 2 are Byzantine
    model_dim = 8
    n_rounds = 10
    n_samples = 100

    print(f"Setup:")
    print(f"  Operators: {n_operators}")
    print(f"  Byzantine: {byzantine_operators}")
    print(f"  Model dimension: {model_dim}")
    print(f"  Training rounds: {n_rounds}")

    # Create datasets
    operators_data = create_federated_datasets(
        n_operators=n_operators,
        n_samples_per_operator=n_samples,
        n_features=model_dim,
        byzantine_operators=byzantine_operators,
        task_type='regression'
    )

    # Create test data
    X_test = np.random.randn(200, model_dim)
    true_theta = np.random.randn(model_dim)
    y_test = X_test @ true_theta + np.random.normal(0, 0.1, 200)
    test_data = {'X': X_test, 'y': y_test}

    # Train FORTRESS-FL
    print(f"\n🔄 Training FORTRESS-FL...")

    final_model, history = train_fortress_fl(
        operators_data=operators_data,
        n_rounds=n_rounds,
        model_dim=model_dim,
        learning_rate=0.01,
        security_param=512,  # Small for demo
        lambda_rep=0.1,
        sigma_dp=0.05,
        epsilon_dp=0.1,
        test_data=test_data,
        verbose=True
    )

    # Results
    print(f"\n✅ Training completed!")
    print(f"Final model norm: {np.linalg.norm(final_model):.4f}")
    print(f"True model norm: {np.linalg.norm(true_theta):.4f}")

    if history['test_losses']:
        print(f"Test loss: {history['test_losses'][0]:.4f} → {history['test_losses'][-1]:.4f}")

    if history['byzantine_detection_accuracy']:
        avg_detection = np.mean(history['byzantine_detection_accuracy'])
        print(f"Byzantine detection accuracy: {avg_detection:.1%}")

    print(f"\n🎉 FORTRESS-FL simple example completed!")

if __name__ == "__main__":
    main()