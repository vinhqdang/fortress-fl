#!/usr/bin/env python3
"""
Simple Linear Regression Example with FORTRESS-FL

Demonstrates FORTRESS-FL on a simple linear regression task with synthetic data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fortress_fl.core import train_fortress_fl, create_federated_datasets

def main():
    print("🚀 FORTRESS-FL: Simple Linear Regression Example")
    print("="*60)

    # Configuration
    n_operators = 5
    n_byzantine = 1  # 20% Byzantine
    n_samples_per_operator = 100
    model_dim = 3
    n_rounds = 8

    print(f"Configuration:")
    print(f"  - Operators: {n_operators} ({n_byzantine} Byzantine)")
    print(f"  - Samples per operator: {n_samples_per_operator}")
    print(f"  - Model dimension: {model_dim}")
    print(f"  - Training rounds: {n_rounds}")
    print()

    # Generate federated datasets for linear regression
    byzantine_operators = [n_operators - 1]  # Last operator is Byzantine

    operators_data = create_federated_datasets(
        n_operators=n_operators,
        n_samples_per_operator=n_samples_per_operator,
        n_features=model_dim,
        byzantine_operators=byzantine_operators,
        task_type='regression'
    )

    # Create test dataset with known ground truth
    np.random.seed(123)
    X_test = np.random.randn(50, model_dim)
    true_weights = np.array([2.5, -1.8, 0.7])  # Known ground truth
    y_test = X_test @ true_weights + np.random.normal(0, 0.1, 50)
    test_data = {'X': X_test, 'y': y_test, 'true_weights': true_weights}

    print(f"Ground truth weights: {true_weights}")
    print()

    # Train FORTRESS-FL
    final_model, history = train_fortress_fl(
        operators_data=operators_data,
        n_rounds=n_rounds,
        model_dim=model_dim,
        learning_rate=0.02,
        security_param=512,  # Smaller for demo speed
        lambda_rep=0.2,
        sigma_dp=0.05,
        epsilon_dp=0.1,
        test_data=test_data,
        verbose=True
    )

    # Evaluate results
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)

    print(f"Ground truth weights: {true_weights}")
    print(f"Learned weights:      {final_model}")
    print(f"Weight error:         {np.linalg.norm(final_model - true_weights):.4f}")

    if history['test_losses']:
        print(f"Final test loss:      {history['test_losses'][-1]:.4f}")
        print(f"Initial test loss:    {history['test_losses'][0]:.4f}")
        improvement = (history['test_losses'][0] - history['test_losses'][-1]) / history['test_losses'][0]
        print(f"Loss improvement:     {improvement:.1%}")

    # Byzantine detection analysis
    if history.get('byzantine_detection_accuracy'):
        avg_detection = np.mean(history['byzantine_detection_accuracy'])
        print(f"Avg Byzantine detection: {avg_detection:.1%}")

    print(f"Privacy budget used:  {history.get('total_privacy_budget', 'N/A')}")

    print("\n✅ Simple linear regression completed successfully!")
    print("🔒 FORTRESS-FL demonstrated Byzantine robustness with privacy protection.")

if __name__ == "__main__":
    main()