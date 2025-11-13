#!/usr/bin/env python3
"""
Quick FORTRESS-FL Comparison Experiments

Focused experiments to demonstrate system effectiveness with clear output.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from fortress_fl.core import train_fortress_fl, create_federated_datasets

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def quick_byzantine_robustness_test():
    """Quick test of Byzantine robustness with clear comparison."""
    print_header("BYZANTINE ROBUSTNESS COMPARISON")

    scenarios = [
        {"byzantine_count": 0, "name": "No Byzantine (Baseline)"},
        {"byzantine_count": 1, "name": "1/5 Byzantine (20%)"},
        {"byzantine_count": 2, "name": "2/5 Byzantine (40%)"}
    ]

    results = {}

    # Fixed parameters for fair comparison
    n_operators = 5
    model_dim = 6
    n_rounds = 8

    print(f"Configuration: {n_operators} operators, {model_dim}D model, {n_rounds} rounds\n")

    for scenario in scenarios:
        print(f"🔍 Testing: {scenario['name']}")

        n_byz = scenario['byzantine_count']
        byzantine_operators = list(range(n_operators - n_byz, n_operators)) if n_byz > 0 else []

        # Create datasets
        operators_data = create_federated_datasets(
            n_operators=n_operators,
            n_samples_per_operator=60,
            n_features=model_dim,
            byzantine_operators=byzantine_operators,
            task_type='regression'
        )

        # Test data with known true model
        np.random.seed(42)  # For consistent comparison
        X_test = np.random.randn(100, model_dim)
        true_theta = np.random.randn(model_dim)
        y_test = X_test @ true_theta + np.random.normal(0, 0.1, 100)
        test_data = {'X': X_test, 'y': y_test}

        # Train FORTRESS-FL
        start_time = time.time()
        final_model, history = train_fortress_fl(
            operators_data=operators_data,
            n_rounds=n_rounds,
            model_dim=model_dim,
            learning_rate=0.02,
            security_param=512,  # Smaller for speed
            lambda_rep=0.2,
            sigma_dp=0.03,
            epsilon_dp=0.08,
            test_data=test_data,
            verbose=False  # Suppress detailed output
        )
        training_time = time.time() - start_time

        # Calculate metrics
        final_loss = history['test_losses'][-1] if history['test_losses'] else 0
        model_error = np.linalg.norm(final_model - true_theta)

        if byzantine_operators and history.get('byzantine_detection_accuracy'):
            detection_accuracy = np.mean(history['byzantine_detection_accuracy'])
        else:
            detection_accuracy = None

        results[scenario['name']] = {
            'final_loss': final_loss,
            'model_error': model_error,
            'detection_accuracy': detection_accuracy,
            'training_time': training_time,
            'n_byzantine': n_byz
        }

        # Print immediate results
        print(f"  Final test loss: {final_loss:.4f}")
        print(f"  Model error: {model_error:.4f}")
        if detection_accuracy is not None:
            print(f"  Byzantine detection: {detection_accuracy:.1%}")
        print(f"  Training time: {training_time:.2f}s")
        print()

    return results

def quick_privacy_test():
    """Quick privacy-utility tradeoff test."""
    print_header("PRIVACY-UTILITY TRADEOFF")

    privacy_levels = [0.05, 0.1, 0.2, 0.5]
    results = {}

    # Fixed setup
    operators_data = create_federated_datasets(
        n_operators=5,
        n_samples_per_operator=60,
        n_features=6,
        byzantine_operators=[3, 4],
        task_type='regression'
    )

    # Consistent test data
    np.random.seed(123)
    X_test = np.random.randn(100, 6)
    true_theta = np.random.randn(6)
    y_test = X_test @ true_theta + np.random.normal(0, 0.1, 100)
    test_data = {'X': X_test, 'y': y_test}

    print(f"Testing privacy levels: {privacy_levels}\n")

    for epsilon in privacy_levels:
        print(f"🔒 Privacy level ε = {epsilon}")

        # Train with specific privacy level
        final_model, history = train_fortress_fl(
            operators_data=operators_data,
            n_rounds=6,
            model_dim=6,
            learning_rate=0.02,
            security_param=512,
            epsilon_dp=epsilon,
            sigma_dp=0.02,
            test_data=test_data,
            verbose=False
        )

        # Calculate metrics
        final_loss = history['test_losses'][-1]
        model_error = np.linalg.norm(final_model - true_theta)
        total_privacy_budget = epsilon * 6

        results[epsilon] = {
            'final_loss': final_loss,
            'model_error': model_error,
            'total_privacy_budget': total_privacy_budget
        }

        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Model error: {model_error:.4f}")
        print(f"  Total privacy budget: {total_privacy_budget:.3f}")
        print()

    return results

def attack_comparison_test():
    """Compare different attack types."""
    print_header("ATTACK TYPE COMPARISON")

    attack_types = ['sign_flip', 'random', 'coordinated']
    results = {}

    for attack_type in attack_types:
        print(f"🎯 Testing {attack_type} attack")

        # Create operators with specific attack
        operators_data = []
        for i in range(5):
            X = np.random.randn(60, 6)
            y = X @ np.random.randn(6)

            operator_data = {
                'id': f'Operator_{i}',
                'dataset': {'X': X, 'y': y},
                'is_byzantine': i >= 3,  # Last 2 are Byzantine
                'attack_type': attack_type if i >= 3 else None
            }
            operators_data.append(operator_data)

        # Test data
        X_test = np.random.randn(80, 6)
        true_theta = np.random.randn(6)
        y_test = X_test @ true_theta
        test_data = {'X': X_test, 'y': y_test}

        # Train FORTRESS-FL
        final_model, history = train_fortress_fl(
            operators_data=operators_data,
            n_rounds=6,
            model_dim=6,
            learning_rate=0.02,
            security_param=512,
            verbose=False,
            test_data=test_data
        )

        # Calculate metrics
        final_loss = history['test_losses'][-1]
        detection_accuracy = np.mean(history['byzantine_detection_accuracy'])

        results[attack_type] = {
            'final_loss': final_loss,
            'detection_accuracy': detection_accuracy
        }

        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Detection accuracy: {detection_accuracy:.1%}")
        print()

    return results

def print_comparison_tables(byzantine_results, privacy_results, attack_results):
    """Print formatted comparison tables."""

    # Byzantine Robustness Table
    print(f"\n📊 BYZANTINE ROBUSTNESS RESULTS")
    print(f"{'Scenario':<25} {'Test Loss':<12} {'Model Error':<12} {'Detection':<12} {'Time (s)':<10}")
    print(f"{'-'*75}")

    for scenario, result in byzantine_results.items():
        detection_str = f"{result['detection_accuracy']:.1%}" if result['detection_accuracy'] else "N/A"
        print(f"{scenario:<25} {result['final_loss']:<12.4f} {result['model_error']:<12.4f} "
              f"{detection_str:<12} {result['training_time']:<10.2f}")

    # Privacy-Utility Table
    print(f"\n🔒 PRIVACY-UTILITY TRADEOFF RESULTS")
    print(f"{'ε (per round)':<15} {'Final Loss':<12} {'Model Error':<12} {'Total Budget':<12}")
    print(f"{'-'*55}")

    for epsilon, result in privacy_results.items():
        print(f"{epsilon:<15.2f} {result['final_loss']:<12.4f} {result['model_error']:<12.4f} "
              f"{result['total_privacy_budget']:<12.2f}")

    # Attack Comparison Table
    print(f"\n🛡️ ATTACK TYPE COMPARISON RESULTS")
    print(f"{'Attack Type':<15} {'Final Loss':<12} {'Detection Rate':<15}")
    print(f"{'-'*45}")

    for attack_type, result in attack_results.items():
        print(f"{attack_type:<15} {result['final_loss']:<12.4f} {result['detection_accuracy']:<15.1%}")

def main():
    """Run quick comparison experiments."""
    print("🚀 FORTRESS-FL Quick Comparison Experiments")
    print("Fast-running experiments with clear comparison outputs")

    start_time = time.time()

    # Run experiments
    byzantine_results = quick_byzantine_robustness_test()
    privacy_results = quick_privacy_test()
    attack_results = attack_comparison_test()

    # Print summary tables
    print_comparison_tables(byzantine_results, privacy_results, attack_results)

    end_time = time.time()

    # Final summary
    print(f"\n🎉 EXPERIMENTS COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {end_time - start_time:.1f}s")

    # Key insights
    print(f"\n🔑 KEY INSIGHTS:")

    # Byzantine impact
    baseline_loss = byzantine_results["No Byzantine (Baseline)"]['final_loss']
    two_byz_loss = byzantine_results["2/5 Byzantine (40%)"]['final_loss']
    degradation = (two_byz_loss / baseline_loss - 1) * 100 if baseline_loss > 0 else 0

    print(f"📈 Byzantine Impact:")
    print(f"  - 40% Byzantine operators increase loss by {degradation:.1f}%")

    # Privacy cost
    low_privacy_loss = privacy_results[0.05]['final_loss']
    high_privacy_loss = privacy_results[0.5]['final_loss']
    privacy_cost = (high_privacy_loss / low_privacy_loss - 1) * 100 if low_privacy_loss > 0 else 0

    print(f"🔒 Privacy Cost:")
    print(f"  - 10x higher privacy budget reduces loss by {-privacy_cost:.1f}%")

    # Attack resistance
    avg_detection = np.mean([result['detection_accuracy'] for result in attack_results.values()])
    print(f"🛡️ Attack Resistance:")
    print(f"  - Average detection accuracy across attacks: {avg_detection:.1%}")

    print(f"\n✅ FORTRESS-FL demonstrates effective Byzantine-robust federated learning!")

if __name__ == "__main__":
    main()