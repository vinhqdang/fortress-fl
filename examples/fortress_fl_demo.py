#!/usr/bin/env python3
"""
FORTRESS-FL Comprehensive Demo

This script demonstrates the complete FORTRESS-FL system with:
1. Multi-operator federated learning setup
2. Byzantine attack scenarios
3. Cryptographic commitments and verification
4. Spectral clustering for Byzantine detection
5. Reputation-based aggregation
6. Differential privacy
7. Performance evaluation and visualization

Usage:
    python fortress_fl_demo.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from fortress_fl.core import FortressFL, train_fortress_fl, create_federated_datasets
from fortress_fl.utils.attacks import AttackFactory, MultiAttackScenario
from fortress_fl.utils.evaluation import (
    evaluate_convergence, evaluate_byzantine_robustness,
    evaluate_privacy_utility_tradeoff, plot_training_metrics,
    generate_performance_report
)


def print_banner():
    """Print FORTRESS-FL banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          FORTRESS-FL DEMONSTRATION                           ║
║        Federated Operator Resilient Trustworthy Resource Efficient          ║
║                        Secure Slice Learning                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def demo_basic_fortress_fl():
    """Demonstrate basic FORTRESS-FL functionality."""
    print("\n🚀 DEMO 1: Basic FORTRESS-FL Training")
    print("="*70)

    # Setup parameters
    n_operators = 6
    byzantine_operators = [4, 5]  # Last 2 operators are Byzantine
    model_dim = 12
    n_rounds = 15
    n_samples_per_op = 200

    print(f"Configuration:")
    print(f"  - Operators: {n_operators}")
    print(f"  - Byzantine operators: {byzantine_operators}")
    print(f"  - Model dimension: {model_dim}")
    print(f"  - Training rounds: {n_rounds}")
    print(f"  - Samples per operator: {n_samples_per_op}")

    # Create federated datasets
    operators_data = create_federated_datasets(
        n_operators=n_operators,
        n_samples_per_operator=n_samples_per_op,
        n_features=model_dim,
        byzantine_operators=byzantine_operators,
        data_heterogeneity=0.2,  # Moderate heterogeneity
        task_type='regression'
    )

    # Create test data
    X_test = np.random.randn(300, model_dim)
    true_theta = np.random.randn(model_dim)
    y_test = X_test @ true_theta + np.random.normal(0, 0.1, 300)
    test_data = {'X': X_test, 'y': y_test}

    print(f"\n📊 Test data created: {X_test.shape[0]} samples")

    # Run FORTRESS-FL training
    print(f"\n🔄 Starting FORTRESS-FL training...")

    final_model, history = train_fortress_fl(
        operators_data=operators_data,
        n_rounds=n_rounds,
        model_dim=model_dim,
        learning_rate=0.02,
        security_param=1024,  # Moderate security for demo
        lambda_rep=0.15,
        sigma_dp=0.08,
        epsilon_dp=0.12,
        test_data=test_data,
        verbose=True
    )

    print(f"\n✅ Training completed!")
    print(f"Final model norm: {np.linalg.norm(final_model):.4f}")
    print(f"True model norm: {np.linalg.norm(true_theta):.4f}")
    print(f"Model error: {np.linalg.norm(final_model - true_theta):.4f}")

    # Evaluate Byzantine detection accuracy
    if history.get('byzantine_detection_accuracy'):
        avg_detection = np.mean(history['byzantine_detection_accuracy'])
        print(f"Average Byzantine detection accuracy: {avg_detection:.2%}")

    return final_model, history, test_data, byzantine_operators, true_theta


def demo_advanced_attacks():
    """Demonstrate advanced Byzantine attack scenarios."""
    print("\n🎯 DEMO 2: Advanced Byzantine Attacks")
    print("="*70)

    # Setup
    n_operators = 8
    model_dim = 10
    n_rounds = 12

    # Create multi-attack scenario
    attack_configs = [
        {
            'operator_ids': ['Operator_5', 'Operator_6'],
            'attack_type': 'sign_flip',
            'attack_strength': 1.2
        },
        {
            'operator_ids': ['Operator_7'],
            'attack_type': 'adaptive',
            'attack_strength': 1.0
        }
    ]

    multi_attack = MultiAttackScenario(attack_configs)

    print(f"Multi-attack scenario:")
    attack_summary = multi_attack.get_attack_summary()
    for attack_id, info in attack_summary.items():
        print(f"  - {attack_id}: {info}")

    # Create operator data with mixed attack types
    operators_data = []
    for i in range(n_operators):
        X = np.random.randn(150, model_dim)
        y = X @ np.random.randn(model_dim) + np.random.normal(0, 0.1, 150)

        if i >= 5:  # Byzantine operators
            is_byzantine = True
            attack_type = 'mixed'  # Will use MultiAttackScenario
        else:
            is_byzantine = False
            attack_type = None

        operator_data = {
            'id': f'Operator_{i}',
            'dataset': {'X': X, 'y': y},
            'is_byzantine': is_byzantine,
            'attack_type': attack_type
        }
        operators_data.append(operator_data)

    # Initialize FORTRESS-FL
    fortress_fl = FortressFL(
        n_operators=n_operators,
        model_dim=model_dim,
        security_param=1024,
        lambda_rep=0.12,
        sigma_dp=0.06,
        epsilon_dp=0.1
    )

    print(f"\n🔄 Running training with advanced attacks...")

    # Training loop with advanced attacks
    for round_idx in range(n_rounds):
        print(f"\nRound {round_idx + 1}:")

        local_gradients = []
        for op_data in operators_data:
            if op_data['is_byzantine'] and op_data['attack_type'] == 'mixed':
                # Use multi-attack scenario
                gradient = multi_attack.generate_gradient(
                    op_data['id'], model_dim, fortress_fl.get_global_model()
                )
            elif op_data['is_byzantine']:
                # Simple Byzantine
                gradient = np.random.randn(model_dim) * -0.5
            else:
                # Honest gradient
                gradient = np.random.randn(model_dim) * 0.08

            local_gradients.append(gradient)

        # Execute training round
        result = fortress_fl.train_round(local_gradients)

        # Update adaptive attacks based on detection
        if hasattr(multi_attack.attacks.get('adaptive_0'), 'set_detection_result'):
            adaptive_attack = multi_attack.attacks['adaptive_0']
            was_detected = 7 in result['byzantine_indices']  # Operator_7 index
            adaptive_attack.set_detection_result(was_detected)

        multi_attack.update_round()

    print(f"\n✅ Advanced attack demo completed!")

    # Analyze results
    byzantine_indices = [5, 6, 7]  # Ground truth
    final_result = fortress_fl.history['round_results'][-1]
    detected_indices = final_result['byzantine_indices']

    print(f"Ground truth Byzantine: {byzantine_indices}")
    print(f"Detected Byzantine: {detected_indices}")

    detection_accuracy = len(set(byzantine_indices) & set(detected_indices)) / len(byzantine_indices)
    print(f"Final detection accuracy: {detection_accuracy:.2%}")

    return fortress_fl


def demo_privacy_analysis():
    """Demonstrate privacy-utility tradeoff analysis."""
    print("\n🔒 DEMO 3: Privacy-Utility Tradeoff Analysis")
    print("="*70)

    # Test different privacy levels
    privacy_levels = [0.05, 0.1, 0.2, 0.5]
    results = {}

    # Setup
    n_operators = 5
    byzantine_operators = [3, 4]
    model_dim = 8
    n_rounds = 10

    # Create consistent dataset
    operators_data = create_federated_datasets(
        n_operators, 100, model_dim, byzantine_operators, task_type='regression'
    )

    # Test data
    X_test = np.random.randn(200, model_dim)
    true_theta = np.random.randn(model_dim)
    y_test = X_test @ true_theta + np.random.normal(0, 0.1, 200)
    test_data = {'X': X_test, 'y': y_test}

    print(f"Testing privacy levels: {privacy_levels}")

    for epsilon in privacy_levels:
        print(f"\n📊 Testing ε = {epsilon}")

        # Train with specific privacy level
        final_model, history = train_fortress_fl(
            operators_data=operators_data,
            n_rounds=n_rounds,
            model_dim=model_dim,
            learning_rate=0.01,
            security_param=512,  # Smaller for speed
            epsilon_dp=epsilon,
            test_data=test_data,
            verbose=False
        )

        # Evaluate performance
        test_loss = history['test_losses'][-1] if history['test_losses'] else None
        model_error = np.linalg.norm(final_model - true_theta)

        results[epsilon] = {
            'test_loss': test_loss,
            'model_error': model_error,
            'total_privacy_budget': epsilon * n_rounds
        }

        print(f"  Test loss: {test_loss:.4f}, Model error: {model_error:.4f}")

    # Plot privacy-utility tradeoff
    epsilons = list(results.keys())
    test_losses = [results[eps]['test_loss'] for eps in epsilons]
    model_errors = [results[eps]['model_error'] for eps in epsilons]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epsilons, test_losses, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Privacy Budget (ε per round)')
    plt.ylabel('Test Loss')
    plt.title('Privacy vs. Test Performance')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epsilons, model_errors, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Privacy Budget (ε per round)')
    plt.ylabel('Model Error ||θ - θ*||')
    plt.title('Privacy vs. Model Accuracy')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✅ Privacy analysis completed! Plot saved as 'privacy_utility_tradeoff.png'")

    return results


def demo_comprehensive_evaluation(fortress_fl, test_data, byzantine_operators, true_theta):
    """Demonstrate comprehensive evaluation and reporting."""
    print("\n📊 DEMO 4: Comprehensive Performance Evaluation")
    print("="*70)

    # Convergence analysis
    print(f"\n🔍 Convergence Analysis:")
    conv_eval = evaluate_convergence(fortress_fl, true_theta)
    for key, value in conv_eval.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Byzantine robustness analysis
    print(f"\n🛡️ Byzantine Robustness Analysis:")
    rob_eval = evaluate_byzantine_robustness(fortress_fl, byzantine_operators)
    for key, value in rob_eval.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Privacy analysis
    print(f"\n🔒 Privacy Analysis:")
    priv_eval = evaluate_privacy_utility_tradeoff(fortress_fl, test_data)
    for key, value in priv_eval.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Generate comprehensive report
    print(f"\n📋 Generating comprehensive performance report...")
    report = generate_performance_report(
        fortress_fl, test_data, byzantine_operators,
        save_path='fortress_fl_performance_report.json'
    )

    # Print executive summary
    print(f"\n🎯 Executive Summary:")
    summary = report['executive_summary']
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Create visualization
    print(f"\n📈 Creating training metrics visualization...")
    plot_training_metrics(fortress_fl, save_path='fortress_fl_training_metrics.png')

    return report


def demo_mpc_optimization():
    """Demonstrate MPC for cross-operator optimization."""
    print("\n🤝 DEMO 5: MPC Cross-Operator Optimization")
    print("="*70)

    from fortress_fl.crypto.mpc import secure_joint_optimization_mpc

    # Simulate interference matrices for two operators
    operator_A_data = {
        'interference_matrix': np.random.rand(4, 4) * 0.5,
        'power_range': (0.1, 2.0)
    }

    operator_B_data = {
        'interference_matrix': np.random.rand(4, 4) * 0.5,
        'power_range': (0.1, 2.0)
    }

    print(f"Operator A interference matrix shape: {operator_A_data['interference_matrix'].shape}")
    print(f"Operator B interference matrix shape: {operator_B_data['interference_matrix'].shape}")
    print(f"Power ranges: A={operator_A_data['power_range']}, B={operator_B_data['power_range']}")

    # Run secure joint optimization
    print(f"\n🔐 Running secure joint optimization...")
    optimal_power_A, optimal_power_B = secure_joint_optimization_mpc(
        operator_A_data, operator_B_data, n_iterations=15
    )

    print(f"\n✅ MPC optimization completed!")
    print(f"Optimal power allocation:")
    print(f"  Operator A: {optimal_power_A:.4f}")
    print(f"  Operator B: {optimal_power_B:.4f}")

    # Compute joint interference with optimal powers
    I_A = operator_A_data['interference_matrix'].flatten()
    I_B = operator_B_data['interference_matrix'].flatten()
    joint_interference = np.sum((I_A * optimal_power_A + I_B * optimal_power_B) ** 2)

    print(f"Final joint interference: {joint_interference:.4f}")

    return optimal_power_A, optimal_power_B


def main():
    """Run complete FORTRESS-FL demonstration."""
    print_banner()

    print("🎬 Welcome to the FORTRESS-FL comprehensive demonstration!")
    print("\nThis demo will showcase:")
    print("  1. Basic FORTRESS-FL training with Byzantine operators")
    print("  2. Advanced attack scenarios with adaptive attacks")
    print("  3. Privacy-utility tradeoff analysis")
    print("  4. Comprehensive performance evaluation")
    print("  5. MPC cross-operator optimization")

    input("\nPress Enter to start the demonstration...")

    try:
        # Demo 1: Basic FORTRESS-FL
        final_model, history, test_data, byzantine_operators, true_theta = demo_basic_fortress_fl()

        # Initialize FortressFL for further analysis
        fortress_fl = FortressFL(
            n_operators=6, model_dim=12, security_param=1024
        )
        # Simulate some training for analysis
        for i in range(3):
            gradients = [np.random.randn(12) * (0.1 if j < 4 else -0.2) for j in range(6)]
            fortress_fl.train_round(gradients)

        input("\nPress Enter to continue to Demo 2...")

        # Demo 2: Advanced attacks
        fortress_fl_advanced = demo_advanced_attacks()

        input("\nPress Enter to continue to Demo 3...")

        # Demo 3: Privacy analysis
        privacy_results = demo_privacy_analysis()

        input("\nPress Enter to continue to Demo 4...")

        # Demo 4: Comprehensive evaluation
        report = demo_comprehensive_evaluation(fortress_fl, test_data, byzantine_operators, true_theta)

        input("\nPress Enter to continue to Demo 5...")

        # Demo 5: MPC optimization
        power_A, power_B = demo_mpc_optimization()

        # Final summary
        print(f"\n🎊 FORTRESS-FL DEMONSTRATION COMPLETED!")
        print("="*70)
        print(f"✅ All demos executed successfully!")
        print(f"📁 Generated files:")
        print(f"  - fortress_fl_performance_report.json")
        print(f"  - fortress_fl_training_metrics.png")
        print(f"  - privacy_utility_tradeoff.png")

        print(f"\n🔑 Key Findings:")
        if 'executive_summary' in report:
            summary = report['executive_summary']
            print(f"  - Overall Assessment: {summary.get('overall_assessment', 'N/A')}")
            print(f"  - Detection Accuracy: {summary.get('avg_detection_accuracy', 0):.2%}")
            print(f"  - System Resilience: {summary.get('system_resilience', 0):.2%}")

        print(f"\n🌟 FORTRESS-FL successfully demonstrated Byzantine-robust")
        print(f"    federated learning with cryptographic commitments,")
        print(f"    spectral clustering, reputation systems, and differential privacy!")

    except KeyboardInterrupt:
        print(f"\n\n⏹️ Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n👋 Thank you for trying FORTRESS-FL!")


if __name__ == "__main__":
    main()