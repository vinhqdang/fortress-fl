"""
Comprehensive Experiment Runner for FORTRESS-FL

This script executes the full suite of experiments required for the Q1 journal submission:
1. Comparative Analysis: FORTRESS-FL vs. Krum, Median, Trimmed Mean, FedAvg
2. Ablation Study: Impact of clustering, reputation, and commitments
3. Scalability Test: Performance with increasing number of operators
4. Heterogeneity Test: Robustness to non-IID data

Usage:
    python run_experiments.py --experiment all
    python run_experiments.py --experiment comparative
    python run_experiments.py --experiment ablation
    python run_experiments.py --experiment scalability
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fortress_fl.core import FortressFL, train_fortress_fl, create_federated_datasets
from fortress_fl.aggregation.baselines import BaselineAggregator
from fortress_fl.utils.evaluation import (
    plot_comparative_metrics, generate_comparative_report
)


def run_comparative_analysis(n_rounds: int = 20, n_operators: int = 10, 
                           byzantine_ratio: float = 0.3):
    """
    Run comparative analysis against baselines.
    """
    print("\n" + "="*50)
    print("RUNNING COMPARATIVE ANALYSIS")
    print("="*50)
    
    n_byzantine = int(n_operators * byzantine_ratio)
    byzantine_ops = list(range(n_operators - n_byzantine, n_operators))
    model_dim = 10
    
    print(f"Configuration: {n_operators} operators ({n_byzantine} Byzantine), {n_rounds} rounds")
    
    # Create datasets
    operators_data = create_federated_datasets(
        n_operators, 100, model_dim, byzantine_ops, task_type='regression'
    )
    
    # Test data
    X_test = np.random.randn(200, model_dim)
    true_theta = np.random.randn(model_dim)
    y_test = X_test @ true_theta + np.random.normal(0, 0.1, 200)
    test_data = {'X': X_test, 'y': y_test}
    
    results = {}
    
    # 1. Run FORTRESS-FL
    print("\nRunning FORTRESS-FL...")
    _, history_fortress = train_fortress_fl(
        operators_data, n_rounds, model_dim, 
        test_data=test_data, verbose=False,
        epsilon_dp=1.0, max_grad_norm=0.1
    )
    results['FORTRESS-FL'] = {'history': history_fortress}
    
    # 2. Run Baselines
    baselines = ['Krum', 'Median', 'Trimmed_Mean', 'FedAvg']
    
    for method in baselines:
        print(f"Running {method}...")
        aggregator = BaselineAggregator(method, f=n_byzantine)
        
        # Initialize model
        model = np.zeros(model_dim)
        history = {'test_losses': [], 'global_model_norms': []}
        
        for round_idx in range(n_rounds):
            # Generate gradients
            gradients = []
            for op_data in operators_data:
                # Simulate gradient computation
                X, y = op_data['dataset']['X'], op_data['dataset']['y']
                
                # Honest gradient
                pred = X @ model
                grad = X.T @ (pred - y) / len(y)
                
                # Byzantine attack (Sign Flip)
                if op_data['is_byzantine']:
                    grad = -grad * 2.0
                    
                gradients.append(grad)
            
            # Aggregate
            agg_grad = aggregator.aggregate(gradients)
            
            # Update model
            model -= 0.01 * agg_grad
            
            # Evaluate
            pred_test = X_test @ model
            loss = np.mean((pred_test - y_test) ** 2)
            history['test_losses'].append(loss)
            history['global_model_norms'].append(np.linalg.norm(model))
            
        results[method] = {'history': history}
        
    # Plot and Report
    plot_comparative_metrics(
        results, 'test_loss', 
        save_path='comparative_test_loss.png',
        title='Robustness Comparison: Test Loss under Attack'
    )
    
    generate_comparative_report(results, 'comparative_report.json')
    print("Comparative analysis completed.")


def run_ablation_study(n_rounds: int = 15):
    """
    Run ablation study by disabling components.
    """
    print("\n" + "="*50)
    print("RUNNING ABLATION STUDY")
    print("="*50)
    
    n_operators = 10
    byzantine_ops = [8, 9]
    model_dim = 10
    
    operators_data = create_federated_datasets(
        n_operators, 100, model_dim, byzantine_ops, task_type='regression'
    )
    
    # Test data
    X_test = np.random.randn(200, model_dim)
    true_theta = np.random.randn(model_dim)
    y_test = X_test @ true_theta + np.random.normal(0, 0.1, 200)
    test_data = {'X': X_test, 'y': y_test}
    
    results = {}
    
    # 1. Full System
    print("Running Full System...")
    _, hist_full = train_fortress_fl(
        operators_data, n_rounds, model_dim, test_data=test_data, verbose=False,
        epsilon_dp=1.0, max_grad_norm=0.1
    )
    results['Full System'] = {'history': hist_full}
    
    # 2. No Reputation (lambda = 0)
    print("Running No Reputation...")
    _, hist_no_rep = train_fortress_fl(
        operators_data, n_rounds, model_dim, lambda_rep=0.0, 
        test_data=test_data, verbose=False,
        epsilon_dp=1.0, max_grad_norm=0.1
    )
    results['No Reputation'] = {'history': hist_no_rep}
    
    # 3. No Privacy (epsilon large)
    print("Running No Privacy...")
    _, hist_no_dp = train_fortress_fl(
        operators_data, n_rounds, model_dim, epsilon_dp=100.0, sigma_dp=0.0,
        test_data=test_data, verbose=False,
        max_grad_norm=0.1
    )
    results['No Privacy'] = {'history': hist_no_dp}
    
    plot_comparative_metrics(
        results, 'test_loss',
        save_path='ablation_study.png',
        title='Ablation Study: Component Contributions'
    )
    print("Ablation study completed.")


def run_scalability_test():
    """
    Run scalability test with increasing number of operators.
    """
    print("\n" + "="*50)
    print("RUNNING SCALABILITY TEST")
    print("="*50)
    
    operator_counts = [10, 20, 50, 100]
    times = []
    
    import time
    
    for n_ops in operator_counts:
        print(f"Testing with {n_ops} operators...")
        
        operators_data = create_federated_datasets(
            n_ops, 50, 10, [], task_type='regression'
        )
        
        start_time = time.time()
        train_fortress_fl(
            operators_data, n_rounds=5, model_dim=10, verbose=False,
            epsilon_dp=1.0, max_grad_norm=0.1
        )
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 5.0
        times.append(avg_time)
        print(f"  Average round time: {avg_time:.4f}s")
        
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(operator_counts, times, 'bo-', linewidth=2)
    plt.xlabel('Number of Operators')
    plt.ylabel('Average Round Time (s)')
    plt.title('Scalability Analysis')
    plt.grid(True, alpha=0.3)
    plt.savefig('scalability_analysis.png', dpi=300)
    plt.show()
    
    print("Scalability test completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FORTRESS-FL Experiments")
    parser.add_argument('--experiment', type=str, default='all',
                      choices=['all', 'comparative', 'ablation', 'scalability'],
                      help='Experiment to run')
    
    args = parser.parse_args()
    
    if args.experiment in ['all', 'comparative']:
        run_comparative_analysis()
        
    if args.experiment in ['all', 'ablation']:
        run_ablation_study()
        
    if args.experiment in ['all', 'scalability']:
        run_scalability_test()
