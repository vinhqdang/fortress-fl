#!/usr/bin/env python3
"""
FORTRESS-FL Test Runner

Run all tests for the FORTRESS-FL system with comprehensive reporting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import subprocess
from test_crypto import run_crypto_tests
from test_aggregation import run_aggregation_tests
from test_end_to_end import run_end_to_end_tests


def print_test_header():
    """Print test suite header."""
    header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        FORTRESS-FL TEST SUITE                               ║
║        Comprehensive Testing for Byzantine-Robust Federated Learning        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(header)


def run_single_test_module(test_name, test_function):
    """Run a single test module and return results."""
    print(f"\n🧪 Running {test_name}...")
    print("─" * 60)

    start_time = time.time()
    success = test_function()
    end_time = time.time()

    duration = end_time - start_time
    status = "✅ PASSED" if success else "❌ FAILED"

    print(f"\n{status} - {test_name} completed in {duration:.2f}s")

    return success, duration


def check_environment():
    """Check if the environment is properly set up."""
    print("🔍 Checking test environment...")

    try:
        import numpy
        import scipy
        import sklearn
        import matplotlib
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        return False


def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("\n📊 Running Performance Benchmarks...")
    print("─" * 60)

    try:
        from fortress_fl.core import FortressFL
        import numpy as np

        # Benchmark 1: Basic training round performance
        print("Benchmark 1: Basic Training Round Performance")

        fortress_fl = FortressFL(n_operators=10, model_dim=20, security_param=512)
        gradients = [np.random.randn(20) * 0.1 for _ in range(10)]

        start_time = time.time()
        for _ in range(5):  # 5 rounds
            result = fortress_fl.train_round(gradients)
        end_time = time.time()

        avg_round_time = (end_time - start_time) / 5
        print(f"  Average round time (10 operators, 20D model): {avg_round_time:.3f}s")

        # Benchmark 2: Cryptographic operations
        print("\nBenchmark 2: Cryptographic Operations Performance")
        from fortress_fl.crypto.pedersen import setup_pedersen_commitment, commit_gradient

        start_time = time.time()
        comm_params = setup_pedersen_commitment(1024)
        setup_time = time.time() - start_time

        gradient = np.random.randn(20)
        start_time = time.time()
        for _ in range(100):  # 100 commitments
            commit_gradient(gradient, *comm_params)
        end_time = time.time()

        avg_commit_time = (end_time - start_time) / 100
        print(f"  Commitment setup time (1024-bit): {setup_time:.3f}s")
        print(f"  Average commitment time: {avg_commit_time*1000:.3f}ms")

        # Benchmark 3: Spectral clustering performance
        print("\nBenchmark 3: Spectral Clustering Performance")
        from fortress_fl.aggregation.spectral_clustering import spectral_clustering_byzantine_detection

        gradients_large = [np.random.randn(50) for _ in range(20)]  # 20 operators, 50D

        start_time = time.time()
        cluster_labels, byzantine_cluster_id = spectral_clustering_byzantine_detection(gradients_large)
        end_time = time.time()

        clustering_time = end_time - start_time
        print(f"  Spectral clustering time (20 operators, 50D): {clustering_time:.3f}s")

        print("\n✅ Performance benchmarks completed")
        return True

    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


def generate_test_report(results):
    """Generate a comprehensive test report."""
    print("\n📋 FORTRESS-FL TEST REPORT")
    print("=" * 70)

    total_tests = len(results)
    passed_tests = sum(1 for success, _ in results.values() if success)
    total_time = sum(duration for _, duration in results.values())

    print(f"Test Summary:")
    print(f"  Total test modules: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {total_tests - passed_tests}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Success rate: {passed_tests/total_tests*100:.1f}%")

    print(f"\nDetailed Results:")
    for test_name, (success, duration) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:<30} {status:>10} ({duration:>6.2f}s)")

    # Overall assessment
    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! FORTRESS-FL is ready for deployment.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test module(s) failed. Please review the failures above.")
        return False


def main():
    """Main test runner."""
    print_test_header()

    print("🚀 Starting FORTRESS-FL comprehensive test suite...")

    # Check environment
    if not check_environment():
        print("❌ Environment check failed. Please install missing dependencies.")
        sys.exit(1)

    # Define test modules
    test_modules = [
        ("Cryptographic Components", run_crypto_tests),
        ("Aggregation Algorithms", run_aggregation_tests),
        ("End-to-End Integration", run_end_to_end_tests),
    ]

    # Run all test modules
    results = {}
    overall_start_time = time.time()

    for test_name, test_function in test_modules:
        success, duration = run_single_test_module(test_name, test_function)
        results[test_name] = (success, duration)

    # Run performance benchmarks
    print("\n" + "="*70)
    benchmark_success = run_performance_benchmarks()

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time

    # Generate comprehensive report
    print("\n" + "="*70)
    all_success = generate_test_report(results)

    print(f"\n⏱️  Total test suite execution time: {total_duration:.2f}s")

    if benchmark_success:
        print("📊 Performance benchmarks: ✅ PASSED")
    else:
        print("📊 Performance benchmarks: ❌ FAILED")

    # Final status
    if all_success and benchmark_success:
        print(f"\n🌟 FORTRESS-FL TEST SUITE: ✅ ALL TESTS PASSED")
        print("🔒 System is ready for Byzantine-robust federated learning!")
        sys.exit(0)
    else:
        print(f"\n💥 FORTRESS-FL TEST SUITE: ❌ SOME TESTS FAILED")
        print("🔧 Please review and fix the failing tests before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Test suite interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)