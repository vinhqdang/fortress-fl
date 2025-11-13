#!/usr/bin/env python3
"""
End-to-End Tests for FORTRESS-FL

Comprehensive integration tests for the complete FORTRESS-FL system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import tempfile
from fortress_fl.core import FortressFL, train_fortress_fl, create_federated_datasets
from fortress_fl.utils.attacks import AttackFactory
from fortress_fl.utils.evaluation import (
    evaluate_convergence, evaluate_byzantine_robustness,
    evaluate_privacy_utility_tradeoff, generate_performance_report
)


class TestFortressFLCore(unittest.TestCase):
    """Test core FORTRESS-FL functionality."""

    def setUp(self):
        """Set up test parameters."""
        self.n_operators = 5
        self.model_dim = 8
        self.security_param = 512  # Small for testing

    def test_fortress_fl_initialization(self):
        """Test FORTRESS-FL initialization."""
        fortress_fl = FortressFL(
            n_operators=self.n_operators,
            model_dim=self.model_dim,
            security_param=self.security_param
        )

        # Check initialization
        self.assertEqual(fortress_fl.n_operators, self.n_operators)
        self.assertEqual(fortress_fl.model_dim, self.model_dim)
        self.assertEqual(len(fortress_fl.operator_ids), self.n_operators)
        self.assertEqual(fortress_fl.global_model.shape, (self.model_dim,))
        self.assertEqual(fortress_fl.round_number, 0)

    def test_single_training_round(self):
        """Test single training round execution."""
        fortress_fl = FortressFL(
            n_operators=self.n_operators,
            model_dim=self.model_dim,
            security_param=self.security_param
        )

        # Create test gradients
        gradients = [np.random.randn(self.model_dim) * 0.1 for _ in range(self.n_operators)]

        # Execute training round
        result = fortress_fl.train_round(gradients, learning_rate=0.01)

        # Check result structure
        required_keys = [
            'aggregated_gradient', 'updated_reputations', 'honest_indices',
            'byzantine_indices', 'round_number', 'global_model'
        ]
        for key in required_keys:
            self.assertIn(key, result)

        # Check that round number incremented
        self.assertEqual(fortress_fl.round_number, 1)

        # Check global model was updated
        self.assertFalse(np.allclose(fortress_fl.global_model, np.zeros(self.model_dim)))

    def test_multiple_training_rounds(self):
        """Test multiple training rounds."""
        fortress_fl = FortressFL(
            n_operators=self.n_operators,
            model_dim=self.model_dim,
            security_param=self.security_param,
            lambda_rep=0.2
        )

        n_rounds = 5
        for round_idx in range(n_rounds):
            # Create gradients with some Byzantine operators
            gradients = []
            for i in range(self.n_operators):
                if i < 3:  # Honest
                    gradient = np.random.randn(self.model_dim) * 0.1
                else:  # Byzantine
                    gradient = -np.random.randn(self.model_dim) * 0.2

                gradients.append(gradient)

            result = fortress_fl.train_round(gradients)

            # Check round progression
            self.assertEqual(result['round_number'], round_idx + 1)

        # Check history accumulation
        self.assertEqual(len(fortress_fl.history['round_results']), n_rounds)
        self.assertEqual(fortress_fl.round_number, n_rounds)

    def test_model_persistence(self):
        """Test model saving and loading."""
        fortress_fl = FortressFL(
            n_operators=self.n_operators,
            model_dim=self.model_dim,
            security_param=self.security_param
        )

        # Train for a few rounds
        for _ in range(3):
            gradients = [np.random.randn(self.model_dim) * 0.1 for _ in range(self.n_operators)]
            fortress_fl.train_round(gradients)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            save_path = tmp_file.name

        try:
            fortress_fl.save_model(save_path)

            # Load model
            loaded_fortress_fl = FortressFL.load_model(save_path)

            # Check that key attributes are preserved
            np.testing.assert_array_equal(fortress_fl.global_model, loaded_fortress_fl.global_model)
            self.assertEqual(fortress_fl.round_number, loaded_fortress_fl.round_number)
            self.assertEqual(fortress_fl.operator_ids, loaded_fortress_fl.operator_ids)

        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestByzantineRobustness(unittest.TestCase):
    """Test Byzantine robustness of FORTRESS-FL."""

    def setUp(self):
        """Set up Byzantine test scenarios."""
        self.n_operators = 6
        self.byzantine_operators = [4, 5]
        self.model_dim = 10
        self.security_param = 512

    def test_sign_flip_attack_detection(self):
        """Test detection of sign flip attacks."""
        fortress_fl = FortressFL(
            n_operators=self.n_operators,
            model_dim=self.model_dim,
            security_param=self.security_param
        )

        n_rounds = 8
        detection_results = []

        for round_idx in range(n_rounds):
            gradients = []
            honest_base = np.random.randn(self.model_dim) * 0.1

            for i in range(self.n_operators):
                if i in self.byzantine_operators:
                    # Sign flip attack
                    gradient = -honest_base + np.random.randn(self.model_dim) * 0.02
                else:
                    # Honest gradient
                    gradient = honest_base + np.random.randn(self.model_dim) * 0.02

                gradients.append(gradient)

            result = fortress_fl.train_round(gradients)

            # Check detection accuracy
            detected = set(result['byzantine_indices'])
            expected = set(self.byzantine_operators)
            accuracy = len(detected & expected) / len(expected) if expected else 1.0
            detection_results.append(accuracy)

        # Should achieve reasonable detection accuracy over time
        avg_detection = np.mean(detection_results[-5:])  # Last 5 rounds
        self.assertGreater(avg_detection, 0.3)  # At least 30% detection

    def test_adaptive_attack_resilience(self):
        """Test resilience against adaptive attacks."""
        fortress_fl = FortressFL(
            n_operators=self.n_operators,
            model_dim=self.model_dim,
            security_param=self.security_param
        )

        # Create adaptive attack
        adaptive_attack = AttackFactory.create_attack('adaptive', attack_strength=1.0)

        n_rounds = 10
        reputation_evolution = []

        for round_idx in range(n_rounds):
            gradients = []

            for i in range(self.n_operators):
                if i == self.byzantine_operators[0]:  # Only one adaptive attacker
                    gradient = adaptive_attack.generate_gradient(
                        self.model_dim, fortress_fl.get_global_model()
                    )
                else:
                    # Honest gradient
                    gradient = np.random.randn(self.model_dim) * 0.1

                gradients.append(gradient)

            result = fortress_fl.train_round(gradients)

            # Update adaptive attack based on detection
            was_detected = self.byzantine_operators[0] in result['byzantine_indices']
            adaptive_attack.set_detection_result(was_detected)

            # Track reputation of adaptive attacker
            attacker_id = fortress_fl.operator_ids[self.byzantine_operators[0]]
            reputation = result['updated_reputations'][attacker_id]
            reputation_evolution.append(reputation)

        # Adaptive attacker's reputation should eventually decrease
        final_reputation = reputation_evolution[-1]
        self.assertLess(final_reputation, 0.4)  # Should be penalized

    def test_multiple_attack_types(self):
        """Test system with multiple attack types."""
        fortress_fl = FortressFL(
            n_operators=7,  # More operators for multiple attacks
            model_dim=self.model_dim,
            security_param=self.security_param
        )

        # Create different attacks
        sign_flip_attack = AttackFactory.create_attack('sign_flip', 1.0)
        random_attack = AttackFactory.create_attack('random', 0.8)

        n_rounds = 6
        for round_idx in range(n_rounds):
            gradients = []

            for i in range(7):
                if i == 4:  # Sign flip attacker
                    gradient = sign_flip_attack.generate_gradient(self.model_dim)
                elif i == 5:  # Random attacker
                    gradient = random_attack.generate_gradient(self.model_dim)
                elif i == 6:  # Zero gradient attacker
                    gradient = np.zeros(self.model_dim)
                else:  # Honest operators
                    gradient = np.random.randn(self.model_dim) * 0.1

                gradients.append(gradient)

            result = fortress_fl.train_round(gradients)

            # System should continue to function
            self.assertIsNotNone(result['aggregated_gradient'])
            self.assertGreater(result['n_honest'], 0)


class TestPrivacyAndSecurity(unittest.TestCase):
    """Test privacy and security properties."""

    def test_differential_privacy_noise(self):
        """Test differential privacy noise addition."""
        fortress_fl = FortressFL(
            n_operators=4, model_dim=6, security_param=512,
            sigma_dp=0.1, epsilon_dp=0.1
        )

        # Run same gradients twice
        gradients = [np.random.randn(6) * 0.1 for _ in range(4)]

        result1 = fortress_fl.train_round(gradients.copy())
        result2 = fortress_fl.train_round(gradients.copy())

        # Results should be different due to DP noise
        diff = np.linalg.norm(result1['aggregated_gradient'] - result2['aggregated_gradient'])
        self.assertGreater(diff, 1e-6)  # Should be different due to noise

    def test_privacy_budget_tracking(self):
        """Test privacy budget consumption tracking."""
        epsilon_per_round = 0.05
        fortress_fl = FortressFL(
            n_operators=4, model_dim=6, security_param=512,
            epsilon_dp=epsilon_per_round
        )

        n_rounds = 5
        for _ in range(n_rounds):
            gradients = [np.random.randn(6) * 0.1 for _ in range(4)]
            fortress_fl.train_round(gradients)

        # Check privacy budget accumulation
        expected_budget = epsilon_per_round * n_rounds
        self.assertAlmostEqual(fortress_fl.total_privacy_budget, expected_budget, places=6)

    def test_commitment_verification_security(self):
        """Test commitment verification prevents cheating."""
        fortress_fl = FortressFL(
            n_operators=3, model_dim=4, security_param=512
        )

        # Create gradients and commitments
        from fortress_fl.crypto.pedersen import commit_gradient

        gradients = [np.random.randn(4) * 0.1 for _ in range(3)]
        commitments = []
        openings = []

        for gradient in gradients:
            commitment, opening = commit_gradient(gradient, *fortress_fl.comm_params)
            commitments.append(commitment)
            openings.append(opening)

        # Simulate cheating: operator 1 changes gradient after commitment
        cheating_gradients = gradients.copy()
        cheating_gradients[1] += 0.5  # Significant change

        # Run aggregation with original commitments but cheating gradients
        from fortress_fl.aggregation.trustchain import trustchain_aggregation

        reputations = {f"Op_{i}": 0.5 for i in range(3)}
        dp_params = {'sigma': 0.01, 'epsilon': 0.1}

        result = trustchain_aggregation(
            cheating_gradients, commitments, openings, reputations,
            [f"Op_{i}" for i in range(3)], fortress_fl.comm_params, dp_params
        )

        # Cheating operator should fail verification
        self.assertEqual(result['verification_results'][1], False)
        self.assertIn(1, result['byzantine_indices'])


class TestHighLevelTraining(unittest.TestCase):
    """Test high-level training functions."""

    def test_train_fortress_fl_function(self):
        """Test the high-level train_fortress_fl function."""
        # Create test datasets
        n_operators = 5
        byzantine_operators = [3, 4]
        model_dim = 8
        n_rounds = 5

        operators_data = create_federated_datasets(
            n_operators=n_operators,
            n_samples_per_operator=50,
            n_features=model_dim,
            byzantine_operators=byzantine_operators,
            task_type='regression'
        )

        # Create test data
        X_test = np.random.randn(100, model_dim)
        y_test = X_test @ np.random.randn(model_dim)
        test_data = {'X': X_test, 'y': y_test}

        # Run training
        final_model, history = train_fortress_fl(
            operators_data=operators_data,
            n_rounds=n_rounds,
            model_dim=model_dim,
            learning_rate=0.01,
            security_param=512,
            test_data=test_data,
            verbose=False
        )

        # Check results
        self.assertEqual(final_model.shape, (model_dim,))
        self.assertIn('test_losses', history)
        self.assertIn('byzantine_detection_accuracy', history)

        # Should have test losses for each round
        self.assertEqual(len(history['test_losses']), n_rounds)

    def test_evaluation_functions(self):
        """Test evaluation and analysis functions."""
        # Create a trained model for evaluation
        fortress_fl = FortressFL(n_operators=5, model_dim=6, security_param=512)

        # Simulate training
        byzantine_operators = [3, 4]
        for _ in range(5):
            gradients = []
            for i in range(5):
                if i in byzantine_operators:
                    gradient = -np.random.randn(6) * 0.2
                else:
                    gradient = np.random.randn(6) * 0.1
                gradients.append(gradient)

            fortress_fl.train_round(gradients)

        # Create test data
        X_test = np.random.randn(100, 6)
        y_test = X_test @ np.random.randn(6)
        test_data = {'X': X_test, 'y': y_test}

        # Test evaluation functions
        conv_eval = evaluate_convergence(fortress_fl)
        self.assertIn('final_model_norm', conv_eval)

        rob_eval = evaluate_byzantine_robustness(fortress_fl, byzantine_operators)
        self.assertIn('avg_detection_accuracy', rob_eval)

        priv_eval = evaluate_privacy_utility_tradeoff(fortress_fl, test_data)
        self.assertIn('total_privacy_budget', priv_eval)

        # Test comprehensive report generation
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            report_path = tmp_file.name

        try:
            report = generate_performance_report(
                fortress_fl, test_data, byzantine_operators, save_path=report_path
            )

            self.assertIn('system_info', report)
            self.assertIn('executive_summary', report)
            self.assertTrue(os.path.exists(report_path))

        finally:
            if os.path.exists(report_path):
                os.unlink(report_path)


class TestSystemScalability(unittest.TestCase):
    """Test system scalability and performance."""

    def test_scaling_with_operators(self):
        """Test system performance with different numbers of operators."""
        model_dim = 6
        operator_counts = [5, 10, 15]

        for n_ops in operator_counts:
            with self.subTest(n_operators=n_ops):
                fortress_fl = FortressFL(
                    n_operators=n_ops, model_dim=model_dim, security_param=512
                )

                # Single round should complete successfully
                gradients = [np.random.randn(model_dim) * 0.1 for _ in range(n_ops)]
                result = fortress_fl.train_round(gradients)

                self.assertIsNotNone(result['aggregated_gradient'])
                self.assertEqual(len(result['updated_reputations']), n_ops)

    def test_scaling_with_model_dimension(self):
        """Test system performance with different model dimensions."""
        n_operators = 5
        model_dims = [5, 10, 20]

        for model_dim in model_dims:
            with self.subTest(model_dim=model_dim):
                fortress_fl = FortressFL(
                    n_operators=n_operators, model_dim=model_dim, security_param=512
                )

                gradients = [np.random.randn(model_dim) * 0.1 for _ in range(n_operators)]
                result = fortress_fl.train_round(gradients)

                self.assertEqual(result['aggregated_gradient'].shape, (model_dim,))


def run_end_to_end_tests():
    """Run all end-to-end tests."""
    print("🧪 Running FORTRESS-FL End-to-End Tests")
    print("="*50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFortressFLCore))
    suite.addTests(loader.loadTestsFromTestCase(TestByzantineRobustness))
    suite.addTests(loader.loadTestsFromTestCase(TestPrivacyAndSecurity))
    suite.addTests(loader.loadTestsFromTestCase(TestHighLevelTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemScalability))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print(f"\n✅ All end-to-end tests passed!")
    else:
        print(f"\n❌ Some tests failed!")

    return success


if __name__ == "__main__":
    run_end_to_end_tests()