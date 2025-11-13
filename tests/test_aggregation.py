#!/usr/bin/env python3
"""
Tests for FORTRESS-FL Aggregation Components

Test suite for spectral clustering, reputation system, and TrustChain aggregation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from fortress_fl.aggregation.spectral_clustering import (
    compute_similarity_matrix, spectral_clustering_byzantine_detection,
    identify_byzantine_cluster, filter_byzantine_gradients
)
from fortress_fl.aggregation.reputation import (
    compute_quality_score, update_reputations, select_operators_by_reputation,
    compute_reputation_weights, ReputationTracker
)
from fortress_fl.aggregation.trustchain import (
    trustchain_aggregation, robust_mean_aggregation, TrustChainAggregator
)
from fortress_fl.crypto.pedersen import setup_pedersen_commitment, commit_gradient


class TestSpectralClustering(unittest.TestCase):
    """Test spectral clustering for Byzantine detection."""

    def setUp(self):
        """Set up test gradients."""
        self.gradient_dim = 8
        np.random.seed(42)  # For reproducibility

    def test_similarity_matrix(self):
        """Test similarity matrix computation."""
        # Create test gradients
        gradients = [
            np.array([1.0, 0.0, 0.0]),  # Gradient 1
            np.array([0.8, 0.6, 0.0]),  # Similar to gradient 1
            np.array([-1.0, 0.0, 0.0])  # Opposite to gradient 1
        ]

        S = compute_similarity_matrix(gradients)

        # Check properties
        self.assertEqual(S.shape, (3, 3))

        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(S), [1.0, 1.0, 1.0])

        # Symmetric matrix
        np.testing.assert_array_almost_equal(S, S.T)

        # Check specific values
        self.assertAlmostEqual(S[0, 1], 0.8, places=1)  # Similar gradients
        self.assertAlmostEqual(S[0, 2], -1.0, places=1)  # Opposite gradients

    def test_spectral_clustering_simple(self):
        """Test spectral clustering with simple case."""
        # Create honest and Byzantine gradients
        honest_base = np.random.randn(self.gradient_dim)
        honest_gradients = [honest_base + 0.1 * np.random.randn(self.gradient_dim) for _ in range(3)]
        byzantine_gradients = [-grad for grad in honest_gradients[:2]]  # Sign flip

        all_gradients = honest_gradients + byzantine_gradients
        n_total = len(all_gradients)

        # Perform clustering
        cluster_labels, byzantine_cluster_id = spectral_clustering_byzantine_detection(all_gradients, k=2)

        # Check results
        self.assertEqual(len(cluster_labels), n_total)
        self.assertTrue(0 <= byzantine_cluster_id < 2)

        # Filter Byzantine gradients
        honest_gradients_filtered, honest_indices = filter_byzantine_gradients(
            all_gradients, cluster_labels, byzantine_cluster_id
        )

        # Should detect some Byzantine operators
        byzantine_indices = [i for i in range(n_total) if i not in honest_indices]
        self.assertGreater(len(byzantine_indices), 0)

    def test_clustering_edge_cases(self):
        """Test clustering edge cases."""
        # Too few gradients
        with self.assertRaises(ValueError):
            spectral_clustering_byzantine_detection([np.random.randn(5)], k=2)

        # All identical gradients
        identical_gradient = np.ones(self.gradient_dim)
        identical_gradients = [identical_gradient.copy() for _ in range(5)]

        cluster_labels, byzantine_cluster_id = spectral_clustering_byzantine_detection(identical_gradients, k=2)
        self.assertEqual(len(cluster_labels), 5)

    def test_byzantine_cluster_identification(self):
        """Test Byzantine cluster identification heuristics."""
        # Create similarity matrix for known scenario
        # 2 honest operators (similar), 1 Byzantine (different)
        S = np.array([
            [1.0, 0.9, -0.8],
            [0.9, 1.0, -0.7],
            [-0.8, -0.7, 1.0]
        ])

        cluster_labels = np.array([0, 0, 1])  # First two in cluster 0, last in cluster 1
        byzantine_cluster_id = identify_byzantine_cluster(S, cluster_labels, k=2)

        # Cluster 1 (single operator with negative similarity) should be identified as Byzantine
        self.assertEqual(byzantine_cluster_id, 1)


class TestReputationSystem(unittest.TestCase):
    """Test reputation system."""

    def setUp(self):
        """Set up test data."""
        self.gradient_dim = 6
        self.operator_ids = [f"Op_{i}" for i in range(5)]

    def test_quality_score_computation(self):
        """Test quality score computation."""
        # Create test gradients
        gradient = np.array([1.0, 0.5, -0.2])
        aggregated_gradient = np.array([0.9, 0.4, -0.1])
        honest_gradients = [
            np.array([1.1, 0.6, -0.3]),
            np.array([0.8, 0.3, 0.0])
        ]

        quality_score = compute_quality_score(gradient, aggregated_gradient, honest_gradients)

        # Score should be in [0, 1]
        self.assertTrue(0 <= quality_score <= 1)

        # Gradient identical to aggregated should have high quality
        perfect_quality = compute_quality_score(aggregated_gradient, aggregated_gradient, honest_gradients)
        self.assertAlmostEqual(perfect_quality, 1.0, places=2)

    def test_reputation_updates(self):
        """Test reputation update mechanism."""
        gradients = [np.random.randn(self.gradient_dim) for _ in range(5)]
        aggregated_gradient = np.mean(gradients[:3], axis=0)  # Average of first 3

        # Initial reputations
        reputations = {op_id: 0.5 for op_id in self.operator_ids}

        # Update reputations
        honest_indices = [0, 1, 2]
        byzantine_indices = [3, 4]

        updated_reputations = update_reputations(
            self.operator_ids, gradients, aggregated_gradient,
            honest_indices, byzantine_indices, reputations,
            lambda_param=0.1, penalty=0.2
        )

        # Check that Byzantine operators are penalized
        for i in byzantine_indices:
            op_id = self.operator_ids[i]
            self.assertLess(updated_reputations[op_id], reputations[op_id])

        # Check that honest operators have updated reputations
        for i in honest_indices:
            op_id = self.operator_ids[i]
            # Reputation may increase or decrease slightly based on quality
            self.assertTrue(0 <= updated_reputations[op_id] <= 1)

    def test_reputation_weights(self):
        """Test reputation weight computation."""
        reputations = {
            "Op_0": 0.8,
            "Op_1": 0.6,
            "Op_2": 0.4,
            "Op_3": 0.0  # Fully penalized
        }

        operator_ids = list(reputations.keys())
        weights = compute_reputation_weights(operator_ids, reputations)

        # Weights should sum to 1
        self.assertAlmostEqual(np.sum(weights), 1.0)

        # Higher reputation should get higher weight
        self.assertGreater(weights[0], weights[1])  # Op_0 > Op_1
        self.assertGreater(weights[1], weights[2])  # Op_1 > Op_2

        # Zero reputation should get minimal weight
        self.assertAlmostEqual(weights[3], 0.0, places=3)

    def test_operator_selection(self):
        """Test operator selection by reputation."""
        reputations = {
            "Op_0": 0.9,
            "Op_1": 0.7,
            "Op_2": 0.5,
            "Op_3": 0.3,
            "Op_4": 0.1
        }

        # Select top 3 operators
        selected = select_operators_by_reputation(list(reputations.keys()), reputations, k=3)

        expected = ["Op_0", "Op_1", "Op_2"]
        self.assertEqual(selected, expected)

    def test_reputation_tracker(self):
        """Test ReputationTracker class."""
        tracker = ReputationTracker(self.operator_ids, initial_reputation=0.6)

        # Check initialization
        self.assertEqual(len(tracker.reputations), len(self.operator_ids))
        self.assertTrue(all(rep == 0.6 for rep in tracker.reputations.values()))

        # Simulate update
        gradients = [np.random.randn(self.gradient_dim) for _ in range(5)]
        aggregated_gradient = np.mean(gradients[:3], axis=0)

        tracker.update(gradients, aggregated_gradient, [0, 1, 2], [3, 4])

        # Check history is recorded
        self.assertEqual(len(tracker.reputation_history), 2)  # Initial + 1 update
        self.assertEqual(len(tracker.byzantine_detections), 1)

        # Check statistics
        stats = tracker.get_statistics()
        self.assertIn('mean', stats)
        self.assertIn('std', stats)


class TestTrustChainAggregation(unittest.TestCase):
    """Test TrustChain aggregation algorithm."""

    def setUp(self):
        """Set up test parameters."""
        self.gradient_dim = 6
        self.operator_ids = [f"Op_{i}" for i in range(4)]
        self.comm_params = setup_pedersen_commitment(512)

    def test_robust_mean_aggregation(self):
        """Test robust mean aggregation."""
        gradients = [
            np.array([1.0, 2.0]),
            np.array([1.2, 1.8]),
            np.array([0.8, 2.2])
        ]

        # Test uniform weights
        aggregated = robust_mean_aggregation(gradients)
        expected = np.mean(gradients, axis=0)
        np.testing.assert_array_almost_equal(aggregated, expected)

        # Test custom weights
        weights = np.array([0.5, 0.3, 0.2])
        aggregated_weighted = robust_mean_aggregation(gradients, weights)

        expected_weighted = np.zeros(2)
        for i, grad in enumerate(gradients):
            expected_weighted += weights[i] * grad

        np.testing.assert_array_almost_equal(aggregated_weighted, expected_weighted)

    def test_trustchain_aggregation_simple(self):
        """Test TrustChain aggregation with valid commitments."""
        # Create test gradients
        gradients = [np.random.randn(self.gradient_dim) * 0.1 for _ in range(4)]

        # Create commitments
        commitments = []
        openings = []
        for gradient in gradients:
            commitment, opening = commit_gradient(gradient, *self.comm_params)
            commitments.append(commitment)
            openings.append(opening)

        # Initial reputations
        reputations = {op_id: 0.5 for op_id in self.operator_ids}

        # DP parameters
        dp_params = {'sigma': 0.01, 'epsilon': 0.1}

        # Run aggregation
        result = trustchain_aggregation(
            gradients, commitments, openings, reputations,
            self.operator_ids, self.comm_params, dp_params
        )

        # Check result structure
        self.assertIn('aggregated_gradient', result)
        self.assertIn('updated_reputations', result)
        self.assertIn('honest_indices', result)
        self.assertIn('byzantine_indices', result)

        # Check gradient shape
        self.assertEqual(result['aggregated_gradient'].shape, (self.gradient_dim,))

        # All should be verified (no invalid commitments)
        self.assertEqual(result['n_verified'], len(gradients))

    def test_trustchain_aggregator(self):
        """Test TrustChain aggregator class."""
        aggregator = TrustChainAggregator(
            self.operator_ids, self.comm_params,
            lambda_param=0.1, penalty=0.2, base_sigma=0.05
        )

        # Check initialization
        self.assertEqual(len(aggregator.reputations), len(self.operator_ids))

        # Test aggregation
        gradients = [np.random.randn(self.gradient_dim) * 0.1 for _ in range(4)]
        commitments = []
        openings = []
        for gradient in gradients:
            commitment, opening = commit_gradient(gradient, *self.comm_params)
            commitments.append(commitment)
            openings.append(opening)

        result = aggregator.aggregate(gradients, commitments, openings, epsilon=0.1)

        # Check result
        self.assertIn('aggregated_gradient', result)
        self.assertEqual(len(aggregator.aggregation_history), 1)

    def test_aggregation_with_invalid_commitment(self):
        """Test aggregation with invalid commitment."""
        gradients = [np.random.randn(self.gradient_dim) * 0.1 for _ in range(3)]

        # Create valid commitments for first two gradients
        commitments = []
        openings = []
        for i, gradient in enumerate(gradients):
            if i < 2:
                commitment, opening = commit_gradient(gradient, *self.comm_params)
            else:
                # Create invalid commitment for third gradient
                commitment, opening = commit_gradient(gradients[0], *self.comm_params)  # Wrong gradient

            commitments.append(commitment)
            openings.append(opening)

        reputations = {op_id: 0.5 for op_id in self.operator_ids[:3]}
        dp_params = {'sigma': 0.01, 'epsilon': 0.1}

        result = trustchain_aggregation(
            gradients, commitments, openings, reputations,
            self.operator_ids[:3], self.comm_params, dp_params
        )

        # Third operator should be penalized
        self.assertLess(result['updated_reputations']['Op_2'], 0.5)
        self.assertEqual(result['n_verified'], 2)


class TestAggregationIntegration(unittest.TestCase):
    """Integration tests for aggregation components."""

    def test_end_to_end_aggregation(self):
        """Test complete aggregation pipeline."""
        # Setup
        gradient_dim = 8
        n_honest = 3
        n_byzantine = 2
        n_total = n_honest + n_byzantine

        # Create gradients (honest vs Byzantine)
        honest_base = np.random.randn(gradient_dim) * 0.1
        gradients = []

        # Honest gradients (similar)
        for _ in range(n_honest):
            gradient = honest_base + np.random.randn(gradient_dim) * 0.02
            gradients.append(gradient)

        # Byzantine gradients (sign flip)
        for _ in range(n_byzantine):
            gradient = -honest_base + np.random.randn(gradient_dim) * 0.02
            gradients.append(gradient)

        # Test spectral clustering
        cluster_labels, byzantine_cluster_id = spectral_clustering_byzantine_detection(gradients, k=2)
        honest_gradients, honest_indices = filter_byzantine_gradients(gradients, cluster_labels, byzantine_cluster_id)

        # Should identify most Byzantine operators
        byzantine_indices = [i for i in range(n_total) if i not in honest_indices]
        detection_rate = len([i for i in byzantine_indices if i >= n_honest]) / n_byzantine

        # Should detect at least some Byzantine operators
        self.assertGreater(detection_rate, 0.0)

        # Test reputation update
        operator_ids = [f"Op_{i}" for i in range(n_total)]
        reputations = {op_id: 0.5 for op_id in operator_ids}

        aggregated_gradient = np.mean(honest_gradients, axis=0)

        updated_reputations = update_reputations(
            operator_ids, gradients, aggregated_gradient,
            honest_indices, byzantine_indices, reputations
        )

        # Byzantine operators should be penalized
        for i in byzantine_indices:
            op_id = f"Op_{i}"
            self.assertLess(updated_reputations[op_id], reputations[op_id])


def run_aggregation_tests():
    """Run all aggregation tests."""
    print("🧪 Running FORTRESS-FL Aggregation Tests")
    print("="*50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSpectralClustering))
    suite.addTests(loader.loadTestsFromTestCase(TestReputationSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestTrustChainAggregation))
    suite.addTests(loader.loadTestsFromTestCase(TestAggregationIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print(f"\n✅ All aggregation tests passed!")
    else:
        print(f"\n❌ Some tests failed!")

    return success


if __name__ == "__main__":
    run_aggregation_tests()