#!/usr/bin/env python3
"""
Tests for FORTRESS-FL Cryptographic Components

Test suite for Pedersen commitments and MPC functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from fortress_fl.crypto.pedersen import (
    setup_pedersen_commitment, commit_gradient, verify_commitment,
    hash_to_integer, batch_commit_gradients, batch_verify_commitments
)
from fortress_fl.crypto.mpc import (
    secret_share, reconstruct_secret, share_matrix, reconstruct_matrix,
    MPCProtocol
)


class TestPedersenCommitments(unittest.TestCase):
    """Test Pedersen commitment scheme."""

    def setUp(self):
        """Set up test parameters."""
        self.security_param = 512  # Small for testing
        self.gradient_dim = 10
        self.p, self.q, self.g, self.h = setup_pedersen_commitment(self.security_param)

    def test_setup_pedersen_commitment(self):
        """Test Pedersen commitment setup."""
        # Verify that p = 2q + 1
        self.assertEqual(self.p, 2 * self.q + 1)

        # Verify generators have order q
        self.assertEqual(pow(self.g, self.q, self.p), 1)
        self.assertEqual(pow(self.h, self.q, self.p), 1)

        # Verify generators are different
        self.assertNotEqual(self.g, self.h)

    def test_hash_to_integer(self):
        """Test gradient hashing."""
        gradient = np.random.randn(self.gradient_dim)

        # Test deterministic hashing
        hash1 = hash_to_integer(gradient, self.q)
        hash2 = hash_to_integer(gradient, self.q)
        self.assertEqual(hash1, hash2)

        # Test range
        self.assertTrue(0 <= hash1 < self.q)

        # Test different gradients produce different hashes
        gradient2 = gradient + 0.001
        hash3 = hash_to_integer(gradient2, self.q)
        self.assertNotEqual(hash1, hash3)

    def test_commit_and_verify(self):
        """Test commitment and verification."""
        gradient = np.random.randn(self.gradient_dim)

        # Commit to gradient
        commitment, randomness = commit_gradient(gradient, self.p, self.q, self.g, self.h)

        # Verify commitment
        is_valid = verify_commitment(gradient, commitment, randomness,
                                   self.p, self.q, self.g, self.h)
        self.assertTrue(is_valid)

        # Test with wrong gradient (should fail)
        wrong_gradient = gradient + 0.1
        is_invalid = verify_commitment(wrong_gradient, commitment, randomness,
                                     self.p, self.q, self.g, self.h)
        self.assertFalse(is_invalid)

    def test_commitment_hiding(self):
        """Test commitment hiding property."""
        gradient1 = np.random.randn(self.gradient_dim)
        gradient2 = np.random.randn(self.gradient_dim)

        # Commitments to different gradients should be different
        commitment1, _ = commit_gradient(gradient1, self.p, self.q, self.g, self.h)
        commitment2, _ = commit_gradient(gradient2, self.p, self.q, self.g, self.h)

        # With high probability, commitments are different
        self.assertNotEqual(commitment1, commitment2)

        # Multiple commitments to same gradient should be different (due to randomness)
        commitment1a, _ = commit_gradient(gradient1, self.p, self.q, self.g, self.h)
        commitment1b, _ = commit_gradient(gradient1, self.p, self.q, self.g, self.h)
        self.assertNotEqual(commitment1a, commitment1b)

    def test_batch_operations(self):
        """Test batch commitment operations."""
        gradients = [np.random.randn(self.gradient_dim) for _ in range(5)]

        # Batch commit
        commitments, randomness_list = batch_commit_gradients(
            gradients, self.p, self.q, self.g, self.h
        )

        # Batch verify
        results = batch_verify_commitments(
            gradients, commitments, randomness_list,
            self.p, self.q, self.g, self.h
        )

        # All should verify successfully
        self.assertTrue(all(results))

        # Test with one corrupted gradient
        corrupted_gradients = gradients.copy()
        corrupted_gradients[2] += 0.1  # Corrupt one gradient

        corrupted_results = batch_verify_commitments(
            corrupted_gradients, commitments, randomness_list,
            self.p, self.q, self.g, self.h
        )

        # Only the corrupted one should fail
        expected_results = [True, True, False, True, True]
        self.assertEqual(corrupted_results, expected_results)


class TestMPC(unittest.TestCase):
    """Test Multi-Party Computation functionality."""

    def test_secret_sharing_basic(self):
        """Test basic secret sharing."""
        secret = 12345
        n_shares = 5
        threshold = 3
        modulus = 2**31 - 1

        # Share secret
        shares = secret_share(secret, n_shares, threshold, modulus)
        self.assertEqual(len(shares), n_shares)

        # Reconstruct with threshold shares
        reconstructed = reconstruct_secret(shares[:threshold], modulus=modulus)
        self.assertEqual(reconstructed, secret)

        # Reconstruct with more than threshold shares
        reconstructed_full = reconstruct_secret(shares, modulus=modulus)
        self.assertEqual(reconstructed_full, secret)

    def test_secret_sharing_insufficient_shares(self):
        """Test secret sharing with insufficient shares."""
        secret = 54321
        n_shares = 5
        threshold = 3
        modulus = 2**31 - 1

        shares = secret_share(secret, n_shares, threshold, modulus)

        # Try to reconstruct with insufficient shares
        reconstructed = reconstruct_secret(shares[:threshold-1], modulus=modulus)
        # Should not equal original secret (with high probability)
        self.assertNotEqual(reconstructed, secret)

    def test_matrix_sharing(self):
        """Test matrix secret sharing."""
        # Create test matrix
        matrix = np.random.randn(3, 4) * 10  # Scale up for integer conversion
        n_shares = 3
        modulus = 2**31 - 1

        # Share matrix
        matrix_shares = share_matrix(matrix, n_shares, modulus)
        self.assertEqual(len(matrix_shares), n_shares)

        # Reconstruct matrix
        reconstructed = reconstruct_matrix(matrix_shares, modulus)

        # Should be approximately equal (due to scaling)
        np.testing.assert_allclose(matrix, reconstructed, rtol=1e-2)

    def test_mpc_protocol(self):
        """Test MPC protocol operations."""
        protocol = MPCProtocol(n_parties=3)

        # Share values
        val1 = 10.5
        val2 = 20.3

        protocol.share_value(val1, "x")
        protocol.share_value(val2, "y")

        # Test addition
        protocol.add_shares("x", "y", "sum")
        result_sum = protocol.reveal_value("sum")

        # Should be approximately equal
        expected_sum = val1 + val2
        self.assertAlmostEqual(result_sum, expected_sum, places=2)

    def test_secure_joint_optimization(self):
        """Test secure joint optimization (simplified)."""
        from fortress_fl.crypto.mpc import secure_joint_optimization_mpc

        # Create test interference matrices
        I_A = np.random.rand(3, 3) * 0.5
        I_B = np.random.rand(3, 3) * 0.5

        operator_A_data = {
            'interference_matrix': I_A,
            'power_range': (0.1, 1.0)
        }

        operator_B_data = {
            'interference_matrix': I_B,
            'power_range': (0.1, 1.0)
        }

        # Run optimization
        power_A, power_B = secure_joint_optimization_mpc(
            operator_A_data, operator_B_data, n_iterations=5
        )

        # Check results are in valid range
        self.assertTrue(operator_A_data['power_range'][0] <= power_A <= operator_A_data['power_range'][1])
        self.assertTrue(operator_B_data['power_range'][0] <= power_B <= operator_B_data['power_range'][1])

    def test_large_numbers(self):
        """Test with large numbers near modulus."""
        modulus = 2**31 - 1
        secret = modulus - 100  # Large number

        shares = secret_share(secret, 5, 3, modulus)
        reconstructed = reconstruct_secret(shares[:3], modulus=modulus)

        self.assertEqual(reconstructed, secret)


class TestCryptoIntegration(unittest.TestCase):
    """Integration tests for cryptographic components."""

    def setUp(self):
        """Set up test environment."""
        self.p, self.q, self.g, self.h = setup_pedersen_commitment(512)
        self.gradient_dim = 8

    def test_commitment_in_aggregation_workflow(self):
        """Test commitment scheme in aggregation context."""
        # Simulate multiple operators
        n_operators = 4
        gradients = [np.random.randn(self.gradient_dim) for _ in range(n_operators)]

        # Phase 1: All operators commit
        commitments = []
        openings = []

        for gradient in gradients:
            commitment, opening = commit_gradient(gradient, self.p, self.q, self.g, self.h)
            commitments.append(commitment)
            openings.append(opening)

        # Phase 2: Verify all commitments
        verification_results = []
        for i, gradient in enumerate(gradients):
            is_valid = verify_commitment(gradient, commitments[i], openings[i],
                                       self.p, self.q, self.g, self.h)
            verification_results.append(is_valid)

        # All should verify
        self.assertTrue(all(verification_results))

        # Simulate one operator cheating
        cheating_gradient = gradients[2] + 0.1  # Operator 2 cheats
        cheating_result = verify_commitment(cheating_gradient, commitments[2], openings[2],
                                          self.p, self.q, self.g, self.h)
        self.assertFalse(cheating_result)

    def test_crypto_performance(self):
        """Test cryptographic performance with realistic parameters."""
        import time

        # Test with moderate number of operators
        n_operators = 10
        gradients = [np.random.randn(self.gradient_dim) for _ in range(n_operators)]

        # Time commitment phase
        start_time = time.time()
        commitments, openings = batch_commit_gradients(gradients, self.p, self.q, self.g, self.h)
        commit_time = time.time() - start_time

        # Time verification phase
        start_time = time.time()
        results = batch_verify_commitments(gradients, commitments, openings,
                                         self.p, self.q, self.g, self.h)
        verify_time = time.time() - start_time

        # Performance should be reasonable (< 1 second for small test)
        self.assertLess(commit_time, 1.0)
        self.assertLess(verify_time, 1.0)

        # All should verify
        self.assertTrue(all(results))


def run_crypto_tests():
    """Run all cryptographic tests."""
    print("🧪 Running FORTRESS-FL Cryptographic Tests")
    print("="*50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPedersenCommitments))
    suite.addTests(loader.loadTestsFromTestCase(TestMPC))
    suite.addTests(loader.loadTestsFromTestCase(TestCryptoIntegration))

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
            print(f"  - {test}: {traceback}")

    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print(f"\n✅ All cryptographic tests passed!")
    else:
        print(f"\n❌ Some tests failed!")

    return success


if __name__ == "__main__":
    run_crypto_tests()