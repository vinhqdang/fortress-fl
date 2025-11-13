"""
Multi-Party Computation (MPC) for FORTRESS-FL

Implementation of secret sharing and secure joint optimization
for cross-operator interference management.
"""

import random
import numpy as np
from typing import List, Tuple, Dict


def secret_share(value: int, n_shares: int = 3, threshold: int = None, modulus: int = 2**31 - 1) -> List[int]:
    """
    Shamir secret sharing: Split value into n shares with threshold reconstruction.

    Args:
        value: Scalar value to share (integer)
        n_shares: Number of shares (default: 3)
        threshold: Minimum shares needed for reconstruction (default: n_shares)
        modulus: Prime modulus for finite field (default: Mersenne prime 2^31-1)

    Returns:
        shares: List of n shares
    """
    if threshold is None:
        threshold = n_shares

    # Random coefficients for polynomial of degree (threshold-1)
    coeffs = [value] + [random.randint(0, modulus-1) for _ in range(threshold-1)]

    # Evaluate polynomial at points 1, 2, ..., n
    shares = []
    for x in range(1, n_shares + 1):
        share = sum(c * (x ** i) for i, c in enumerate(coeffs)) % modulus
        shares.append(share)

    return shares


def reconstruct_secret(shares: List[int], share_indices: List[int] = None,
                      modulus: int = 2**31 - 1) -> int:
    """
    Reconstruct secret from shares using Lagrange interpolation.

    Args:
        shares: List of shares (must have at least threshold shares)
        share_indices: Indices of shares (default: [1, 2, ..., len(shares)])
        modulus: Prime modulus

    Returns:
        secret: Reconstructed value
    """
    n = len(shares)
    if share_indices is None:
        share_indices = list(range(1, n + 1))

    secret = 0

    for i in range(n):
        # Lagrange basis polynomial
        numerator = 1
        denominator = 1
        for j in range(n):
            if i != j:
                numerator = (numerator * (0 - share_indices[j])) % modulus
                denominator = (denominator * (share_indices[i] - share_indices[j])) % modulus

        # Modular inverse of denominator
        denominator_inv = pow(denominator, modulus - 2, modulus)
        lagrange_coeff = (numerator * denominator_inv) % modulus
        secret = (secret + shares[i] * lagrange_coeff) % modulus

    return secret


def share_matrix(matrix: np.ndarray, n_shares: int = 3, modulus: int = 2**31 - 1) -> List[np.ndarray]:
    """
    Share each element of a matrix using secret sharing.

    Args:
        matrix: Input matrix to share
        n_shares: Number of shares per element
        modulus: Prime modulus

    Returns:
        matrix_shares: List of n_shares matrices, each containing one share per element
    """
    rows, cols = matrix.shape
    matrix_shares = [np.zeros((rows, cols), dtype=np.int64) for _ in range(n_shares)]

    for i in range(rows):
        for j in range(cols):
            # Convert to integer (scale by 1000 to preserve decimals)
            value_int = int(matrix[i, j] * 1000) % modulus
            shares = secret_share(value_int, n_shares, modulus=modulus)

            for k in range(n_shares):
                matrix_shares[k][i, j] = shares[k]

    return matrix_shares


def reconstruct_matrix(matrix_shares: List[np.ndarray], modulus: int = 2**31 - 1) -> np.ndarray:
    """
    Reconstruct matrix from shares.

    Args:
        matrix_shares: List of share matrices
        modulus: Prime modulus

    Returns:
        reconstructed_matrix: Original matrix (scaled back)
    """
    rows, cols = matrix_shares[0].shape
    reconstructed = np.zeros((rows, cols), dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            shares = [matrix_shares[k][i, j] for k in range(len(matrix_shares))]
            value_int = reconstruct_secret(shares, modulus=modulus)
            # Scale back from integer
            reconstructed[i, j] = (value_int / 1000.0)

    return reconstructed


def secure_joint_optimization_mpc(operator_A_data: Dict, operator_B_data: Dict,
                                 n_iterations: int = 10) -> Tuple[float, float]:
    """
    Secure multi-party computation for joint interference optimization.

    Args:
        operator_A_data: Dict {'interference_matrix': I_A, 'power_range': (Pmin, Pmax)}
        operator_B_data: Dict {'interference_matrix': I_B, 'power_range': (Pmin, Pmax)}
        n_iterations: Number of optimization iterations

    Returns:
        (power_A_opt, power_B_opt): Optimal power allocations
    """
    print("[MPC] Starting secure joint optimization...")

    # Extract data
    I_A = operator_A_data['interference_matrix']
    I_B = operator_B_data['interference_matrix']
    power_range_A = operator_A_data['power_range']
    power_range_B = operator_B_data['power_range']

    # ===== SECRET SHARING PHASE =====
    print("[MPC] Phase 1: Secret sharing interference matrices...")

    # Share interference matrices (3-party computation)
    I_A_shares = share_matrix(I_A, n_shares=3)
    I_B_shares = share_matrix(I_B, n_shares=3)

    # Each party gets one share (in real system, sent over network)
    # Party 0: Coordinator, Party 1: Operator A, Party 2: Operator B

    # ===== SECURE COMPUTATION PHASE =====
    print("[MPC] Phase 2: Secure gradient descent...")

    # Initialize power allocations
    power_A = (power_range_A[0] + power_range_A[1]) / 2
    power_B = (power_range_B[0] + power_range_B[1]) / 2

    # Learning rate
    learning_rate = 0.01

    for iter_idx in range(n_iterations):
        # In real MPC, this computation would be done in shares
        # For demonstration, we use cleartext but show the MPC structure

        # Compute gradients in MPC circuit
        # Joint interference: J = ||I_A * power_A + I_B * power_B||^2
        interference = I_A.flatten() * power_A + I_B.flatten() * power_B
        joint_interference = np.sum(interference ** 2)

        # Gradients
        grad_A = 2 * np.sum(interference * I_A.flatten())
        grad_B = 2 * np.sum(interference * I_B.flatten())

        # Update power allocations
        power_A -= learning_rate * grad_A
        power_B -= learning_rate * grad_B

        # Project to feasible range
        power_A = np.clip(power_A, power_range_A[0], power_range_A[1])
        power_B = np.clip(power_B, power_range_B[0], power_range_B[1])

        if iter_idx % 5 == 0:
            print(f"[MPC] Iteration {iter_idx}: J={joint_interference:.4f}, "
                  f"P_A={power_A:.3f}, P_B={power_B:.3f}")

    # ===== REVEAL PHASE =====
    print("[MPC] Phase 3: Revealing optimal power allocations...")

    return power_A, power_B


def compute_interference_gradient(I_A: np.ndarray, I_B: np.ndarray,
                                power_A: float, power_B: float, wrt: str = 'A') -> float:
    """
    Compute gradient of joint interference objective.

    Args:
        I_A, I_B: Interference matrices
        power_A, power_B: Power allocations
        wrt: Compute gradient with respect to 'A' or 'B'

    Returns:
        gradient: Scalar gradient value
    """
    # Joint interference: J = ||I_A * power_A + I_B * power_B||^2
    interference = I_A.flatten() * power_A + I_B.flatten() * power_B

    if wrt == 'A':
        gradient = 2 * np.sum(interference * I_A.flatten())
    else:
        gradient = 2 * np.sum(interference * I_B.flatten())

    return gradient


class MPCProtocol:
    """
    Simple MPC protocol for demonstration.
    In practice, would use libraries like PySyft, MP-SPDZ, or CrypTen.
    """

    def __init__(self, n_parties: int = 3, threshold: int = 2, modulus: int = 2**31 - 1):
        self.n_parties = n_parties
        self.threshold = threshold
        self.modulus = modulus
        self.shares = {}

    def share_value(self, value: float, value_id: str) -> List[int]:
        """Share a value among parties."""
        value_int = int(value * 1000) % self.modulus
        shares = secret_share(value_int, self.n_parties, modulus=self.modulus)
        self.shares[value_id] = shares
        return shares

    def add_shares(self, value_id1: str, value_id2: str, result_id: str) -> None:
        """Add two shared values."""
        shares1 = self.shares[value_id1]
        shares2 = self.shares[value_id2]
        result_shares = [(s1 + s2) % self.modulus for s1, s2 in zip(shares1, shares2)]
        self.shares[result_id] = result_shares

    def multiply_shares(self, value_id1: str, value_id2: str, result_id: str) -> None:
        """Multiply two shared values (simplified - requires more complex protocol)."""
        # This is a simplified version - real MPC multiplication is more complex
        shares1 = self.shares[value_id1]
        shares2 = self.shares[value_id2]

        # For demonstration only - not secure
        val1 = reconstruct_secret(shares1, modulus=self.modulus)
        val2 = reconstruct_secret(shares2, modulus=self.modulus)
        product = (val1 * val2) % self.modulus

        self.shares[result_id] = secret_share(product, self.n_parties, modulus=self.modulus)

    def reveal_value(self, value_id: str) -> float:
        """Reveal a shared value."""
        shares = self.shares[value_id]
        value_int = reconstruct_secret(shares, modulus=self.modulus)
        return value_int / 1000.0

    def secure_joint_optimization_mpc(self, operator1_data: Dict, operator2_data: Dict,
                                     model_dim: int, learning_rate: float = 0.01,
                                     n_iterations: int = 5) -> Dict:
        """
        Perform secure joint optimization between two operators using MPC.

        Args:
            operator1_data: First operator's dataset {'X': features, 'y': labels}
            operator2_data: Second operator's dataset {'X': features, 'y': labels}
            model_dim: Model dimension
            learning_rate: Learning rate for optimization
            n_iterations: Number of optimization iterations

        Returns:
            Dictionary with optimization results

        Note:
            This is a simplified demonstration. Full MPC implementation would
            require specialized libraries and more complex protocols.
        """
        # Initialize model
        model = np.random.randn(model_dim) * 0.1

        # Extract data
        X1, y1 = operator1_data['X'], operator1_data['y']
        X2, y2 = operator2_data['X'], operator2_data['y']

        convergence_history = []

        for iteration in range(n_iterations):
            # Compute local gradients (in practice, would be done in shares)
            grad1 = compute_gradient(model, {'X': X1, 'y': y1})
            grad2 = compute_gradient(model, {'X': X2, 'y': y2})

            # Simulate MPC aggregation (simplified)
            # In reality, gradients would be secret-shared and aggregated securely
            joint_gradient = (grad1 + grad2) / 2

            # Update model
            model -= learning_rate * joint_gradient

            # Track convergence
            loss1 = np.mean((X1 @ model - y1) ** 2)
            loss2 = np.mean((X2 @ model - y2) ** 2)
            joint_loss = (loss1 + loss2) / 2
            convergence_history.append(joint_loss)

        # Check convergence
        converged = False
        if len(convergence_history) > 1:
            recent_improvement = abs(convergence_history[-2] - convergence_history[-1])
            converged = recent_improvement < 0.001

        return {
            'final_model': model,
            'convergence_history': convergence_history,
            'final_loss': convergence_history[-1] if convergence_history else float('inf'),
            'converged': converged,
            'n_iterations': n_iterations
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing MPC Protocol...")

    # Test secret sharing
    secret = 12345
    shares = secret_share(secret, n_shares=5, threshold=3)
    print(f"Secret: {secret}")
    print(f"Shares: {shares[:3]}... (showing first 3)")

    # Reconstruct with threshold shares
    reconstructed = reconstruct_secret(shares[:3])
    print(f"Reconstructed: {reconstructed}")
    assert reconstructed == secret, "Reconstruction failed!"

    # Test matrix sharing
    matrix = np.random.randn(3, 3)
    matrix_shares = share_matrix(matrix, n_shares=3)
    reconstructed_matrix = reconstruct_matrix(matrix_shares)

    print(f"Matrix sharing error: {np.mean(np.abs(matrix - reconstructed_matrix))}")

    # Test MPC protocol
    mpc = MPCProtocol()
    mpc.share_value(10.5, "x")
    mpc.share_value(20.3, "y")
    mpc.add_shares("x", "y", "sum")
    result = mpc.reveal_value("sum")
    print(f"MPC addition: 10.5 + 20.3 = {result}")

    print("MPC test completed!")