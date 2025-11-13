"""
Pedersen Commitment Scheme for FORTRESS-FL

Implementation of Pedersen commitments to prevent adaptive attacks where Byzantine
operators observe honest gradients before choosing their poisoned gradients.
"""

import hashlib
import random
import numpy as np
from Crypto.Util import number
from typing import Tuple


def setup_pedersen_commitment(security_param: int = 2048) -> Tuple[int, int, int, int]:
    """
    Generate parameters for Pedersen commitment scheme.

    Args:
        security_param: Bit length of prime p (default: 2048 for 128-bit security)

    Returns:
        (p, q, g, h): Commitment parameters where:
            - p = 2q + 1: Safe prime
            - q: Sophie Germain prime
            - g, h: Independent generators of order q
    """
    print(f"[Pedersen] Setting up commitment scheme (security={security_param} bits)...")

    # Generate safe prime p = 2q + 1 where q is also prime
    while True:
        q = number.getPrime(security_param // 2)
        p = 2 * q + 1
        if number.isPrime(p):
            break

    print(f"[Pedersen] Generated safe prime p = 2q + 1 ({security_param} bits)")

    # Generate two independent generators g, h of order q
    g = find_generator(p, q)
    h = find_generator(p, q, exclude=[g])

    # Verify g ≠ h and both have order q
    assert pow(g, q, p) == 1 and g != 1, "Invalid generator g"
    assert pow(h, q, p) == 1 and h != 1, "Invalid generator h"
    assert g != h, "Generators must be different"

    print(f"[Pedersen] Generated independent generators g and h")

    return (p, q, g, h)


def find_generator(p: int, q: int, exclude: list = None) -> int:
    """
    Find a generator of order q in the multiplicative group Z_p*.

    Args:
        p: Prime modulus (p = 2q + 1)
        q: Order of desired generator
        exclude: List of values to exclude

    Returns:
        generator: Element of order q
    """
    exclude = exclude or []

    while True:
        # Random element in Z_p*
        candidate = random.randint(2, p - 1)

        if candidate in exclude:
            continue

        # Check if candidate^2 has order q (since p = 2q + 1)
        generator = pow(candidate, 2, p)

        # Verify it has order q
        if pow(generator, q, p) == 1 and generator != 1:
            return generator


def hash_to_integer(gradient: np.ndarray, modulus: int) -> int:
    """
    Hash gradient vector to integer in Z_q.

    Args:
        gradient: Gradient vector (numpy array)
        modulus: Prime modulus q

    Returns:
        hash_int: Integer in [0, modulus-1]
    """
    # Serialize gradient (convert to bytes)
    gradient_bytes = gradient.tobytes()

    # SHA-256 hash
    hash_digest = hashlib.sha256(gradient_bytes).digest()

    # Convert to integer and reduce modulo q
    hash_int = int.from_bytes(hash_digest, 'big') % modulus

    return hash_int


def commit_gradient(gradient: np.ndarray, p: int, q: int, g: int, h: int) -> Tuple[int, int]:
    """
    Commit to gradient using Pedersen commitment.

    Args:
        gradient: Gradient vector (numpy array of shape (d,))
        p, q, g, h: Commitment parameters from setup

    Returns:
        (commitment, randomness): Commitment C_i and opening randomness r_i
    """
    # Hash gradient to integer in Z_q
    message = hash_to_integer(gradient, q)

    # Sample random nonce from Z_q
    randomness = random.randint(1, q - 1)

    # Compute commitment: C_i = g^m_i * h^r_i mod p
    commitment = (pow(g, message, p) * pow(h, randomness, p)) % p

    return (commitment, randomness)


def verify_commitment(gradient: np.ndarray, commitment: int, randomness: int,
                     p: int, q: int, g: int, h: int) -> bool:
    """
    Verify that commitment opens to gradient with given randomness.

    Args:
        gradient: Claimed gradient vector
        commitment: Commitment value C_i
        randomness: Opening randomness r_i
        p, q, g, h: Commitment parameters

    Returns:
        is_valid: True if verification succeeds, False otherwise
    """
    # Hash gradient to integer
    message = hash_to_integer(gradient, q)

    # Recompute commitment
    commitment_check = (pow(g, message, p) * pow(h, randomness, p)) % p

    # Verify equality
    return commitment == commitment_check


def batch_commit_gradients(gradients: list, p: int, q: int, g: int, h: int) -> Tuple[list, list]:
    """
    Commit to multiple gradients in batch.

    Args:
        gradients: List of gradient vectors
        p, q, g, h: Commitment parameters

    Returns:
        (commitments, randomness_list): Lists of commitments and randomness values
    """
    commitments = []
    randomness_list = []

    for gradient in gradients:
        commitment, randomness = commit_gradient(gradient, p, q, g, h)
        commitments.append(commitment)
        randomness_list.append(randomness)

    return commitments, randomness_list


def batch_verify_commitments(gradients: list, commitments: list, randomness_list: list,
                            p: int, q: int, g: int, h: int) -> list:
    """
    Verify multiple commitments in batch.

    Args:
        gradients: List of gradient vectors
        commitments: List of commitment values
        randomness_list: List of opening randomness values
        p, q, g, h: Commitment parameters

    Returns:
        verification_results: List of boolean verification results
    """
    results = []

    for i, (gradient, commitment, randomness) in enumerate(zip(gradients, commitments, randomness_list)):
        is_valid = verify_commitment(gradient, commitment, randomness, p, q, g, h)
        results.append(is_valid)

    return results


# Example usage and testing
if __name__ == "__main__":
    print("Testing Pedersen Commitment Scheme...")

    # Setup commitment parameters
    p, q, g, h = setup_pedersen_commitment(1024)  # Smaller for testing

    # Create test gradient
    gradient = np.random.randn(10)
    print(f"Test gradient: {gradient[:3]}... (showing first 3 elements)")

    # Commit to gradient
    commitment, randomness = commit_gradient(gradient, p, q, g, h)
    print(f"Commitment: {commitment}")

    # Verify commitment
    is_valid = verify_commitment(gradient, commitment, randomness, p, q, g, h)
    print(f"Verification result: {is_valid}")

    # Test with wrong gradient (should fail)
    wrong_gradient = gradient + 0.1
    is_invalid = verify_commitment(wrong_gradient, commitment, randomness, p, q, g, h)
    print(f"Wrong gradient verification: {is_invalid}")

    print("Pedersen commitment test completed!")