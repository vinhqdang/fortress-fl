"""Cryptographic components for FORTRESS-FL."""

from .pedersen import (
    setup_pedersen_commitment,
    commit_gradient,
    verify_commitment,
    hash_to_integer
)

from .mpc import (
    secret_share,
    reconstruct_secret,
    secure_joint_optimization_mpc
)

__all__ = [
    'setup_pedersen_commitment',
    'commit_gradient',
    'verify_commitment',
    'hash_to_integer',
    'secret_share',
    'reconstruct_secret',
    'secure_joint_optimization_mpc'
]