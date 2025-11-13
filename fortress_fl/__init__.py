"""
FORTRESS-FL: Federated Operator Resilient Trustworthy Resource Efficient Secure Slice Learning

A multi-operator Byzantine-robust federated learning system for network orchestration.
"""

__version__ = "0.1.0"
__author__ = "FORTRESS-FL Team"

from .core.fortress_fl import FortressFL
from .core.training import train_fortress_fl

__all__ = ['FortressFL', 'train_fortress_fl']