"""
Byzantine Attack Generators for FORTRESS-FL Testing

Collection of different Byzantine attack strategies for evaluating system robustness.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class ByzantineAttack(ABC):
    """Base class for Byzantine attacks."""

    def __init__(self, attack_strength: float = 1.0):
        """
        Initialize Byzantine attack.

        Args:
            attack_strength: Multiplier for attack magnitude
        """
        self.attack_strength = attack_strength
        self.round_number = 0
        self.attack_history = []

    @abstractmethod
    def generate_gradient(self, model_dim: int, current_model: np.ndarray = None,
                         honest_gradients: List[np.ndarray] = None,
                         operator_id: str = None) -> np.ndarray:
        """
        Generate malicious gradient for current round.

        Args:
            model_dim: Dimension of gradient
            current_model: Current global model
            honest_gradients: List of honest gradients (if available)
            operator_id: ID of Byzantine operator

        Returns:
            byzantine_gradient: Malicious gradient vector
        """
        pass

    def update_round(self) -> None:
        """Update internal state for next round."""
        self.round_number += 1

    def get_attack_info(self) -> Dict:
        """Get information about the attack."""
        return {
            'attack_type': self.__class__.__name__,
            'attack_strength': self.attack_strength,
            'rounds_active': self.round_number
        }


class SignFlipAttack(ByzantineAttack):
    """Sign flip attack: reverse the direction of honest gradients."""

    def generate_gradient(self, model_dim: int, current_model: np.ndarray = None,
                         honest_gradients: List[np.ndarray] = None,
                         operator_id: str = None) -> np.ndarray:
        """Generate sign-flipped gradient."""
        if honest_gradients:
            # Use average of honest gradients and flip sign
            avg_honest = np.mean(honest_gradients, axis=0)
            byzantine_gradient = -avg_honest * self.attack_strength
        else:
            # Generate typical gradient direction and flip
            typical_gradient = -np.random.randn(model_dim) * 0.1  # Typical minimization direction
            byzantine_gradient = -typical_gradient * self.attack_strength

        self.attack_history.append(byzantine_gradient.copy())
        return byzantine_gradient


class RandomAttack(ByzantineAttack):
    """Random noise attack: send random gradients."""

    def __init__(self, attack_strength: float = 1.0, noise_std: float = 1.0):
        super().__init__(attack_strength)
        self.noise_std = noise_std

    def generate_gradient(self, model_dim: int, current_model: np.ndarray = None,
                         honest_gradients: List[np.ndarray] = None,
                         operator_id: str = None) -> np.ndarray:
        """Generate random noise gradient."""
        byzantine_gradient = np.random.normal(0, self.noise_std, model_dim) * self.attack_strength
        self.attack_history.append(byzantine_gradient.copy())
        return byzantine_gradient


class ModelPoisoningAttack(ByzantineAttack):
    """Model poisoning attack: target specific model parameters."""

    def __init__(self, attack_strength: float = 1.0, target_direction: np.ndarray = None):
        super().__init__(attack_strength)
        self.target_direction = target_direction

    def generate_gradient(self, model_dim: int, current_model: np.ndarray = None,
                         honest_gradients: List[np.ndarray] = None,
                         operator_id: str = None) -> np.ndarray:
        """Generate model poisoning gradient."""
        if self.target_direction is None or len(self.target_direction) != model_dim:
            # Generate random target direction
            self.target_direction = np.random.randn(model_dim)
            self.target_direction = self.target_direction / np.linalg.norm(self.target_direction)

        # Create gradient that pushes model toward target
        if current_model is not None:
            # Push away from current model toward target
            to_target = self.target_direction - current_model
            to_target = to_target / (np.linalg.norm(to_target) + 1e-8)
            byzantine_gradient = to_target * self.attack_strength
        else:
            byzantine_gradient = self.target_direction * self.attack_strength

        self.attack_history.append(byzantine_gradient.copy())
        return byzantine_gradient


class CoordinatedAttack(ByzantineAttack):
    """Coordinated attack: multiple Byzantine operators work together."""

    def __init__(self, attack_strength: float = 1.0, coordination_vector: np.ndarray = None):
        super().__init__(attack_strength)
        self.coordination_vector = coordination_vector
        self.byzantine_ids = set()

    def generate_gradient(self, model_dim: int, current_model: np.ndarray = None,
                         honest_gradients: List[np.ndarray] = None,
                         operator_id: str = None) -> np.ndarray:
        """Generate coordinated attack gradient."""
        if operator_id:
            self.byzantine_ids.add(operator_id)

        if self.coordination_vector is None or len(self.coordination_vector) != model_dim:
            # Create shared coordination vector
            self.coordination_vector = np.random.randn(model_dim)
            self.coordination_vector = self.coordination_vector / np.linalg.norm(self.coordination_vector)

        # All Byzantine operators send the same malicious direction
        byzantine_gradient = self.coordination_vector * self.attack_strength

        # Add small random perturbation to avoid perfect coordination detection
        noise = np.random.normal(0, 0.01, model_dim)
        byzantine_gradient += noise

        self.attack_history.append(byzantine_gradient.copy())
        return byzantine_gradient


class AdaptiveAttack(ByzantineAttack):
    """Adaptive attack: changes strategy based on detection history."""

    def __init__(self, attack_strength: float = 1.0):
        super().__init__(attack_strength)
        self.detection_history = []
        self.current_strategy = 'sign_flip'
        self.strategies = ['sign_flip', 'random', 'model_poisoning', 'stealthy']

    def set_detection_result(self, was_detected: bool) -> None:
        """Update attack based on detection result."""
        self.detection_history.append(was_detected)

        # Change strategy if detected multiple times recently
        recent_detections = sum(self.detection_history[-3:])  # Last 3 rounds
        if recent_detections >= 2:
            # Switch to more stealthy strategy
            if self.current_strategy != 'stealthy':
                self.current_strategy = 'stealthy'
                print(f"[AdaptiveAttack] Switching to stealthy mode after detections")

    def generate_gradient(self, model_dim: int, current_model: np.ndarray = None,
                         honest_gradients: List[np.ndarray] = None,
                         operator_id: str = None) -> np.ndarray:
        """Generate adaptive attack gradient."""
        if self.current_strategy == 'sign_flip':
            if honest_gradients:
                avg_honest = np.mean(honest_gradients, axis=0)
                byzantine_gradient = -avg_honest * self.attack_strength
            else:
                byzantine_gradient = -np.random.randn(model_dim) * 0.1 * self.attack_strength

        elif self.current_strategy == 'random':
            byzantine_gradient = np.random.randn(model_dim) * self.attack_strength

        elif self.current_strategy == 'model_poisoning':
            target = np.random.randn(model_dim)
            target = target / np.linalg.norm(target)
            byzantine_gradient = target * self.attack_strength

        elif self.current_strategy == 'stealthy':
            # Try to mimic honest gradients but with subtle bias
            if honest_gradients:
                avg_honest = np.mean(honest_gradients, axis=0)
                # Add small bias toward malicious direction
                malicious_bias = np.random.randn(model_dim) * 0.1
                byzantine_gradient = avg_honest + malicious_bias * self.attack_strength * 0.3
            else:
                # Very small random gradient to avoid detection
                byzantine_gradient = np.random.randn(model_dim) * 0.01 * self.attack_strength

        else:
            # Default to sign flip
            byzantine_gradient = -np.random.randn(model_dim) * 0.1 * self.attack_strength

        self.attack_history.append(byzantine_gradient.copy())
        return byzantine_gradient


class ZeroGradientAttack(ByzantineAttack):
    """Free-riding attack: send zero gradients."""

    def generate_gradient(self, model_dim: int, current_model: np.ndarray = None,
                         honest_gradients: List[np.ndarray] = None,
                         operator_id: str = None) -> np.ndarray:
        """Generate zero gradient (free-riding)."""
        byzantine_gradient = np.zeros(model_dim)
        self.attack_history.append(byzantine_gradient.copy())
        return byzantine_gradient


class LabelFlipAttack(ByzantineAttack):
    """Label flipping attack: corrupt local training data."""

    def __init__(self, attack_strength: float = 1.0, flip_probability: float = 0.5):
        super().__init__(attack_strength)
        self.flip_probability = flip_probability

    def corrupt_dataset(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Corrupt local dataset by flipping labels.

        Args:
            X: Feature matrix
            y: Original labels

        Returns:
            (X, corrupted_y): Features and corrupted labels
        """
        corrupted_y = y.copy()
        n_samples = len(y)

        # Randomly flip labels based on flip probability
        flip_mask = np.random.rand(n_samples) < self.flip_probability
        corrupted_y[flip_mask] = 1 - corrupted_y[flip_mask]  # Flip binary labels

        return X, corrupted_y

    def generate_gradient(self, model_dim: int, current_model: np.ndarray = None,
                         honest_gradients: List[np.ndarray] = None,
                         operator_id: str = None) -> np.ndarray:
        """
        Generate gradient from corrupted data.
        Note: This requires access to local dataset for realistic simulation.
        """
        # For demonstration, generate a gradient that would result from label flipping
        if honest_gradients:
            avg_honest = np.mean(honest_gradients, axis=0)
            # Label flipping typically causes gradients in opposite direction
            byzantine_gradient = -avg_honest * self.flip_probability * self.attack_strength
        else:
            byzantine_gradient = np.random.randn(model_dim) * 0.1 * self.attack_strength

        self.attack_history.append(byzantine_gradient.copy())
        return byzantine_gradient


class AttackFactory:
    """Factory for creating different types of Byzantine attacks."""

    @staticmethod
    def create_attack(attack_type: str, attack_strength: float = 1.0,
                     **kwargs) -> ByzantineAttack:
        """
        Create a Byzantine attack instance.

        Args:
            attack_type: Type of attack
            attack_strength: Attack strength multiplier
            **kwargs: Additional parameters for specific attacks

        Returns:
            attack: Byzantine attack instance
        """
        attack_type = attack_type.lower()

        if attack_type == 'sign_flip':
            return SignFlipAttack(attack_strength)

        elif attack_type == 'random':
            noise_std = kwargs.get('noise_std', 1.0)
            return RandomAttack(attack_strength, noise_std)

        elif attack_type == 'model_poisoning':
            target_direction = kwargs.get('target_direction', None)
            return ModelPoisoningAttack(attack_strength, target_direction)

        elif attack_type == 'coordinated':
            coordination_vector = kwargs.get('coordination_vector', None)
            return CoordinatedAttack(attack_strength, coordination_vector)

        elif attack_type == 'adaptive':
            return AdaptiveAttack(attack_strength)

        elif attack_type == 'zero':
            return ZeroGradientAttack(attack_strength)

        elif attack_type == 'label_flip':
            flip_probability = kwargs.get('flip_probability', 0.5)
            return LabelFlipAttack(attack_strength, flip_probability)

        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

    @staticmethod
    def get_available_attacks() -> List[str]:
        """Get list of available attack types."""
        return ['sign_flip', 'random', 'model_poisoning', 'coordinated',
                'adaptive', 'zero', 'label_flip']


class MultiAttackScenario:
    """Scenario with multiple types of Byzantine attacks."""

    def __init__(self, attack_configs: List[Dict]):
        """
        Initialize multi-attack scenario.

        Args:
            attack_configs: List of attack configurations, each with:
                - 'operator_ids': List of operator IDs using this attack
                - 'attack_type': Type of attack
                - 'attack_strength': Attack strength
                - Additional attack-specific parameters
        """
        self.attacks = {}
        self.operator_to_attack = {}

        for config in attack_configs:
            attack = AttackFactory.create_attack(
                config['attack_type'],
                config.get('attack_strength', 1.0),
                **{k: v for k, v in config.items() if k not in ['operator_ids', 'attack_type', 'attack_strength']}
            )

            attack_id = f"{config['attack_type']}_{len(self.attacks)}"
            self.attacks[attack_id] = attack

            for op_id in config['operator_ids']:
                self.operator_to_attack[op_id] = attack_id

    def generate_gradient(self, operator_id: str, model_dim: int,
                         current_model: np.ndarray = None,
                         honest_gradients: List[np.ndarray] = None) -> np.ndarray:
        """Generate gradient for specified operator."""
        if operator_id in self.operator_to_attack:
            attack_id = self.operator_to_attack[operator_id]
            attack = self.attacks[attack_id]
            return attack.generate_gradient(model_dim, current_model, honest_gradients, operator_id)
        else:
            raise ValueError(f"Operator {operator_id} not configured for any attack")

    def update_round(self) -> None:
        """Update all attacks for next round."""
        for attack in self.attacks.values():
            attack.update_round()

    def get_attack_summary(self) -> Dict:
        """Get summary of all active attacks."""
        summary = {}
        for attack_id, attack in self.attacks.items():
            summary[attack_id] = attack.get_attack_info()
        return summary


# Example usage and testing
if __name__ == "__main__":
    print("Testing Byzantine Attack Generators...")

    model_dim = 8
    n_rounds = 5

    # Test individual attacks
    attacks_to_test = [
        SignFlipAttack(1.0),
        RandomAttack(1.0, noise_std=0.5),
        ModelPoisoningAttack(1.0),
        CoordinatedAttack(1.0),
        AdaptiveAttack(1.0),
        ZeroGradientAttack(1.0)
    ]

    # Generate some honest gradients for context
    honest_gradients = [np.random.randn(model_dim) * 0.1 for _ in range(3)]
    current_model = np.random.randn(model_dim)

    print(f"\nTesting {len(attacks_to_test)} attack types:")

    for attack in attacks_to_test:
        print(f"\n{attack.__class__.__name__}:")

        for round_idx in range(n_rounds):
            gradient = attack.generate_gradient(
                model_dim, current_model, honest_gradients, f"Byzantine_{attack.__class__.__name__}"
            )

            print(f"  Round {round_idx + 1}: norm={np.linalg.norm(gradient):.4f}, "
                  f"mean={np.mean(gradient):.4f}")

            attack.update_round()

        info = attack.get_attack_info()
        print(f"  Attack info: {info}")

    # Test multi-attack scenario
    print(f"\nTesting Multi-Attack Scenario:")

    attack_configs = [
        {
            'operator_ids': ['Op_3', 'Op_4'],
            'attack_type': 'sign_flip',
            'attack_strength': 1.0
        },
        {
            'operator_ids': ['Op_5'],
            'attack_type': 'random',
            'attack_strength': 0.8,
            'noise_std': 0.3
        }
    ]

    multi_scenario = MultiAttackScenario(attack_configs)

    for round_idx in range(3):
        print(f"\nRound {round_idx + 1}:")
        for op_id in ['Op_3', 'Op_4', 'Op_5']:
            gradient = multi_scenario.generate_gradient(op_id, model_dim, current_model, honest_gradients)
            print(f"  {op_id}: norm={np.linalg.norm(gradient):.4f}")

        multi_scenario.update_round()

    summary = multi_scenario.get_attack_summary()
    print(f"\nAttack summary: {summary}")

    print("Byzantine attack generators test completed!")