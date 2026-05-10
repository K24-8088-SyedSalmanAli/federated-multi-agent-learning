"""
Model Poisoning Attacks for Federated Learning
===============================================
Implements adversarial attacks to test Byzantine-robust defenses:

1. Untargeted Poisoning — Gaussian noise injection into gradients
   Goal: Degrade overall model accuracy

2. Targeted Poisoning — Label-flip attack on specific attack category
   Goal: Misclassify specific attacks as benign while maintaining overall accuracy

Reference: Proposal Section 6.9 (Adversarial Model Poisoning Defense Strategy)

Usage:
    from src.attacks.untargeted_poisoning import UntargetedPoisoner
    from src.attacks.targeted_poisoning import TargetedPoisoner
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class UntargetedPoisoner:
    """
    Untargeted Model Poisoning Attack.
    
    Injects Gaussian noise into model gradients/parameters to degrade
    overall detection accuracy. Simulates a compromised organization
    submitting corrupted model updates.
    
    Args:
        noise_scale: Multiplier for Gaussian noise (higher = more aggressive)
        seed: Random seed for reproducibility
    """

    def __init__(self, noise_scale: float = 5.0, seed: int = 42):
        self.noise_scale = noise_scale
        self.seed = seed

    def poison_params(
        self,
        clean_params: Dict[str, torch.Tensor],
        global_params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Add Gaussian noise to model parameters.
        
        Noise is scaled relative to the update magnitude to make
        the attack proportional to normal updates.
        """
        torch.manual_seed(self.seed)
        poisoned = {}

        for key in clean_params:
            # Compute update magnitude
            update = clean_params[key].float() - global_params[key].float()
            update_std = update.std().item() if update.numel() > 1 else 1.0
            if update_std == 0:
                update_std = 1.0

            # Add scaled Gaussian noise (convert to float first for non-float tensors)
            param_float = clean_params[key].float()
            noise = torch.randn_like(param_float) * update_std * self.noise_scale
            poisoned[key] = param_float + noise

        return poisoned


class TargetedPoisoner:
    """
    Targeted Model Poisoning Attack (Label-Flip).
    
    Flips labels of a specific attack class to BENIGN during local training,
    causing the model to misclassify that attack type while maintaining
    overall accuracy (to evade detection).
    
    Args:
        target_class: Class label to flip (attack type)
        flip_to: Class label to flip to (usually 0 = BENIGN)
        seed: Random seed
    """

    def __init__(self, target_class: int = 1, flip_to: int = 0, seed: int = 42):
        self.target_class = target_class
        self.flip_to = flip_to
        self.seed = seed

    def poison_labels(
        self,
        train_y: np.ndarray,
    ) -> np.ndarray:
        """
        Flip labels of target class to flip_to class.
        Returns modified labels array.
        """
        poisoned_y = train_y.copy()
        mask = poisoned_y == self.target_class
        n_flipped = mask.sum()
        poisoned_y[mask] = self.flip_to

        logger.info("  [TARGETED POISON] Flipped %d labels: class %d → class %d",
                    n_flipped, self.target_class, self.flip_to)
        return poisoned_y


class PoisonedFedClient:
    """
    A malicious federated client that applies poisoning during training.
    
    Wraps a normal FedClient and injects poison based on attack type.
    """

    def __init__(
        self,
        fed_client,
        attack_type: str = "untargeted",
        noise_scale: float = 5.0,
        target_class: int = 1,
        flip_to: int = 0,
    ):
        self.client = fed_client
        self.attack_type = attack_type
        self.untargeted = UntargetedPoisoner(noise_scale=noise_scale)
        self.targeted = TargetedPoisoner(target_class=target_class, flip_to=flip_to)

    def train_round_poisoned(
        self,
        global_params: Dict[str, torch.Tensor],
        train_X: np.ndarray,
        train_y: np.ndarray,
        local_epochs: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute a poisoned training round.
        
        For untargeted: Train normally, then add noise to parameters.
        For targeted: Flip labels before training, train on poisoned labels.
        """
        if self.attack_type == "untargeted":
            # Train normally first
            clean_params = self.client.train_round(
                global_params, train_X, train_y, local_epochs
            )
            # Then add noise
            poisoned_params = self.untargeted.poison_params(clean_params, global_params)
            return poisoned_params

        elif self.attack_type == "targeted":
            # Flip labels, then train on poisoned data
            poisoned_y = self.targeted.poison_labels(train_y)
            poisoned_params = self.client.train_round(
                global_params, train_X, poisoned_y, local_epochs
            )
            return poisoned_params

        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")


if __name__ == "__main__":
    print("Testing Poisoning Attacks...")

    # Test untargeted
    poisoner = UntargetedPoisoner(noise_scale=5.0)
    clean = {"w": torch.randn(10, 5), "b": torch.randn(10)}
    glob = {"w": torch.zeros(10, 5), "b": torch.zeros(10)}
    poisoned = poisoner.poison_params(clean, glob)
    print(f"Clean norm: {torch.norm(torch.cat([p.flatten() for p in clean.values()])):.4f}")
    print(f"Poisoned norm: {torch.norm(torch.cat([p.flatten() for p in poisoned.values()])):.4f}")

    # Test targeted
    targeter = TargetedPoisoner(target_class=1, flip_to=0)
    labels = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    poisoned_labels = targeter.poison_labels(labels)
    print(f"\nOriginal labels:  {labels}")
    print(f"Poisoned labels:  {poisoned_labels}")
    print(f"Flipped: {(labels != poisoned_labels).sum()} labels")

    print("\n✓ Poisoning attacks working!")
