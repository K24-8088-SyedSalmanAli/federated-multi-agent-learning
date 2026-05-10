"""
Unit Tests for Week 10: Model Poisoning Attacks
================================================
Run: python tests/test_poisoning.py
"""

import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.attacks.poisoning import (
    UntargetedPoisoner, TargetedPoisoner, PoisonedFedClient,
)
from src.federated.fed_client import FedClient


class TestUntargetedPoisoner(unittest.TestCase):
    """Tests for untargeted (noise injection) poisoning."""

    def test_adds_noise(self):
        """Poisoned params should differ from clean."""
        poisoner = UntargetedPoisoner(noise_scale=5.0)
        clean = {"w": torch.randn(10, 5)}
        glob = {"w": torch.zeros(10, 5)}
        poisoned = poisoner.poison_params(clean, glob)
        self.assertFalse(torch.equal(clean["w"], poisoned["w"]))

    def test_noise_scale_effect(self):
        """Higher noise scale = bigger difference."""
        clean = {"w": torch.randn(10, 5)}
        glob = {"w": torch.zeros(10, 5)}

        low = UntargetedPoisoner(noise_scale=1.0).poison_params(clean, glob)
        high = UntargetedPoisoner(noise_scale=10.0).poison_params(clean, glob)

        diff_low = (clean["w"] - low["w"]).abs().mean().item()
        diff_high = (clean["w"] - high["w"]).abs().mean().item()
        self.assertGreater(diff_high, diff_low)

    def test_preserves_shape(self):
        """Poisoned params should have same shape."""
        poisoner = UntargetedPoisoner()
        clean = {"w": torch.randn(10, 5), "b": torch.randn(10)}
        glob = {"w": torch.zeros(10, 5), "b": torch.zeros(10)}
        poisoned = poisoner.poison_params(clean, glob)
        self.assertEqual(poisoned["w"].shape, clean["w"].shape)
        self.assertEqual(poisoned["b"].shape, clean["b"].shape)

    def test_reproducible(self):
        """Same seed = same noise."""
        clean = {"w": torch.randn(10, 5)}
        glob = {"w": torch.zeros(10, 5)}
        p1 = UntargetedPoisoner(seed=42).poison_params(clean, glob)
        p2 = UntargetedPoisoner(seed=42).poison_params(clean, glob)
        torch.testing.assert_close(p1["w"], p2["w"])


class TestTargetedPoisoner(unittest.TestCase):
    """Tests for targeted (label-flip) poisoning."""

    def test_flips_target_class(self):
        """Target class labels should be flipped."""
        poisoner = TargetedPoisoner(target_class=1, flip_to=0)
        labels = np.array([0, 0, 1, 1, 1, 0])
        poisoned = poisoner.poison_labels(labels)
        expected = np.array([0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(poisoned, expected)

    def test_preserves_other_classes(self):
        """Non-target classes should remain unchanged."""
        poisoner = TargetedPoisoner(target_class=1, flip_to=0)
        labels = np.array([0, 0, 1, 1, 0])
        poisoned = poisoner.poison_labels(labels)
        self.assertEqual(poisoned[0], 0)
        self.assertEqual(poisoned[1], 0)
        self.assertEqual(poisoned[4], 0)

    def test_no_target_class_unchanged(self):
        """If no target class present, all labels stay same."""
        poisoner = TargetedPoisoner(target_class=5, flip_to=0)
        labels = np.array([0, 1, 0, 1])
        poisoned = poisoner.poison_labels(labels)
        np.testing.assert_array_equal(labels, poisoned)

    def test_does_not_modify_original(self):
        """Original array should not be modified."""
        poisoner = TargetedPoisoner(target_class=1, flip_to=0)
        labels = np.array([0, 1, 1, 0])
        original = labels.copy()
        poisoner.poison_labels(labels)
        np.testing.assert_array_equal(labels, original)


class TestPoisonedFedClient(unittest.TestCase):
    """Tests for poisoned federated client wrapper."""

    def setUp(self):
        self.input_dim = 65
        self.X = np.random.rand(100, self.input_dim).astype(np.float32)
        self.y = np.random.randint(0, 2, 100).astype(np.int64)

    def test_untargeted_returns_params(self):
        """Untargeted poisoned client should return parameters."""
        client = FedClient(org_name="Test", input_dim=self.input_dim)
        poisoned = PoisonedFedClient(client, attack_type="untargeted")
        global_params = {n: p.clone() for n, p in client.model.state_dict().items()}
        result = poisoned.train_round_poisoned(global_params, self.X, self.y, local_epochs=1)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_targeted_returns_params(self):
        """Targeted poisoned client should return parameters."""
        client = FedClient(org_name="Test", input_dim=self.input_dim)
        poisoned = PoisonedFedClient(client, attack_type="targeted")
        global_params = {n: p.clone() for n, p in client.model.state_dict().items()}
        result = poisoned.train_round_poisoned(global_params, self.X, self.y, local_epochs=1)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_untargeted_differs_from_clean(self):
        """Untargeted should produce different params than clean training."""
        client1 = FedClient(org_name="Clean", input_dim=self.input_dim)
        client2 = FedClient(org_name="Poison", input_dim=self.input_dim)

        # Same initial params
        init_params = {n: p.clone() for n, p in client1.model.state_dict().items()}
        client2.model.load_state_dict(client1.model.state_dict())

        # Clean training
        clean_result = client1.train_round(init_params, self.X, self.y, local_epochs=1)

        # Poisoned training
        poisoned_client = PoisonedFedClient(client2, attack_type="untargeted", noise_scale=10.0)
        poison_result = poisoned_client.train_round_poisoned(init_params, self.X, self.y, local_epochs=1)

        # Should be different
        first_key = list(clean_result.keys())[0]
        self.assertFalse(torch.equal(clean_result[first_key], poison_result[first_key]))

    def test_invalid_attack_type(self):
        """Invalid attack type should raise error."""
        client = FedClient(org_name="Test", input_dim=self.input_dim)
        poisoned = PoisonedFedClient(client, attack_type="invalid")
        global_params = {n: p.clone() for n, p in client.model.state_dict().items()}
        with self.assertRaises(ValueError):
            poisoned.train_round_poisoned(global_params, self.X, self.y)


if __name__ == "__main__":
    unittest.main(verbosity=2)
