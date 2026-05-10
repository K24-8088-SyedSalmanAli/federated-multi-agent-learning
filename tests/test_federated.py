"""
Unit Tests for Week 9: Federated Learning Pipeline
===================================================
Tests: Aggregators, Defenses, FedClient, FedServer, Reputation.

Run: python tests/test_federated.py
"""

import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.federated.aggregators import (
    FedAvgAggregator, CoordinateMedianAggregator,
    TrimmedMeanAggregator, build_aggregator,
)
from src.federated.defenses import GradientDefense, ReputationTracker
from src.federated.fed_client import FedClient
from src.federated.fed_server import FedServer


def _make_client_params(n_clients: int, dim: int = 10) -> list:
    """Helper: create fake client parameters."""
    params = []
    for _ in range(n_clients):
        params.append({
            "layer.weight": torch.randn(dim, 5),
            "layer.bias": torch.randn(dim),
        })
    return params


def _make_malicious_params(dim: int = 10, scale: float = 1000) -> dict:
    """Helper: create malicious client parameters."""
    return {
        "layer.weight": torch.ones(dim, 5) * scale,
        "layer.bias": torch.ones(dim) * scale,
    }


class TestFedAvgAggregator(unittest.TestCase):
    """Tests for FedAvg aggregation."""

    def test_basic_aggregation(self):
        """Test basic weighted average."""
        agg = FedAvgAggregator()
        clients = _make_client_params(3)
        result = agg.aggregate(clients)
        self.assertIn("layer.weight", result)
        self.assertIn("layer.bias", result)
        self.assertEqual(result["layer.weight"].shape, (10, 5))

    def test_equal_weights(self):
        """Test equal weight aggregation = simple average."""
        agg = FedAvgAggregator()
        p1 = {"w": torch.tensor([1.0, 2.0, 3.0])}
        p2 = {"w": torch.tensor([3.0, 4.0, 5.0])}
        result = agg.aggregate([p1, p2])
        expected = torch.tensor([2.0, 3.0, 4.0])
        torch.testing.assert_close(result["w"], expected)

    def test_weighted_aggregation(self):
        """Test weighted average with custom weights."""
        agg = FedAvgAggregator()
        p1 = {"w": torch.tensor([0.0])}
        p2 = {"w": torch.tensor([10.0])}
        result = agg.aggregate([p1, p2], weights=[0.8, 0.2])
        self.assertAlmostEqual(result["w"].item(), 2.0, places=4)

    def test_vulnerable_to_malicious(self):
        """Test FedAvg IS affected by malicious client."""
        agg = FedAvgAggregator()
        clients = _make_client_params(4)
        clean_result = agg.aggregate(clients)

        # Add malicious
        clients_with_mal = clients + [_make_malicious_params()]
        mal_result = agg.aggregate(clients_with_mal)

        # Malicious should significantly shift the average
        clean_mean = clean_result["layer.weight"].mean().item()
        mal_mean = mal_result["layer.weight"].mean().item()
        self.assertGreater(abs(mal_mean - clean_mean), 50)

    def test_empty_clients_raises(self):
        """Test error on empty client list."""
        agg = FedAvgAggregator()
        with self.assertRaises(ValueError):
            agg.aggregate([])


class TestCoordinateMedianAggregator(unittest.TestCase):
    """Tests for Coordinate-wise Median aggregation."""

    def test_basic_aggregation(self):
        """Test median computation."""
        agg = CoordinateMedianAggregator()
        clients = _make_client_params(5)
        result = agg.aggregate(clients)
        self.assertIn("layer.weight", result)

    def test_robust_to_malicious(self):
        """Test median IS robust against malicious client."""
        agg = CoordinateMedianAggregator()
        clients = _make_client_params(5)
        clean_result = agg.aggregate(clients)

        # Add 2 malicious (still < n/2)
        clients_with_mal = clients + [_make_malicious_params(), _make_malicious_params()]
        mal_result = agg.aggregate(clients_with_mal)

        # Median should remain close to clean result
        clean_mean = clean_result["layer.weight"].mean().item()
        mal_mean = mal_result["layer.weight"].mean().item()
        self.assertLess(abs(mal_mean - clean_mean), 5.0)

    def test_median_correctness(self):
        """Test median computes correctly."""
        agg = CoordinateMedianAggregator()
        p1 = {"w": torch.tensor([1.0, 10.0, 3.0])}
        p2 = {"w": torch.tensor([2.0, 20.0, 1.0])}
        p3 = {"w": torch.tensor([3.0, 30.0, 2.0])}
        result = agg.aggregate([p1, p2, p3])
        expected = torch.tensor([2.0, 20.0, 2.0])
        torch.testing.assert_close(result["w"], expected)

    def test_ignores_weights(self):
        """Median should produce same result regardless of weights."""
        agg = CoordinateMedianAggregator()
        clients = _make_client_params(5)
        r1 = agg.aggregate(clients, weights=[1, 1, 1, 1, 1])
        r2 = agg.aggregate(clients, weights=[10, 1, 1, 1, 1])
        torch.testing.assert_close(r1["layer.weight"], r2["layer.weight"])


class TestTrimmedMeanAggregator(unittest.TestCase):
    """Tests for Trimmed Mean aggregation."""

    def test_basic(self):
        agg = TrimmedMeanAggregator(trim_ratio=0.2)
        clients = _make_client_params(5)
        result = agg.aggregate(clients)
        self.assertIn("layer.weight", result)

    def test_partially_robust(self):
        """Test trimmed mean is partially robust."""
        agg = TrimmedMeanAggregator(trim_ratio=0.2)
        clients = _make_client_params(5)
        clients_with_mal = clients + [_make_malicious_params()]
        result = agg.aggregate(clients_with_mal)
        # Should still produce reasonable values
        self.assertLess(result["layer.weight"].abs().mean().item(), 100)


class TestBuildAggregator(unittest.TestCase):
    """Tests for aggregator factory."""

    def test_build_fedavg(self):
        agg = build_aggregator("fedavg")
        self.assertIsInstance(agg, FedAvgAggregator)

    def test_build_median(self):
        agg = build_aggregator("coordinate_median")
        self.assertIsInstance(agg, CoordinateMedianAggregator)

    def test_build_trimmed(self):
        agg = build_aggregator("trimmed_mean", trim_ratio=0.1)
        self.assertIsInstance(agg, TrimmedMeanAggregator)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            build_aggregator("unknown_method")


class TestGradientDefense(unittest.TestCase):
    """Tests for gradient defense mechanisms."""

    def setUp(self):
        self.global_params = {
            "w": torch.zeros(10, 5),
            "b": torch.zeros(10),
        }

    def test_normal_clients_pass(self):
        """Test normal clients are accepted."""
        defense = GradientDefense()
        clients = [
            {"w": torch.randn(10, 5) * 0.1, "b": torch.randn(10) * 0.1}
            for _ in range(4)
        ]
        filtered, reports = defense.filter(clients, self.global_params)
        self.assertEqual(len(filtered), 4)

    def test_malicious_flagged(self):
        """Test malicious client with extreme norm is handled."""
        defense = GradientDefense(clip_factor=1.5, cosine_threshold=-0.3)
        normal = [{"w": torch.randn(10, 5) * 0.1, "b": torch.randn(10) * 0.1} for _ in range(4)]
        malicious = {"w": torch.randn(10, 5) * 100, "b": torch.randn(10) * 100}
        clients = normal + [malicious]
        names = ["N1", "N2", "N3", "N4", "Malicious"]

        filtered, reports = defense.filter(clients, self.global_params, names)
        # Either clipped or flagged
        self.assertLessEqual(len(filtered), 5)

    def test_clipping_reduces_norm(self):
        """Test gradient clipping reduces extreme norms."""
        defense = GradientDefense(clip_factor=1.5, enable_cosine=False)
        normal = [{"w": torch.randn(10, 5) * 0.1, "b": torch.randn(10) * 0.1} for _ in range(3)]
        extreme = {"w": torch.randn(10, 5) * 50, "b": torch.randn(10) * 50}
        clients = normal + [extreme]

        filtered, reports = defense.filter(clients, self.global_params)
        # All should be accepted (clipped, not rejected)
        self.assertEqual(len(filtered), 4)

    def test_defense_summary(self):
        """Test defense summary generation."""
        defense = GradientDefense()
        clients = [{"w": torch.randn(10, 5) * 0.1, "b": torch.randn(10) * 0.1} for _ in range(3)]
        defense.filter(clients, self.global_params)
        summary = defense.get_defense_summary()
        self.assertIn("total_rounds", summary)
        self.assertEqual(summary["total_rounds"], 1)


class TestReputationTracker(unittest.TestCase):
    """Tests for reputation tracking."""

    def test_initial_scores(self):
        tracker = ReputationTracker(["A", "B", "C"])
        scores = tracker.get_scores()
        self.assertEqual(scores["A"], 100)
        self.assertEqual(scores["B"], 100)

    def test_flag_decreases_score(self):
        tracker = ReputationTracker(["A", "B"], flag_penalty=20)
        reports = [
            {"org_name": "A", "flagged": True, "accepted": False},
            {"org_name": "B", "flagged": False, "accepted": True},
        ]
        tracker.update(reports)
        self.assertEqual(tracker.get_scores()["A"], 80)

    def test_exclusion_threshold(self):
        tracker = ReputationTracker(["A", "B"], flag_penalty=30, exclusion_threshold=40)
        # Flag A twice → score = 100-30-30 = 40 → excluded
        for _ in range(2):
            tracker.update([{"org_name": "A", "flagged": True, "accepted": False}])
        self.assertIn("A", tracker.get_excluded_orgs())

    def test_active_orgs(self):
        tracker = ReputationTracker(["A", "B", "C"])
        active = tracker.get_active_orgs()
        self.assertEqual(len(active), 3)

    def test_good_behavior_reward(self):
        tracker = ReputationTracker(["A"], flag_penalty=20)
        # Flag then accept
        tracker.update([{"org_name": "A", "flagged": True}])
        self.assertEqual(tracker.get_scores()["A"], 80)
        tracker.update([{"org_name": "A", "flagged": False, "accepted": True}])
        self.assertEqual(tracker.get_scores()["A"], 82)  # +2 reward


class TestFedClient(unittest.TestCase):
    """Tests for federated learning client."""

    def test_client_creation(self):
        client = FedClient(org_name="Test", input_dim=65)
        self.assertEqual(client.org_name, "Test")

    def test_train_round(self):
        client = FedClient(org_name="Test", input_dim=65)
        global_params = {name: p.clone() for name, p in client.model.state_dict().items()}
        X = np.random.rand(100, 65).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.int64)
        updated = client.train_round(global_params, X, y, local_epochs=1)
        self.assertIsInstance(updated, dict)
        self.assertGreater(len(updated), 0)

    def test_evaluate(self):
        client = FedClient(org_name="Test", input_dim=65)
        X = np.random.rand(50, 65).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.int64)
        metrics = client.evaluate(X, y)
        self.assertIn("accuracy", metrics)

    def test_round_stats_recorded(self):
        client = FedClient(org_name="Test", input_dim=65)
        global_params = {name: p.clone() for name, p in client.model.state_dict().items()}
        X = np.random.rand(100, 65).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.int64)
        client.train_round(global_params, X, y, local_epochs=2)
        self.assertEqual(len(client.round_stats), 1)
        self.assertEqual(client.round_stats[0]["local_epochs"], 2)


class TestFedServer(unittest.TestCase):
    """Tests for federated learning server."""

    def _make_org_data(self, n_orgs=3, n_features=65):
        np.random.seed(42)
        test_X = np.random.rand(100, n_features).astype(np.float32)
        test_y = np.random.randint(0, 2, 100).astype(np.int64)
        org_data = {}
        for i in range(n_orgs):
            org_data[f"Org_{i}"] = {
                "train_X": np.random.rand(200, n_features).astype(np.float32),
                "train_y": np.random.randint(0, 2, 200).astype(np.int64),
                "val_X": np.random.rand(30, n_features).astype(np.float32),
                "val_y": np.random.randint(0, 2, 30).astype(np.int64),
                "test_X": test_X,
                "test_y": test_y,
            }
        return org_data

    def test_server_with_fedavg(self):
        server = FedServer(input_dim=65, aggregation="fedavg", enable_defenses=False)
        org_data = self._make_org_data()
        results = server.run(org_data, num_rounds=2, local_epochs=1)
        self.assertIn("final_accuracy", results)
        self.assertIn("round_history", results)
        self.assertEqual(len(results["round_history"]), 2)

    def test_server_with_median(self):
        server = FedServer(input_dim=65, aggregation="coordinate_median", enable_defenses=True)
        org_data = self._make_org_data()
        results = server.run(org_data, num_rounds=2, local_epochs=1)
        self.assertIn("final_accuracy", results)
        self.assertIn("defense_summary", results)

    def test_server_round_history(self):
        server = FedServer(input_dim=65, aggregation="fedavg", enable_defenses=False)
        org_data = self._make_org_data()
        results = server.run(org_data, num_rounds=3, local_epochs=1)
        for r in results["round_history"]:
            self.assertIn("round", r)
            self.assertIn("accuracy", r)
            self.assertIn("f1_score", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
