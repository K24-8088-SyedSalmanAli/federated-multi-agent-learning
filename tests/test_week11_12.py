"""
Unit Tests for Week 11 & 12
============================
Week 11: Blockchain logging, tamper detection, scalability
Week 12: Statistical validation, hypothesis testing

Run: python tests/test_week11_12.py
"""

import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blockchain.blockchain_logger import BlockchainLogger, Block
from src.evaluation.statistical_validation import StatisticalValidator


class TestBlock(unittest.TestCase):
    def test_block_creation(self):
        block = Block(index=1, transactions=[{"type": "test"}], previous_hash="abc")
        self.assertEqual(block.index, 1)
        self.assertIsNotNone(block.hash)

    def test_block_hash_deterministic(self):
        b1 = Block(index=1, transactions=[{"a": 1}], previous_hash="0", timestamp=1000)
        b2 = Block(index=1, transactions=[{"a": 1}], previous_hash="0", timestamp=1000)
        self.assertEqual(b1.hash, b2.hash)

    def test_different_data_different_hash(self):
        b1 = Block(index=1, transactions=[{"a": 1}], previous_hash="0", timestamp=1000)
        b2 = Block(index=1, transactions=[{"a": 2}], previous_hash="0", timestamp=1000)
        self.assertNotEqual(b1.hash, b2.hash)


class TestBlockchainLogger(unittest.TestCase):
    def setUp(self):
        self.bl = BlockchainLogger(num_peers=4)

    def test_genesis_block(self):
        self.assertEqual(len(self.bl.chain), 1)
        self.assertEqual(self.bl.chain[0].index, 0)

    def test_log_gradient(self):
        tx = self.bl.log_gradient_submission("Org_A", 50.0, True, 1)
        self.assertIn("hash", tx)
        self.assertEqual(tx["org_name"], "Org_A")

    def test_chain_integrity_valid(self):
        for i in range(10):
            self.bl.log_gradient_submission(f"Org_{i%3}", 50.0, True, i)
        self.bl.flush()
        self.assertTrue(self.bl.verify_chain_integrity())

    def test_tamper_detection(self):
        for i in range(20):
            self.bl.log_gradient_submission(f"Org_{i%3}", 50.0, True, i)
        self.bl.flush()
        result = self.bl.simulate_tamper_detection()
        self.assertTrue(result["tamper_detected"])

    def test_reputation_accepted(self):
        self.bl.log_gradient_submission("Org_A", 50.0, True, 1)
        self.assertGreaterEqual(self.bl.reputation_scores["Org_A"], 100)

    def test_reputation_rejected(self):
        self.bl.log_gradient_submission("Org_A", 50.0, False, 1)
        self.assertEqual(self.bl.reputation_scores["Org_A"], 80)

    def test_exclusion(self):
        for i in range(5):
            self.bl.log_gradient_submission("Org_Bad", 50.0, False, i)
        excluded = self.bl.get_excluded_orgs()
        self.assertIn("Org_Bad", excluded)

    def test_query_audit_trail(self):
        self.bl.log_gradient_submission("Org_A", 50.0, True, 1)
        self.bl.log_gradient_submission("Org_B", 60.0, True, 1)
        trail = self.bl.query_audit_trail(org_name="Org_A")
        self.assertEqual(len(trail), 1)

    def test_performance_metrics(self):
        for i in range(10):
            self.bl.log_gradient_submission("Org_A", 50.0, True, i)
        metrics = self.bl.get_performance_metrics()
        self.assertIn("total_transactions", metrics)
        self.assertIn("throughput_tps", metrics)
        self.assertEqual(metrics["total_transactions"], 10)

    def test_scalability_benchmark(self):
        results = self.bl.run_scalability_benchmark(peer_configs=[2, 4], n_transactions=20)
        self.assertIn("2_peers", results)
        self.assertIn("4_peers", results)

    def test_log_detection_event(self):
        tx = self.bl.log_detection_event("Org_A", 0.95, "DDoS", 1)
        self.assertEqual(tx["type"], "detection_event")

    def test_log_response_action(self):
        tx = self.bl.log_response_action("payment_hold", "Org_A", 1)
        self.assertEqual(tx["type"], "response_action")


class TestStatisticalValidator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.fed = np.random.normal(0.96, 0.01, 30)
        self.cent = np.random.normal(0.95, 0.012, 30)
        self.validator = StatisticalValidator(significance_level=0.05)

    def test_paired_t_test(self):
        result = self.validator.paired_t_test(self.fed, self.cent)
        self.assertIn("t_statistic", result)
        self.assertIn("p_value", result)
        self.assertIn("significant", result)

    def test_wilcoxon_test(self):
        result = self.validator.wilcoxon_test(self.fed, self.cent)
        self.assertIn("p_value", result)

    def test_cohens_d(self):
        result = self.validator.cohens_d(self.fed, self.cent)
        self.assertIn("d_value", result)
        self.assertIn("magnitude", result)

    def test_full_validation(self):
        result = self.validator.run_validation(self.fed, self.cent)
        self.assertIn("paired_t_test", result)
        self.assertIn("wilcoxon_test", result)
        self.assertIn("cohens_d", result)
        self.assertIn("h1b_verdict", result)
        self.assertEqual(result["num_runs"], 30)

    def test_identical_scores(self):
        same = np.ones(30) * 0.95
        result = self.validator.paired_t_test(same, same)
        self.assertFalse(result["significant"])

    def test_clearly_different(self):
        high = np.random.normal(0.99, 0.001, 30)
        low = np.random.normal(0.50, 0.001, 30)
        result = self.validator.paired_t_test(high, low)
        self.assertTrue(result["significant"])

    def test_cohens_d_magnitude(self):
        # Large effect
        a = np.random.normal(10, 1, 30)
        b = np.random.normal(5, 1, 30)
        result = self.validator.cohens_d(a, b)
        self.assertEqual(result["magnitude"], "Large")


if __name__ == "__main__":
    unittest.main(verbosity=2)
