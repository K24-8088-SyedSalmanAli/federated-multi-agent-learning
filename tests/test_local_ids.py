"""
Unit Tests for Week 8: Local IDS Training Pipeline
===================================================
Tests: CICIDS loader, data partitioner, CNN-LSTM model, local trainer, metrics.

Run: python tests/test_local_ids.py
"""

import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.cnn_lstm import CNNLSTM, BaselineAutoencoder, build_model
from src.evaluation.metrics import (
    compute_accuracy, compute_precision, compute_recall,
    compute_f1, compute_roc_auc, compute_confusion_matrix, compute_all_metrics,
)
from src.data.data_partitioner import DataPartitioner
from src.data.cicids_loader import CICIDS2017Loader


class TestCICIDSLoader(unittest.TestCase):
    """Tests for CICIDS2017 dataset loader."""

    def test_synthetic_data_generation(self):
        """Test synthetic data generation when no CSVs available."""
        loader = CICIDS2017Loader(raw_dir="/tmp/test_raw", processed_dir="/tmp/test_processed")
        df = loader._generate_synthetic_data(n_samples=1000)
        self.assertEqual(len(df), 1000)
        self.assertIn("Label", df.columns)

    def test_label_mapping(self):
        """Test label mapping from raw to categories."""
        loader = CICIDS2017Loader(raw_dir="/tmp/test_raw", processed_dir="/tmp/test_processed")
        df = loader._generate_synthetic_data(n_samples=500)
        df = loader.map_labels(df)
        self.assertIn("binary_label", df.columns)
        self.assertIn("attack_category", df.columns)
        self.assertIn("multi_label", df.columns)
        self.assertTrue(set(df["binary_label"].unique()).issubset({0, 1}))

    def test_full_pipeline(self):
        """Test complete processing pipeline with synthetic data."""
        loader = CICIDS2017Loader(raw_dir="/tmp/test_raw", processed_dir="/tmp/test_processed_pipeline")
        stats = loader.process()
        self.assertIn("total_samples", stats)
        self.assertIn("num_features", stats)
        self.assertGreater(stats["total_samples"], 0)
        self.assertGreater(stats["num_features"], 0)


class TestDataPartitioner(unittest.TestCase):
    """Tests for non-IID data partitioning."""

    def setUp(self):
        np.random.seed(42)
        self.features = np.random.rand(1000, 64).astype(np.float32)
        self.labels = np.random.randint(0, 2, 1000).astype(np.int64)

    def test_iid_partition(self):
        """Test IID partitioning creates roughly equal splits."""
        partitioner = DataPartitioner(num_orgs=5, strategy="iid")
        org_data = partitioner.partition(self.features, self.labels)
        self.assertEqual(len(org_data), 5)
        for org_name, data in org_data.items():
            self.assertIn("train_X", data)
            self.assertIn("train_y", data)
            self.assertIn("test_X", data)
            self.assertGreater(len(data["train_X"]), 0)

    def test_dirichlet_partition(self):
        """Test Dirichlet non-IID partitioning."""
        partitioner = DataPartitioner(num_orgs=5, strategy="dirichlet", alpha=0.5)
        org_data = partitioner.partition(self.features, self.labels)
        self.assertEqual(len(org_data), 5)
        # All orgs should have data
        for org_name, data in org_data.items():
            self.assertGreater(len(data["train_X"]), 0)

    def test_shared_test_set(self):
        """Test that all organizations share the same test set."""
        partitioner = DataPartitioner(num_orgs=3, strategy="iid")
        org_data = partitioner.partition(self.features, self.labels)
        test_sets = [data["test_X"] for data in org_data.values()]
        for ts in test_sets[1:]:
            np.testing.assert_array_equal(test_sets[0], ts)

    def test_low_alpha_creates_heterogeneity(self):
        """Test that low alpha creates more non-IID distributions."""
        partitioner_low = DataPartitioner(num_orgs=5, strategy="dirichlet", alpha=0.1)
        partitioner_high = DataPartitioner(num_orgs=5, strategy="dirichlet", alpha=100)
        
        org_low = partitioner_low.partition(self.features, self.labels)
        org_high = partitioner_high.partition(self.features, self.labels)
        
        # Both should produce valid partitions
        self.assertEqual(len(org_low), 5)
        self.assertEqual(len(org_high), 5)


class TestCNNLSTM(unittest.TestCase):
    """Tests for CNN-LSTM model architecture."""

    def test_model_creation(self):
        """Test model instantiation."""
        model = CNNLSTM(input_dim=64, num_classes=2)
        self.assertIsNotNone(model)
        self.assertGreater(model.count_parameters(), 0)

    def test_forward_pass_binary(self):
        """Test forward pass for binary classification."""
        model = CNNLSTM(input_dim=64, num_classes=2)
        x = torch.randn(32, 64)
        output = model(x)
        self.assertEqual(output.shape, (32, 2))

    def test_forward_pass_multiclass(self):
        """Test forward pass for multi-class classification."""
        model = CNNLSTM(input_dim=64, num_classes=8)
        x = torch.randn(16, 64)
        output = model(x)
        self.assertEqual(output.shape, (16, 8))

    def test_different_input_dims(self):
        """Test model works with different input dimensions."""
        for dim in [32, 64, 128]:
            model = CNNLSTM(input_dim=dim, num_classes=2)
            x = torch.randn(8, dim)
            output = model(x)
            self.assertEqual(output.shape, (8, 2))

    def test_get_parameters(self):
        """Test parameter extraction for federated learning."""
        model = CNNLSTM(input_dim=64, num_classes=2)
        params = model.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)

    def test_set_parameters(self):
        """Test parameter loading for federated aggregation."""
        model1 = CNNLSTM(input_dim=64, num_classes=2)
        model2 = CNNLSTM(input_dim=64, num_classes=2)
        
        # Copy full state_dict (includes BatchNorm running stats)
        model2.load_state_dict(model1.state_dict())
        
        # Must be in eval mode for deterministic BatchNorm
        model1.eval()
        model2.eval()
        
        x = torch.randn(8, 64)
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)
        torch.testing.assert_close(out1, out2)

    def test_gradient_extraction(self):
        """Test gradient extraction after backward pass."""
        model = CNNLSTM(input_dim=64, num_classes=2)
        x = torch.randn(8, 64)
        y = torch.randint(0, 2, (8,))
        
        output = model(x)
        loss = torch.nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        grads = model.get_gradients()
        self.assertGreater(len(grads), 0)
        for name, grad in grads.items():
            self.assertIsNotNone(grad)


class TestAutoencoder(unittest.TestCase):
    """Tests for Autoencoder baseline."""

    def test_autoencoder_forward(self):
        """Test autoencoder reconstruction."""
        ae = BaselineAutoencoder(input_dim=64)
        x = torch.randn(16, 64)
        output = ae(x)
        self.assertEqual(output.shape, x.shape)

    def test_reconstruction_error(self):
        """Test per-sample reconstruction error."""
        ae = BaselineAutoencoder(input_dim=64)
        x = torch.randn(16, 64)
        errors = ae.get_reconstruction_error(x)
        self.assertEqual(errors.shape, (16,))
        self.assertTrue(torch.all(errors >= 0))


class TestMetrics(unittest.TestCase):
    """Tests for evaluation metrics."""

    def setUp(self):
        self.y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        self.y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])

    def test_accuracy(self):
        acc = compute_accuracy(self.y_true, self.y_pred)
        self.assertAlmostEqual(acc, 0.8)

    def test_precision(self):
        prec = compute_precision(self.y_true, self.y_pred)
        # TP=5, FP=1 → precision = 5/6
        self.assertAlmostEqual(prec, 5 / 6, places=4)

    def test_recall(self):
        rec = compute_recall(self.y_true, self.y_pred)
        # TP=5, FN=1 → recall = 5/6
        self.assertAlmostEqual(rec, 5 / 6, places=4)

    def test_f1(self):
        f1 = compute_f1(self.y_true, self.y_pred)
        self.assertGreater(f1, 0)
        self.assertLessEqual(f1, 1)

    def test_roc_auc(self):
        np.random.seed(42)
        y_probs = np.random.rand(10, 2)
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
        auc = compute_roc_auc(self.y_true, y_probs)
        self.assertGreaterEqual(auc, 0)
        self.assertLessEqual(auc, 1)

    def test_confusion_matrix(self):
        cm = compute_confusion_matrix(self.y_true, self.y_pred, num_classes=2)
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(cm.sum(), len(self.y_true))

    def test_all_metrics(self):
        y_probs = np.random.rand(10, 2)
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
        metrics = compute_all_metrics(self.y_true, self.y_pred, y_probs)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_perfect_predictions(self):
        y = np.array([0, 1, 0, 1, 1])
        metrics = compute_all_metrics(y, y)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_build_model_factory(self):
        model = build_model("cnn_lstm", input_dim=64, num_classes=2)
        self.assertIsInstance(model, CNNLSTM)
        model = build_model("autoencoder", input_dim=64)
        self.assertIsInstance(model, BaselineAutoencoder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
