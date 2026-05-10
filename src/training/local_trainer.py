"""
Local IDS Trainer
================
Trains a CNN-LSTM IDS model locally for each supply chain organization.
This is the foundation for federated learning — each org trains independently 
on its own data partition.

Usage:
    python -m src.training.local_trainer                          # Train all 5 orgs
    python -m src.training.local_trainer --org Manufacturer_Assembly  # Train specific org
    python -m src.training.local_trainer --centralized            # Centralized baseline
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.cnn_lstm import CNNLSTM, build_model
from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LocalTrainer:
    """
    Trains a local CNN-LSTM IDS model for a single organization.

    Supports:
    - Standalone training (full epochs on local data)
    - Federated round training (few epochs per round, returns gradients)
    - Early stopping based on validation loss
    """

    def __init__(
        self,
        org_name: str,
        input_dim: int,
        num_classes: int = 2,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 256,
        device: str = "auto",
    ):
        self.org_name = org_name
        self.batch_size = batch_size

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model
        self.model = CNNLSTM(input_dim=input_dim, num_classes=num_classes).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        # Training history
        self.history: Dict[str, List] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        logger.info("[%s] Initialized on %s | %d parameters",
                     org_name, self.device, self.model.count_parameters())

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train_standalone(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        epochs: int = 30,
        patience: int = 5,
    ) -> Dict:
        """
        Full standalone training for a single organization.
        Used for baseline comparison against federated approach.
        """
        logger.info("[%s] Starting standalone training (%d epochs, patience=%d)",
                     self.org_name, epochs, patience)

        train_loader = self._create_dataloader(train_X, train_y)
        val_loader = self._create_dataloader(val_X, val_y, shuffle=False)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self._train_epoch(train_loader)
            # Validate
            val_loss, val_acc = self._evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            self.scheduler.step()

            if epoch % 5 == 0 or epoch == 1:
                logger.info("  Epoch %d/%d | Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f",
                             epoch, epochs, train_loss, train_acc, val_loss, val_acc)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("  Early stopping at epoch %d", epoch)
                    break

        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        elapsed = time.time() - start_time
        logger.info("[%s] Training complete in %.1fs | Best val loss: %.4f",
                     self.org_name, elapsed, best_val_loss)

        return {
            "org_name": self.org_name,
            "epochs_trained": len(self.history["train_loss"]),
            "best_val_loss": float(best_val_loss),
            "training_time_seconds": round(elapsed, 2),
            "final_train_acc": float(self.history["train_acc"][-1]),
            "final_val_acc": float(self.history["val_acc"][-1]),
        }

    def train_federated_round(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        epochs: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Train for a single federated round (few local epochs).
        Returns model parameters after training.
        """
        train_loader = self._create_dataloader(train_X, train_y)

        for epoch in range(epochs):
            self._train_epoch(train_loader)

        return self.model.get_parameters()

    def _train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y_batch)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += len(y_batch)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def _evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item() * len(y_batch)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += len(y_batch)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def evaluate_full(
        self,
        test_X: np.ndarray,
        test_y: np.ndarray,
    ) -> Dict:
        """
        Full evaluation with all metrics: accuracy, precision, recall, F1, ROC-AUC.
        """
        self.model.eval()
        test_loader = self._create_dataloader(test_X, test_y, shuffle=False)

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.numpy())

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        metrics = compute_all_metrics(all_labels, all_preds, all_probs)
        metrics["org_name"] = self.org_name
        return metrics

    def save_model(self, path: str):
        """Save model state dict."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "org_name": self.org_name,
        }, path)
        logger.info("[%s] Model saved to %s", self.org_name, path)

    def load_model(self, path: str):
        """Load model state dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        logger.info("[%s] Model loaded from %s", self.org_name, path)


def train_all_organizations(
    processed_dir: str = "data/processed",
    results_dir: str = "data/results",
    epochs: int = 30,
    num_classes: int = 2,
) -> Dict:
    """Train local IDS models for all organizations."""
    processed_dir = Path(processed_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find organization directories
    org_dirs = [d for d in processed_dir.iterdir() if d.is_dir() and d.name.startswith(("Supplier", "Manufacturer", "Logistics", "OEM"))]

    if not org_dirs:
        logger.error("No organization data found in %s. Run data_partitioner first.", processed_dir)
        return {}

    all_results = {}
    for org_dir in sorted(org_dirs):
        org_name = org_dir.name
        logger.info("\n" + "=" * 60)
        logger.info("Training: %s", org_name)
        logger.info("=" * 60)

        # Load data
        train_X = np.load(org_dir / "train_X.npy")
        train_y = np.load(org_dir / "train_y.npy")
        val_X = np.load(org_dir / "val_X.npy")
        val_y = np.load(org_dir / "val_y.npy")
        test_X = np.load(org_dir / "test_X.npy")
        test_y = np.load(org_dir / "test_y.npy")

        input_dim = train_X.shape[1]
        logger.info("  Data: %d train, %d val, %d test | %d features",
                     len(train_X), len(val_X), len(test_X), input_dim)

        # Train
        trainer = LocalTrainer(
            org_name=org_name,
            input_dim=input_dim,
            num_classes=num_classes,
        )
        train_result = trainer.train_standalone(train_X, train_y, val_X, val_y, epochs=epochs)

        # Evaluate
        eval_result = trainer.evaluate_full(test_X, test_y)

        # Save model
        model_path = results_dir / f"{org_name}_model.pt"
        trainer.save_model(str(model_path))

        # Combine results
        all_results[org_name] = {**train_result, **eval_result}

    # Save all results
    results_path = results_dir / "local_training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nAll results saved to %s", results_path)

    # Print summary
    print("\n" + "=" * 80)
    print("LOCAL IDS TRAINING RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Organization':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 80)
    for org_name, results in all_results.items():
        print(f"{org_name:<30} {results.get('accuracy', 0):>10.4f} {results.get('precision', 0):>10.4f} "
              f"{results.get('recall', 0):>10.4f} {results.get('f1_score', 0):>10.4f} "
              f"{results.get('roc_auc', 0):>10.4f}")
    print("=" * 80)

    return all_results


def train_centralized_baseline(
    processed_dir: str = "data/processed",
    results_dir: str = "data/results",
    epochs: int = 30,
    num_classes: int = 2,
) -> Dict:
    """
    Train a single centralized model on ALL organizations' data combined.
    This is the baseline to compare federated approach against.
    """
    processed_dir = Path(processed_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Combine all org data
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    test_X, test_y = None, None

    org_dirs = [d for d in processed_dir.iterdir() if d.is_dir() and d.name.startswith(("Supplier", "Manufacturer", "Logistics", "OEM"))]

    for org_dir in org_dirs:
        all_train_X.append(np.load(org_dir / "train_X.npy"))
        all_train_y.append(np.load(org_dir / "train_y.npy"))
        all_val_X.append(np.load(org_dir / "val_X.npy"))
        all_val_y.append(np.load(org_dir / "val_y.npy"))
        if test_X is None:
            test_X = np.load(org_dir / "test_X.npy")
            test_y = np.load(org_dir / "test_y.npy")

    train_X = np.concatenate(all_train_X)
    train_y = np.concatenate(all_train_y)
    val_X = np.concatenate(all_val_X)
    val_y = np.concatenate(all_val_y)

    input_dim = train_X.shape[1]
    logger.info("Centralized training: %d train, %d val, %d test | %d features",
                 len(train_X), len(val_X), len(test_X), input_dim)

    trainer = LocalTrainer(
        org_name="Centralized_Baseline",
        input_dim=input_dim,
        num_classes=num_classes,
    )
    train_result = trainer.train_standalone(train_X, train_y, val_X, val_y, epochs=epochs)
    eval_result = trainer.evaluate_full(test_X, test_y)

    # Save
    trainer.save_model(str(results_dir / "centralized_model.pt"))
    results = {**train_result, **eval_result}

    results_path = results_dir / "centralized_baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("CENTRALIZED BASELINE RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {results.get('accuracy', 0):.4f}")
    print(f"Precision: {results.get('precision', 0):.4f}")
    print(f"Recall:    {results.get('recall', 0):.4f}")
    print(f"F1-Score:  {results.get('f1_score', 0):.4f}")
    print(f"ROC-AUC:   {results.get('roc_auc', 0):.4f}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Local IDS Trainer")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--org", type=str, default=None, help="Train specific organization only")
    parser.add_argument("--centralized", action="store_true", help="Train centralized baseline")
    args = parser.parse_args()

    if args.centralized:
        train_centralized_baseline(args.processed_dir, args.results_dir, args.epochs)
    else:
        train_all_organizations(args.processed_dir, args.results_dir, args.epochs)


if __name__ == "__main__":
    main()
