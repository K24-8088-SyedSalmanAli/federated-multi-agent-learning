"""
Federated Learning Client
=========================
Each supply chain organization runs a FedClient that:
1. Receives global model parameters from the server
2. Trains locally on its own data for a few epochs
3. Returns updated model parameters to the server

Usage:
    from src.federated.fed_client import FedClient
    client = FedClient(org_name="Supplier_A", input_dim=65)
    updated_params = client.train_round(global_params, train_X, train_y, epochs=5)
"""

import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.cnn_lstm import CNNLSTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FedClient:
    """
    Federated Learning Client for a single organization.

    In each federated round:
    1. Load global model parameters
    2. Train on local data for `local_epochs` 
    3. Return updated parameters

    Args:
        org_name: Organization identifier
        input_dim: Number of input features
        num_classes: Number of output classes
        learning_rate: Local training learning rate
        batch_size: Training batch size
        device: torch device
    """

    def __init__(
        self,
        org_name: str,
        input_dim: int,
        num_classes: int = 2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        device: str = "auto",
    ):
        self.org_name = org_name
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model
        self.model = CNNLSTM(input_dim=input_dim, num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Training stats per round
        self.round_stats: list = []

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        """Create DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def load_global_params(self, global_params: Dict[str, torch.Tensor]):
        """Load global model parameters at the start of a round."""
        self.model.load_state_dict(global_params, strict=False)
        # Also load buffers (BatchNorm running_mean/var) if present
        state = self.model.state_dict()
        for key in global_params:
            if key in state:
                state[key] = global_params[key]
        self.model.load_state_dict(state)

    def train_round(
        self,
        global_params: Dict[str, torch.Tensor],
        train_X: np.ndarray,
        train_y: np.ndarray,
        local_epochs: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one federated round:
        1. Load global params
        2. Train locally
        3. Return updated params

        Args:
            global_params: Global model state_dict
            train_X: Local training features
            train_y: Local training labels
            local_epochs: Number of local training epochs

        Returns:
            Updated model parameters (state_dict)
        """
        # Load global model
        self.load_global_params(global_params)

        # Create optimizer fresh each round (reset momentum)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # Train
        train_loader = self._create_dataloader(train_X, train_y)
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for epoch in range(local_epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(y_batch)
                _, predicted = outputs.max(1)
                epoch_correct += predicted.eq(y_batch).sum().item()
                epoch_total += len(y_batch)

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_total

        # Record stats
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        self.round_stats.append({
            "org_name": self.org_name,
            "local_epochs": local_epochs,
            "train_loss": round(avg_loss, 4),
            "train_acc": round(avg_acc, 4),
            "train_samples": len(train_X),
        })

        # Return updated parameters (full state_dict including BatchNorm buffers)
        return {name: param.clone().cpu() for name, param in self.model.state_dict().items()}

    def evaluate(
        self,
        test_X: np.ndarray,
        test_y: np.ndarray,
    ) -> Dict:
        """Evaluate model on test data."""
        self.model.eval()
        test_loader = self._create_dataloader(test_X, test_y, shuffle=False)

        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(y_batch).sum().item()
                total_samples += len(y_batch)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        return {
            "org_name": self.org_name,
            "accuracy": round(accuracy, 4),
            "total_samples": total_samples,
        }

    def get_data_size(self, train_X: np.ndarray) -> int:
        """Return local dataset size (for weighted aggregation)."""
        return len(train_X)


if __name__ == "__main__":
    print("Testing FedClient...")

    # Create fake data
    np.random.seed(42)
    X = np.random.rand(500, 65).astype(np.float32)
    y = np.random.randint(0, 2, 500).astype(np.int64)

    # Create client
    client = FedClient(org_name="TestOrg", input_dim=65)

    # Simulate one round
    global_params = {name: param.data.clone() for name, param in client.model.named_parameters()}
    updated_params = client.train_round(global_params, X, y, local_epochs=2)

    print(f"  Round stats: {client.round_stats[-1]}")
    print(f"  Parameters returned: {len(updated_params)} tensors")
    print("✓ FedClient working!")
