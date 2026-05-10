"""
Federated Learning Server
=========================
Central server that orchestrates the federated learning process:
1. Initialize global model
2. Distribute parameters to clients
3. Collect updated parameters
4. Apply defenses (gradient clipping, cosine similarity)
5. Aggregate using Byzantine-robust method
6. Evaluate global model
7. Repeat for R rounds

Usage:
    from src.federated.fed_server import FedServer
    server = FedServer(input_dim=65, num_orgs=5)
    results = server.run(org_data, num_rounds=10)
"""

import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.cnn_lstm import CNNLSTM
from src.federated.fed_client import FedClient
from src.federated.aggregators import build_aggregator, BaseAggregator
from src.federated.defenses import GradientDefense, ReputationTracker
from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FedServer:
    """
    Federated Learning Server.

    Manages the entire federated training process including:
    - Global model management
    - Client coordination
    - Defense mechanisms (gradient clipping + cosine similarity)
    - Aggregation (FedAvg or Coordinate-wise Median)
    - Reputation tracking
    - Per-round evaluation

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        aggregation: Aggregation method ("fedavg" or "coordinate_median")
        enable_defenses: Enable gradient defenses
        clip_factor: Gradient norm clipping factor
        cosine_threshold: Cosine similarity threshold
    """

    def __init__(
        self,
        input_dim: int = 65,
        num_classes: int = 2,
        aggregation: str = "coordinate_median",
        enable_defenses: bool = True,
        clip_factor: float = 1.5,
        cosine_threshold: float = -0.3,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Global model
        self.global_model = CNNLSTM(input_dim=input_dim, num_classes=num_classes)
        self.global_params = self._get_named_params()

        # Aggregator
        self.aggregator = build_aggregator(aggregation)
        logger.info("Aggregation method: %s", self.aggregator.name)

        # Defenses
        self.enable_defenses = enable_defenses
        self.defense = GradientDefense(
            clip_factor=clip_factor,
            cosine_threshold=cosine_threshold,
        ) if enable_defenses else None

        # Training history
        self.round_history: List[Dict] = []

    def _get_named_params(self) -> Dict[str, torch.Tensor]:
        """Get global model parameters (full state_dict including buffers)."""
        return {name: param.clone() for name, param in self.global_model.state_dict().items()}

    def _set_global_params(self, params: Dict[str, torch.Tensor]):
        """Update global model with aggregated parameters."""
        state_dict = self.global_model.state_dict()
        for name in params:
            if name in state_dict:
                state_dict[name] = params[name]
        self.global_model.load_state_dict(state_dict)
        self.global_params = self._get_named_params()

    def _evaluate_global(self, test_X: np.ndarray, test_y: np.ndarray) -> Dict:
        """Evaluate global model on shared test set."""
        self.global_model.eval()
        X_tensor = torch.FloatTensor(test_X)
        y_tensor = torch.LongTensor(test_y)

        all_preds = []
        all_probs = []
        batch_size = 256

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i + batch_size]
                outputs = self.global_model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.numpy())
                all_probs.extend(probs.numpy())

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        metrics = compute_all_metrics(test_y, all_preds, all_probs)
        return metrics

    def run(
        self,
        org_data: Dict[str, Dict[str, np.ndarray]],
        num_rounds: int = 10,
        local_epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> Dict:
        """
        Run the complete federated learning process.

        Args:
            org_data: Dict[org_name] -> {train_X, train_y, val_X, val_y, test_X, test_y}
            num_rounds: Number of federated communication rounds
            local_epochs: Local training epochs per round
            learning_rate: Client learning rate

        Returns:
            Complete training results with per-round metrics
        """
        org_names = list(org_data.keys())
        n_orgs = len(org_names)

        # Get shared test set from first org
        test_X = org_data[org_names[0]]["test_X"]
        test_y = org_data[org_names[0]]["test_y"]

        # Initialize clients
        clients = {}
        for org_name in org_names:
            clients[org_name] = FedClient(
                org_name=org_name,
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                learning_rate=learning_rate,
            )

        # Initialize reputation tracker
        reputation = ReputationTracker(org_names)

        # Data sizes for weighted aggregation
        data_sizes = {org: len(org_data[org]["train_X"]) for org in org_names}
        total_data = sum(data_sizes.values())
        weights = [data_sizes[org] / total_data for org in org_names]

        logger.info("=" * 60)
        logger.info("FEDERATED LEARNING — %s", self.aggregator.name)
        logger.info("Organizations: %d | Rounds: %d | Local epochs: %d",
                    n_orgs, num_rounds, local_epochs)
        logger.info("Data sizes: %s", {org: data_sizes[org] for org in org_names})
        logger.info("=" * 60)

        # Evaluate initial global model (before training)
        initial_metrics = self._evaluate_global(test_X, test_y)
        logger.info("Initial global accuracy: %.4f", initial_metrics["accuracy"])

        start_time = time.time()

        for round_num in range(1, num_rounds + 1):
            round_start = time.time()
            logger.info("\n--- Round %d/%d ---", round_num, num_rounds)

            # Get active organizations (not excluded by reputation)
            active_orgs = reputation.get_active_orgs()
            excluded_orgs = reputation.get_excluded_orgs()
            if excluded_orgs:
                logger.info("  Excluded orgs: %s", excluded_orgs)

            if len(active_orgs) < 2:
                logger.warning("  Too few active orgs (%d). Stopping.", len(active_orgs))
                break

            # Step 1: Distribute global params & train locally
            client_params_list = []
            active_weights = []
            active_names = []

            for org_name in active_orgs:
                client = clients[org_name]
                updated_params = client.train_round(
                    global_params=self.global_params,
                    train_X=org_data[org_name]["train_X"],
                    train_y=org_data[org_name]["train_y"],
                    local_epochs=local_epochs,
                )
                client_params_list.append(updated_params)
                active_weights.append(data_sizes[org_name] / total_data)
                active_names.append(org_name)

            # Step 2: Apply defenses
            defense_reports = []
            if self.enable_defenses and self.defense:
                filtered_params, defense_reports = self.defense.filter(
                    client_params_list, self.global_params, active_names
                )
                reputation.update(defense_reports)

                if len(filtered_params) == 0:
                    logger.warning("  All clients filtered! Using previous global model.")
                    continue

                # Recalculate weights for accepted clients
                accepted_names = [r["org_name"] for r in defense_reports if r.get("accepted", True)]
                accepted_weights = [data_sizes[n] / total_data for n in accepted_names if n in data_sizes]
            else:
                filtered_params = client_params_list
                accepted_weights = active_weights

            # Step 3: Aggregate
            aggregated_params = self.aggregator.aggregate(
                filtered_params,
                weights=accepted_weights if len(accepted_weights) == len(filtered_params) else None,
            )

            # Step 4: Update global model
            self._set_global_params(aggregated_params)

            # Step 5: Evaluate global model
            round_metrics = self._evaluate_global(test_X, test_y)
            round_time = time.time() - round_start

            round_result = {
                "round": round_num,
                "accuracy": round_metrics["accuracy"],
                "precision": round_metrics["precision"],
                "recall": round_metrics["recall"],
                "f1_score": round_metrics["f1_score"],
                "roc_auc": round_metrics.get("roc_auc", 0),
                "active_orgs": len(active_orgs),
                "accepted_clients": len(filtered_params),
                "round_time_seconds": round(round_time, 2),
                "reputation_scores": reputation.get_scores(),
            }
            self.round_history.append(round_result)

            logger.info("  Global Acc: %.4f | F1: %.4f | Accepted: %d/%d | Time: %.1fs",
                       round_metrics["accuracy"], round_metrics["f1_score"],
                       len(filtered_params), len(active_orgs), round_time)

        total_time = time.time() - start_time

        # Final results
        final_metrics = self._evaluate_global(test_X, test_y)
        results = {
            "aggregation": self.aggregator.name,
            "defenses_enabled": self.enable_defenses,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "num_organizations": n_orgs,
            "total_time_seconds": round(total_time, 2),
            "initial_accuracy": initial_metrics["accuracy"],
            "final_accuracy": final_metrics["accuracy"],
            "final_precision": final_metrics["precision"],
            "final_recall": final_metrics["recall"],
            "final_f1": final_metrics["f1_score"],
            "final_roc_auc": final_metrics.get("roc_auc", 0),
            "round_history": self.round_history,
            "reputation_summary": reputation.get_summary(),
            "defense_summary": self.defense.get_defense_summary() if self.defense else None,
        }

        return results


if __name__ == "__main__":
    print("Testing FedServer...")

    # Create fake data for 3 orgs
    np.random.seed(42)
    org_data = {}
    test_X = np.random.rand(200, 65).astype(np.float32)
    test_y = np.random.randint(0, 2, 200).astype(np.int64)

    for name in ["Org_A", "Org_B", "Org_C"]:
        org_data[name] = {
            "train_X": np.random.rand(300, 65).astype(np.float32),
            "train_y": np.random.randint(0, 2, 300).astype(np.int64),
            "val_X": np.random.rand(50, 65).astype(np.float32),
            "val_y": np.random.randint(0, 2, 50).astype(np.int64),
            "test_X": test_X,
            "test_y": test_y,
        }

    server = FedServer(input_dim=65, aggregation="coordinate_median")
    results = server.run(org_data, num_rounds=3, local_epochs=2)

    print(f"\nFinal accuracy: {results['final_accuracy']:.4f}")
    print(f"Total time: {results['total_time_seconds']:.1f}s")
    print("✓ FedServer working!")
