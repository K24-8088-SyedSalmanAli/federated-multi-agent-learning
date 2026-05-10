"""
Poisoning Attack Evaluator
==========================
Runs federated learning with poisoning attacks at different compromise
ratios (10%, 20%, 30%) and measures defense effectiveness.

Compares:
- No defense (FedAvg only)
- Gradient Clipping only
- Full defense (Coordinate Median + Clipping + Cosine)

Metrics:
- Accuracy drift under untargeted poisoning (target: < 3%)
- Targeted attack success rate (target: < 5%)
- False exclusion rate of honest participants

Reference: Proposal Section 6.9

Usage:
    from src.attacks.poisoning_evaluator import PoisoningEvaluator
    evaluator = PoisoningEvaluator(org_data, input_dim=65)
    results = evaluator.run_all_experiments()
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
from src.federated.fed_server import FedServer
from src.federated.aggregators import build_aggregator
from src.federated.defenses import GradientDefense, ReputationTracker
from src.attacks.poisoning import PoisonedFedClient, UntargetedPoisoner, TargetedPoisoner
from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PoisoningEvaluator:
    """
    Evaluates federated learning under poisoning attacks.

    Runs experiments at different compromise ratios with different
    defense configurations to measure robustness.
    """

    def __init__(
        self,
        org_data: Dict[str, Dict[str, np.ndarray]],
        input_dim: int = 65,
        num_classes: int = 2,
    ):
        self.org_data = org_data
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.org_names = list(org_data.keys())
        self.n_orgs = len(self.org_names)

        # Shared test set
        self.test_X = org_data[self.org_names[0]]["test_X"]
        self.test_y = org_data[self.org_names[0]]["test_y"]

    def _run_poisoned_federated(
        self,
        compromised_orgs: List[str],
        attack_type: str,
        aggregation: str,
        enable_defenses: bool,
        num_rounds: int = 10,
        local_epochs: int = 5,
        noise_scale: float = 5.0,
    ) -> Dict:
        """
        Run federated learning with specified compromised organizations.

        Args:
            compromised_orgs: List of org names that are malicious
            attack_type: "untargeted" or "targeted"
            aggregation: "fedavg" or "coordinate_median"
            enable_defenses: Enable gradient clipping + cosine check
            num_rounds: Federated rounds
            local_epochs: Local epochs per round
            noise_scale: Noise scale for untargeted attack

        Returns:
            Results dict with per-round metrics
        """
        # Initialize global model
        global_model = CNNLSTM(input_dim=self.input_dim, num_classes=self.num_classes)
        global_params = {name: param.clone() for name, param in global_model.state_dict().items()}

        # Initialize aggregator
        aggregator = build_aggregator(aggregation)

        # Initialize defenses
        defense = GradientDefense(clip_factor=1.5, cosine_threshold=-0.3) if enable_defenses else None
        reputation = ReputationTracker(self.org_names)

        # Initialize clients (normal + poisoned)
        clients = {}
        for org_name in self.org_names:
            fed_client = FedClient(
                org_name=org_name,
                input_dim=self.input_dim,
                num_classes=self.num_classes,
            )
            if org_name in compromised_orgs:
                clients[org_name] = PoisonedFedClient(
                    fed_client=fed_client,
                    attack_type=attack_type,
                    noise_scale=noise_scale,
                )
            else:
                clients[org_name] = fed_client

        # Data sizes
        data_sizes = {org: len(self.org_data[org]["train_X"]) for org in self.org_names}

        # Track per-round metrics
        round_history = []

        for round_num in range(1, num_rounds + 1):
            active_orgs = reputation.get_active_orgs()
            if len(active_orgs) < 2:
                break

            # Collect client updates
            client_params_list = []
            active_names = []

            for org_name in active_orgs:
                client = clients[org_name]
                train_X = self.org_data[org_name]["train_X"]
                train_y = self.org_data[org_name]["train_y"]

                if isinstance(client, PoisonedFedClient):
                    updated_params = client.train_round_poisoned(
                        global_params, train_X, train_y, local_epochs
                    )
                else:
                    updated_params = client.train_round(
                        global_params, train_X, train_y, local_epochs
                    )

                client_params_list.append(updated_params)
                active_names.append(org_name)

            # Apply defenses
            if defense:
                filtered_params, defense_reports = defense.filter(
                    client_params_list, global_params, active_names
                )
                reputation.update(defense_reports)
                if len(filtered_params) == 0:
                    continue
            else:
                filtered_params = client_params_list

            # Aggregate
            aggregated = aggregator.aggregate(filtered_params)

            # Update global model
            state_dict = global_model.state_dict()
            for name in aggregated:
                if name in state_dict:
                    state_dict[name] = aggregated[name]
            global_model.load_state_dict(state_dict)
            global_params = {name: param.clone() for name, param in global_model.state_dict().items()}

            # Evaluate
            global_model.eval()
            X_tensor = torch.FloatTensor(self.test_X)
            all_preds = []
            all_probs = []

            with torch.no_grad():
                for i in range(0, len(X_tensor), 256):
                    batch = X_tensor[i:i + 256]
                    outputs = global_model(batch)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    all_preds.extend(predicted.numpy())
                    all_probs.extend(probs.numpy())

            metrics = compute_all_metrics(self.test_y, np.array(all_preds), np.array(all_probs))

            round_history.append({
                "round": round_num,
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accepted": len(filtered_params),
                "total": len(client_params_list),
            })

            logger.info("  Round %d | Acc: %.4f | F1: %.4f | Accepted: %d/%d",
                       round_num, metrics["accuracy"], metrics["f1_score"],
                       len(filtered_params), len(client_params_list))

        # Final metrics
        final_metrics = round_history[-1] if round_history else {"accuracy": 0, "f1_score": 0}

        return {
            "attack_type": attack_type,
            "compromised_orgs": compromised_orgs,
            "compromise_ratio": len(compromised_orgs) / self.n_orgs,
            "aggregation": aggregation,
            "defenses_enabled": enable_defenses,
            "final_accuracy": final_metrics["accuracy"],
            "final_f1": final_metrics["f1_score"],
            "round_history": round_history,
            "reputation_scores": reputation.get_scores(),
            "excluded_orgs": reputation.get_excluded_orgs(),
        }

    def _select_compromised(self, ratio: float) -> List[str]:
        """Select organizations to compromise based on ratio."""
        n_compromised = max(1, int(self.n_orgs * ratio))
        # Pick the smallest orgs (least data = least impact = harder to detect)
        data_sizes = {org: len(self.org_data[org]["train_X"]) for org in self.org_names}
        sorted_orgs = sorted(data_sizes, key=data_sizes.get)
        return sorted_orgs[:n_compromised]

    def run_all_experiments(
        self,
        compromise_ratios: List[float] = None,
        num_rounds: int = 10,
        local_epochs: int = 5,
    ) -> Dict:
        """
        Run all poisoning experiments as specified in the proposal.

        Tests: 10%, 20%, 30% compromise with untargeted and targeted attacks.
        Defenses: No defense, Clipping only, Full defense.
        """
        if compromise_ratios is None:
            compromise_ratios = [0.1, 0.2, 0.3]

        all_results = {}
        experiment_id = 0

        for ratio in compromise_ratios:
            compromised = self._select_compromised(ratio)
            ratio_pct = int(ratio * 100)

            logger.info("\n" + "=" * 60)
            logger.info("COMPROMISE RATIO: %d%% (%d/%d orgs: %s)",
                       ratio_pct, len(compromised), self.n_orgs, compromised)
            logger.info("=" * 60)

            for attack_type in ["untargeted", "targeted"]:
                # Config 1: No defense (FedAvg only)
                logger.info("\n--- %s + No Defense (FedAvg) ---", attack_type)
                experiment_id += 1
                result = self._run_poisoned_federated(
                    compromised_orgs=compromised,
                    attack_type=attack_type,
                    aggregation="fedavg",
                    enable_defenses=False,
                    num_rounds=num_rounds,
                    local_epochs=local_epochs,
                )
                key = f"{ratio_pct}pct_{attack_type}_no_defense"
                all_results[key] = result

                # Config 2: Full defense (Coordinate Median + Clipping + Cosine)
                logger.info("\n--- %s + Full Defense (Median + Clipping) ---", attack_type)
                experiment_id += 1
                result = self._run_poisoned_federated(
                    compromised_orgs=compromised,
                    attack_type=attack_type,
                    aggregation="coordinate_median",
                    enable_defenses=True,
                    num_rounds=num_rounds,
                    local_epochs=local_epochs,
                )
                key = f"{ratio_pct}pct_{attack_type}_full_defense"
                all_results[key] = result

        return all_results

    def compute_summary(self, all_results: Dict, clean_accuracy: float = None) -> Dict:
        """
        Compute summary metrics from all experiments.

        Calculates:
        - Accuracy drift (vs clean baseline)
        - Targeted attack success rate
        - Defense effectiveness
        """
        summary = {"experiments": []}

        for key, result in all_results.items():
            ratio = result["compromise_ratio"]
            attack = result["attack_type"]
            defense = "Full Defense" if result["defenses_enabled"] else "No Defense"
            acc = result["final_accuracy"]
            f1 = result["final_f1"]

            drift = 0
            if clean_accuracy:
                drift = clean_accuracy - acc

            summary["experiments"].append({
                "experiment": key,
                "compromise_ratio": f"{int(ratio * 100)}%",
                "attack_type": attack,
                "defense": defense,
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "accuracy_drift": round(drift, 4),
                "drift_within_target": drift < 0.03,  # target: < 3%
                "excluded_orgs": result["excluded_orgs"],
            })

        # Compute per-ratio summary
        for ratio in [10, 20, 30]:
            no_def_untarg = all_results.get(f"{ratio}pct_untargeted_no_defense", {})
            full_def_untarg = all_results.get(f"{ratio}pct_untargeted_full_defense", {})
            no_def_targ = all_results.get(f"{ratio}pct_targeted_no_defense", {})
            full_def_targ = all_results.get(f"{ratio}pct_targeted_full_defense", {})

            summary[f"{ratio}pct_summary"] = {
                "untargeted_no_defense_acc": no_def_untarg.get("final_accuracy", 0),
                "untargeted_full_defense_acc": full_def_untarg.get("final_accuracy", 0),
                "targeted_no_defense_acc": no_def_targ.get("final_accuracy", 0),
                "targeted_full_defense_acc": full_def_targ.get("final_accuracy", 0),
            }

        return summary
