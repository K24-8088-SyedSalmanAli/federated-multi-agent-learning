"""
Federated Learning Defense Mechanisms
=====================================
Implements Defense Layer 2 from the proposal:

1. Gradient Norm Clipping — Clips gradient L2 norm at 1.5x median norm
2. Cosine Similarity Check — Flags updates with direction divergence below -0.3

These defenses work alongside the Coordinate-wise Median aggregation 
(Defense Layer 1) to protect against model poisoning attacks.

Usage:
    from src.federated.defenses import GradientDefense
    defense = GradientDefense(clip_factor=1.5, cosine_threshold=-0.3)
    clean_params, flags = defense.filter(client_params, global_params)
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class GradientDefense:
    """
    Combined gradient defense mechanism for federated learning.
    
    Defense Layer 2 (from proposal Section 6.9):
    - L2 Norm Clipping: Clips at clip_factor × median_norm
    - Cosine Similarity: Flags updates diverging from global direction
    
    Args:
        clip_factor: Multiplier for median norm clipping (default: 1.5)
        cosine_threshold: Minimum cosine similarity (default: -0.3)
        enable_clipping: Enable gradient norm clipping
        enable_cosine: Enable cosine similarity check
    """

    def __init__(
        self,
        clip_factor: float = 1.5,
        cosine_threshold: float = -0.3,
        enable_clipping: bool = True,
        enable_cosine: bool = True,
    ):
        self.clip_factor = clip_factor
        self.cosine_threshold = cosine_threshold
        self.enable_clipping = enable_clipping
        self.enable_cosine = enable_cosine
        self.defense_log: List[Dict] = []

    def _flatten_params(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten all parameters into a single 1D tensor."""
        return torch.cat([p.float().flatten() for p in params.values()])

    def _compute_update(
        self,
        client_params: Dict[str, torch.Tensor],
        global_params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the update (delta) from global to client parameters."""
        update = {}
        for key in client_params:
            update[key] = client_params[key].float() - global_params[key].float()
        return update

    def _compute_l2_norm(self, params: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of flattened parameters."""
        flat = self._flatten_params(params)
        return float(torch.norm(flat, p=2))

    def _cosine_similarity(
        self,
        update: Dict[str, torch.Tensor],
        reference: Dict[str, torch.Tensor],
    ) -> float:
        """Compute cosine similarity between two parameter updates."""
        flat_update = self._flatten_params(update)
        flat_ref = self._flatten_params(reference)

        norm_update = torch.norm(flat_update, p=2)
        norm_ref = torch.norm(flat_ref, p=2)

        if norm_update == 0 or norm_ref == 0:
            return 0.0

        similarity = float(torch.dot(flat_update, flat_ref) / (norm_update * norm_ref))
        return similarity

    def _clip_update(
        self,
        update: Dict[str, torch.Tensor],
        max_norm: float,
    ) -> Dict[str, torch.Tensor]:
        """Clip update to have L2 norm ≤ max_norm."""
        current_norm = self._compute_l2_norm(update)
        if current_norm > max_norm:
            scale = max_norm / current_norm
            clipped = {key: val * scale for key, val in update.items()}
            return clipped
        return update

    def filter(
        self,
        client_params_list: List[Dict[str, torch.Tensor]],
        global_params: Dict[str, torch.Tensor],
        org_names: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict]]:
        """
        Apply defense mechanisms to filter client updates.

        Args:
            client_params_list: List of client model parameters
            global_params: Current global model parameters
            org_names: Organization names for logging

        Returns:
            filtered_params: Clean client parameters (after clipping, excluding flagged)
            defense_reports: Per-client defense reports with flags and metrics
        """
        n_clients = len(client_params_list)
        if org_names is None:
            org_names = [f"Client_{i}" for i in range(n_clients)]

        # Step 1: Compute updates (deltas from global model)
        updates = []
        for params in client_params_list:
            update = self._compute_update(params, global_params)
            updates.append(update)

        # Step 2: Compute L2 norms of all updates
        norms = [self._compute_l2_norm(u) for u in updates]
        median_norm = float(np.median(norms))
        max_allowed_norm = self.clip_factor * median_norm

        # Step 3: Compute mean update direction (reference for cosine similarity)
        mean_update = {}
        for key in updates[0]:
            mean_update[key] = torch.stack([u[key] for u in updates]).mean(dim=0)

        # Step 4: Apply defenses per client
        filtered_params = []
        defense_reports = []

        for i in range(n_clients):
            report = {
                "org_name": org_names[i],
                "original_norm": norms[i],
                "median_norm": median_norm,
                "max_allowed_norm": max_allowed_norm,
                "flagged": False,
                "flag_reason": None,
                "clipped": False,
                "accepted": True,
            }

            current_update = updates[i]

            # Defense 2a: Gradient Norm Clipping
            if self.enable_clipping and norms[i] > max_allowed_norm:
                current_update = self._clip_update(current_update, max_allowed_norm)
                report["clipped"] = True
                report["clipped_norm"] = max_allowed_norm
                logger.info("  [CLIPPED] %s: norm %.4f → clipped to %.4f",
                           org_names[i], norms[i], max_allowed_norm)

            # Defense 2b: Cosine Similarity Check
            if self.enable_cosine:
                cos_sim = self._cosine_similarity(current_update, mean_update)
                report["cosine_similarity"] = cos_sim

                if cos_sim < self.cosine_threshold:
                    report["flagged"] = True
                    report["flag_reason"] = f"Low cosine similarity: {cos_sim:.4f} < {self.cosine_threshold}"
                    report["accepted"] = False
                    logger.warning("  [FLAGGED] %s: cosine_sim=%.4f (below threshold %.2f)",
                                  org_names[i], cos_sim, self.cosine_threshold)
                    continue  # Skip this client

            # Reconstruct clean parameters from clipped update
            clean_params = {}
            for key in global_params:
                clean_params[key] = global_params[key].float() + current_update[key]

            filtered_params.append(clean_params)
            defense_reports.append(report)

        # Log flagged clients that were excluded
        flagged_reports = [r for r in defense_reports if r["flagged"]]
        if not flagged_reports:
            # Add reports for flagged clients
            all_reports = defense_reports.copy()
            for i in range(n_clients):
                if not any(r["org_name"] == org_names[i] for r in defense_reports):
                    report = {
                        "org_name": org_names[i],
                        "original_norm": norms[i],
                        "flagged": True,
                        "accepted": False,
                    }
                    all_reports.append(report)
            defense_reports = all_reports

        n_accepted = len(filtered_params)
        n_flagged = n_clients - n_accepted
        logger.info("  Defense: %d/%d accepted, %d flagged, median_norm=%.4f",
                    n_accepted, n_clients, n_flagged, median_norm)

        self.defense_log.append({
            "round_clients": n_clients,
            "accepted": n_accepted,
            "flagged": n_flagged,
            "median_norm": median_norm,
        })

        return filtered_params, defense_reports

    def get_defense_summary(self) -> Dict:
        """Get summary of all defense actions across rounds."""
        if not self.defense_log:
            return {"total_rounds": 0}

        total_flagged = sum(d["flagged"] for d in self.defense_log)
        total_clients = sum(d["round_clients"] for d in self.defense_log)

        return {
            "total_rounds": len(self.defense_log),
            "total_clients_processed": total_clients,
            "total_flagged": total_flagged,
            "flag_rate": total_flagged / total_clients if total_clients > 0 else 0,
            "clip_factor": self.clip_factor,
            "cosine_threshold": self.cosine_threshold,
        }


class ReputationTracker:
    """
    On-chain reputation tracking for federated participants.
    
    Defense Layer 3 (from proposal Section 6.9):
    Each organization has a reputation score that decreases when flagged.
    Organizations below exclusion_threshold are automatically excluded.
    
    Note: This is the Python-side tracker. In Week 11, this will be
    moved to Hyperledger Fabric chaincode for tamper-evident tracking.
    """

    def __init__(
        self,
        org_names: List[str],
        initial_score: int = 100,
        flag_penalty: int = 20,
        exclusion_threshold: int = 40,
    ):
        self.scores = {name: initial_score for name in org_names}
        self.initial_score = initial_score
        self.flag_penalty = flag_penalty
        self.exclusion_threshold = exclusion_threshold
        self.history: List[Dict] = []

    def update(self, defense_reports: List[Dict]):
        """Update reputation scores based on defense reports."""
        for report in defense_reports:
            org = report["org_name"]
            if org not in self.scores:
                continue

            if report.get("flagged", False):
                self.scores[org] = max(0, self.scores[org] - self.flag_penalty)
                logger.info("  [REPUTATION] %s: -%d → score=%d",
                           org, self.flag_penalty, self.scores[org])
            elif report.get("accepted", True):
                # Small reward for good behavior
                self.scores[org] = min(self.initial_score, self.scores[org] + 2)

        self.history.append(dict(self.scores))

    def get_excluded_orgs(self) -> List[str]:
        """Get list of organizations excluded due to low reputation."""
        return [org for org, score in self.scores.items()
                if score <= self.exclusion_threshold]

    def get_active_orgs(self) -> List[str]:
        """Get list of organizations still active."""
        return [org for org, score in self.scores.items()
                if score > self.exclusion_threshold]

    def get_scores(self) -> Dict[str, int]:
        """Get current reputation scores."""
        return dict(self.scores)

    def get_summary(self) -> Dict:
        """Get reputation summary."""
        return {
            "current_scores": self.scores,
            "excluded": self.get_excluded_orgs(),
            "active": self.get_active_orgs(),
            "total_rounds": len(self.history),
        }


if __name__ == "__main__":
    print("Testing Defense Mechanisms...")

    # Create fake params
    global_p = {"w": torch.zeros(10, 5), "b": torch.zeros(10)}

    # Normal clients
    clients = []
    for i in range(4):
        clients.append({"w": torch.randn(10, 5) * 0.1, "b": torch.randn(10) * 0.1})

    # Malicious client (huge norm, opposite direction)
    malicious = {"w": torch.randn(10, 5) * 100, "b": torch.randn(10) * 100}
    clients.append(malicious)

    org_names = ["Org_A", "Org_B", "Org_C", "Org_D", "Malicious_Org"]

    defense = GradientDefense(clip_factor=1.5, cosine_threshold=-0.3)
    filtered, reports = defense.filter(clients, global_p, org_names)

    print(f"\nInput: {len(clients)} clients")
    print(f"Output: {len(filtered)} accepted")
    for r in reports:
        status = "✓ ACCEPTED" if r["accepted"] else "✗ FLAGGED"
        print(f"  {r['org_name']}: {status} (norm={r.get('original_norm', 0):.4f})")

    # Test reputation
    print("\nTesting Reputation Tracker...")
    tracker = ReputationTracker(org_names)
    tracker.update(reports)
    print(f"Scores: {tracker.get_scores()}")
    print(f"Excluded: {tracker.get_excluded_orgs()}")
