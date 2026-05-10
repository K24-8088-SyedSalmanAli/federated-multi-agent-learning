"""
Data Partitioner for Federated Learning
=======================================
Splits CICIDS2017 dataset across N supply chain organizations 
using Dirichlet distribution for non-IID (heterogeneous) data partitioning.

Non-IID is realistic because different supply chain partners 
see different types of attacks based on their role and position.

Usage:
    python -m src.data.data_partitioner
    python -m src.data.data_partitioner --num-orgs 5 --alpha 0.5
    python -m src.data.data_partitioner --strategy iid  # for comparison
"""

import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Supply chain organization roles
ORG_NAMES = [
    "Supplier_RawMaterials",
    "Supplier_Components",
    "Manufacturer_Assembly",
    "Logistics_Provider",
    "OEM_Distributor",
]


class DataPartitioner:
    """
    Partitions dataset across N organizations for federated learning.
    
    Supports:
    - IID: Uniform random split (baseline)
    - Dirichlet: Non-IID split using Dir(alpha) distribution
      - alpha → ∞: approaches IID
      - alpha → 0: each org gets mostly one class
      - alpha = 0.5: moderate heterogeneity (recommended)
    """

    def __init__(
        self,
        num_orgs: int = 5,
        strategy: str = "dirichlet",
        alpha: float = 0.5,
        seed: int = 42,
    ):
        self.num_orgs = num_orgs
        self.strategy = strategy
        self.alpha = alpha
        self.seed = seed
        self.org_names = ORG_NAMES[:num_orgs]
        self.partition_stats: Dict = {}

    def partition(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_split: float = 0.2,
        val_split: float = 0.1,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Partition data into per-organization train/val/test splits.

        Args:
            features: (N, D) feature matrix
            labels: (N,) label array
            test_split: fraction for global test set
            val_split: fraction for per-org validation set

        Returns:
            Dict[org_name] -> {
                "train_X": ..., "train_y": ...,
                "val_X": ..., "val_y": ...,
                "test_X": ..., "test_y": ...  (shared global test)
            }
        """
        np.random.seed(self.seed)
        n_samples = len(features)
        logger.info("Partitioning %d samples across %d organizations (strategy=%s, alpha=%.2f)",
                     n_samples, self.num_orgs, self.strategy, self.alpha)

        # Step 1: Separate global test set (shared across all orgs)
        indices = np.random.permutation(n_samples)
        n_test = int(n_samples * test_split)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        test_X = features[test_indices]
        test_y = labels[test_indices]
        logger.info("  Global test set: %d samples", len(test_X))

        # Step 2: Partition training data across organizations
        if self.strategy == "iid":
            org_indices = self._partition_iid(train_indices)
        elif self.strategy == "dirichlet":
            org_indices = self._partition_dirichlet(train_indices, labels[train_indices])
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Step 3: Create per-org datasets with train/val splits
        org_data = {}
        for i, org_name in enumerate(self.org_names):
            org_idx = org_indices[i]
            org_X = features[org_idx]
            org_y = labels[org_idx]

            # Split into train and validation
            n_val = int(len(org_idx) * val_split)
            perm = np.random.permutation(len(org_idx))
            val_perm = perm[:n_val]
            train_perm = perm[n_val:]

            org_data[org_name] = {
                "train_X": org_X[train_perm],
                "train_y": org_y[train_perm],
                "val_X": org_X[val_perm],
                "val_y": org_y[val_perm],
                "test_X": test_X,
                "test_y": test_y,
            }

            # Stats
            train_dist = Counter(org_y[train_perm].tolist())
            logger.info("  %s: %d train, %d val | Distribution: %s",
                         org_name, len(train_perm), len(val_perm), dict(train_dist))

        self._compute_partition_stats(org_data, labels)
        return org_data

    def _partition_iid(self, indices: np.ndarray) -> List[np.ndarray]:
        """IID partition: random uniform split."""
        np.random.shuffle(indices)
        splits = np.array_split(indices, self.num_orgs)
        return splits

    def _partition_dirichlet(
        self, indices: np.ndarray, labels: np.ndarray
    ) -> List[np.ndarray]:
        """
        Non-IID partition using Dirichlet distribution.
        
        For each class, sample a probability vector from Dir(alpha)
        and assign samples to organizations according to those probabilities.
        This naturally creates heterogeneous distributions.
        """
        unique_classes = np.unique(labels)
        org_indices = [[] for _ in range(self.num_orgs)]

        for cls in unique_classes:
            # Get indices for this class
            cls_indices = indices[labels == cls]
            np.random.shuffle(cls_indices)

            # Sample Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * self.num_orgs)

            # Ensure minimum samples per org (at least 1%)
            proportions = np.maximum(proportions, 0.01)
            proportions /= proportions.sum()

            # Split according to proportions
            splits = (proportions * len(cls_indices)).astype(int)
            # Assign remainder to largest partition
            remainder = len(cls_indices) - splits.sum()
            splits[np.argmax(splits)] += remainder

            start = 0
            for org_id in range(self.num_orgs):
                end = start + splits[org_id]
                org_indices[org_id].extend(cls_indices[start:end].tolist())
                start = end

        return [np.array(idx) for idx in org_indices]

    def _compute_partition_stats(self, org_data: Dict, all_labels: np.ndarray):
        """Compute and log partition statistics."""
        stats = {
            "strategy": self.strategy,
            "alpha": self.alpha,
            "num_organizations": self.num_orgs,
            "organizations": {},
        }

        for org_name, data in org_data.items():
            train_dist = Counter(data["train_y"].tolist())
            total = len(data["train_y"])

            # Compute label skewness (higher = more non-IID)
            proportions = np.array([train_dist.get(c, 0) / total for c in sorted(train_dist.keys())])
            entropy = -np.sum(proportions[proportions > 0] * np.log(proportions[proportions > 0]))

            stats["organizations"][org_name] = {
                "train_samples": int(total),
                "val_samples": int(len(data["val_y"])),
                "label_distribution": {str(k): int(v) for k, v in train_dist.items()},
                "entropy": float(entropy),
            }

        self.partition_stats = stats

    def save_partition(self, org_data: Dict, output_dir: str = "data/processed"):
        """Save partitioned data to disk."""
        output_dir = Path(output_dir)

        for org_name, data in org_data.items():
            org_dir = output_dir / org_name
            org_dir.mkdir(parents=True, exist_ok=True)

            np.save(org_dir / "train_X.npy", data["train_X"])
            np.save(org_dir / "train_y.npy", data["train_y"])
            np.save(org_dir / "val_X.npy", data["val_X"])
            np.save(org_dir / "val_y.npy", data["val_y"])
            np.save(org_dir / "test_X.npy", data["test_X"])
            np.save(org_dir / "test_y.npy", data["test_y"])

        # Save stats
        stats_path = output_dir / "partition_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.partition_stats, f, indent=2)

        logger.info("Partition saved to %s", output_dir)

    def load_partition(self, output_dir: str = "data/processed") -> Dict[str, Dict[str, np.ndarray]]:
        """Load partitioned data from disk."""
        output_dir = Path(output_dir)
        org_data = {}

        for org_name in self.org_names:
            org_dir = output_dir / org_name
            if not org_dir.exists():
                raise FileNotFoundError(f"Partition not found for {org_name} at {org_dir}")

            org_data[org_name] = {
                "train_X": np.load(org_dir / "train_X.npy"),
                "train_y": np.load(org_dir / "train_y.npy"),
                "val_X": np.load(org_dir / "val_X.npy"),
                "val_y": np.load(org_dir / "val_y.npy"),
                "test_X": np.load(org_dir / "test_X.npy"),
                "test_y": np.load(org_dir / "test_y.npy"),
            }

        logger.info("Loaded partitions for %d organizations", len(org_data))
        return org_data


def main():
    parser = argparse.ArgumentParser(description="Data Partitioner for Federated Learning")
    parser.add_argument("--num-orgs", type=int, default=5, help="Number of organizations")
    parser.add_argument("--strategy", choices=["iid", "dirichlet"], default="dirichlet")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (lower = more non-IID)")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    # Load processed data
    processed_dir = Path(args.processed_dir)
    features = np.load(processed_dir / "features.npy")
    binary_labels = np.load(processed_dir / "binary_labels.npy")

    logger.info("Loaded %d samples with %d features", len(features), features.shape[1])

    # Partition
    partitioner = DataPartitioner(
        num_orgs=args.num_orgs,
        strategy=args.strategy,
        alpha=args.alpha,
    )
    org_data = partitioner.partition(features, binary_labels)
    partitioner.save_partition(org_data, output_dir=args.processed_dir)


if __name__ == "__main__":
    main()
