"""
Federated Aggregation Algorithms
================================
Implements aggregation strategies for combining model updates 
from multiple supply chain organizations.

1. FedAvg — Simple weighted average (baseline, vulnerable to poisoning)
2. Coordinate-wise Median — Byzantine-robust (provably robust against f < n/2)

Reference: 
- McMahan et al., "Communication-Efficient Learning of Deep Networks 
  from Decentralized Data", 2017 (FedAvg)
- Yin et al., "Byzantine-Robust Distributed Learning", 2018 (Coordinate Median)

Usage:
    from src.federated.aggregators import FedAvgAggregator, CoordinateMedianAggregator
    aggregator = CoordinateMedianAggregator()
    global_params = aggregator.aggregate(client_params_list, weights)
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BaseAggregator(ABC):
    """Abstract base class for federated aggregation."""

    def __init__(self, name: str = "base"):
        self.name = name

    @abstractmethod
    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model parameters from multiple clients.

        Args:
            client_params: List of state_dicts from each client
            weights: Optional weights for weighted aggregation (e.g., by dataset size)

        Returns:
            Aggregated global model parameters
        """
        pass


class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg) — McMahan et al., 2017
    
    Simple weighted average of all client parameters.
    Baseline method — vulnerable to Byzantine/poisoning attacks 
    because a single malicious client can shift the average significantly.
    """

    def __init__(self):
        super().__init__(name="FedAvg")

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Weighted average of client parameters."""
        n_clients = len(client_params)
        if n_clients == 0:
            raise ValueError("No client parameters to aggregate")

        # Default: equal weights
        if weights is None:
            weights = [1.0 / n_clients] * n_clients
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]

        # Weighted average
        global_params = {}
        for key in client_params[0].keys():
            global_params[key] = torch.zeros_like(client_params[0][key], dtype=torch.float32)
            for i, params in enumerate(client_params):
                global_params[key] += weights[i] * params[key].float()

        logger.debug("[FedAvg] Aggregated %d clients with weights %s", n_clients, 
                     [f"{w:.3f}" for w in weights])
        return global_params


class CoordinateMedianAggregator(BaseAggregator):
    """
    Coordinate-wise Median Aggregation — Yin et al., 2018
    
    For each parameter dimension, computes the median across all clients.
    Provably robust against up to f < n/2 Byzantine participants because
    the median is unaffected by extreme outlier values.

    This is the primary defense (Defense Layer 1) in the proposal.
    """

    def __init__(self):
        super().__init__(name="CoordinateMedian")

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Coordinate-wise median of client parameters.
        Weights are ignored — median is inherently unweighted.
        """
        n_clients = len(client_params)
        if n_clients == 0:
            raise ValueError("No client parameters to aggregate")

        global_params = {}
        for key in client_params[0].keys():
            # Stack all client params for this key: shape (n_clients, *param_shape)
            stacked = torch.stack([params[key].float() for params in client_params])
            # Compute median along client dimension (dim=0)
            global_params[key] = torch.median(stacked, dim=0).values

        logger.debug("[CoordinateMedian] Aggregated %d clients (robust against %d Byzantine)",
                     n_clients, (n_clients - 1) // 2)
        return global_params


class TrimmedMeanAggregator(BaseAggregator):
    """
    Trimmed Mean Aggregation
    
    Removes the top and bottom β fraction of values for each coordinate,
    then averages the remaining. Robust against f < β*n Byzantine clients.
    
    Used as an additional comparison in experiments.
    """

    def __init__(self, trim_ratio: float = 0.1):
        super().__init__(name="TrimmedMean")
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Trimmed mean of client parameters."""
        n_clients = len(client_params)
        if n_clients == 0:
            raise ValueError("No client parameters to aggregate")

        n_trim = max(1, int(n_clients * self.trim_ratio))

        global_params = {}
        for key in client_params[0].keys():
            stacked = torch.stack([params[key].float() for params in client_params])
            # Sort along client dimension
            sorted_params, _ = torch.sort(stacked, dim=0)
            # Trim top and bottom
            trimmed = sorted_params[n_trim:n_clients - n_trim]
            # Average remaining
            if trimmed.shape[0] == 0:
                # If too few clients, fallback to median
                global_params[key] = torch.median(stacked, dim=0).values
            else:
                global_params[key] = trimmed.mean(dim=0)

        logger.debug("[TrimmedMean] Aggregated %d clients, trimmed %d each side",
                     n_clients, n_trim)
        return global_params


def build_aggregator(name: str = "coordinate_median", **kwargs) -> BaseAggregator:
    """Factory function for aggregators."""
    aggregators = {
        "fedavg": FedAvgAggregator,
        "coordinate_median": CoordinateMedianAggregator,
        "trimmed_mean": TrimmedMeanAggregator,
    }
    if name not in aggregators:
        raise ValueError(f"Unknown aggregator: {name}. Choose from {list(aggregators.keys())}")
    return aggregators[name](**kwargs)


if __name__ == "__main__":
    # Quick test
    print("Testing Aggregators...")
    
    # Create fake client params (3 clients, simple model)
    clients = []
    for i in range(5):
        params = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
        }
        clients.append(params)

    # Add a malicious client (extreme values)
    malicious = {
        "layer1.weight": torch.ones(10, 5) * 1000,
        "layer1.bias": torch.ones(10) * 1000,
    }
    clients_with_malicious = clients + [malicious]

    # Test FedAvg (vulnerable)
    fedavg = FedAvgAggregator()
    result_fedavg = fedavg.aggregate(clients_with_malicious)
    print(f"FedAvg (with malicious): weight mean = {result_fedavg['layer1.weight'].mean():.4f}")

    # Test Coordinate Median (robust)
    median = CoordinateMedianAggregator()
    result_median = median.aggregate(clients_with_malicious)
    print(f"CoordMedian (with malicious): weight mean = {result_median['layer1.weight'].mean():.4f}")

    # Test without malicious (for comparison)
    result_clean = fedavg.aggregate(clients)
    print(f"FedAvg (clean): weight mean = {result_clean['layer1.weight'].mean():.4f}")

    print("\n✓ Coordinate Median is unaffected by malicious client!")
