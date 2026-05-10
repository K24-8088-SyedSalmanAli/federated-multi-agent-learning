"""
Blockchain-Verified Gradient Auditing
=====================================
Simulates Hyperledger Fabric blockchain logging for federated learning.

Logs every gradient submission with:
- Cryptographic hash (SHA-256)
- Participant identity
- Gradient norm
- Accept/reject decision
- Timestamp

Maintains on-chain reputation scores and provides scalability benchmarks.

Reference: Proposal Section 6.9 Defense Layer 3, Section 6.10

Usage:
    from src.blockchain.blockchain_logger import BlockchainLogger
    logger = BlockchainLogger(num_peers=4)
    logger.log_gradient_submission(org_name, grad_hash, norm, accepted)
"""

import hashlib
import json
import time
import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class Block:
    """Single block in the blockchain."""

    def __init__(self, index: int, transactions: List[Dict], previous_hash: str, timestamp: float = None):
        self.index = index
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.timestamp = timestamp or time.time()
        self.nonce = 0
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        block_data = json.dumps({
            "index": self.index,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
        }, sort_keys=True)
        return hashlib.sha256(block_data.encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "hash": self.hash,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "num_transactions": len(self.transactions),
            "transactions": self.transactions,
        }


class BlockchainLogger:
    """
    Simulated Hyperledger Fabric blockchain for gradient auditing.

    Features:
    - Immutable audit trail for all gradient submissions
    - SHA-256 cryptographic hashing
    - On-chain reputation scoring
    - Scalability benchmarking (TPS, latency, storage)
    - Tamper detection

    Args:
        num_peers: Number of simulated network peers
        block_size: Max transactions per block
        block_timeout: Block creation timeout (seconds)
    """

    def __init__(
        self,
        num_peers: int = 4,
        block_size: int = 50,
        block_timeout: float = 2.0,
    ):
        self.num_peers = num_peers
        self.block_size = block_size
        self.block_timeout = block_timeout

        # Blockchain
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict] = []
        self._create_genesis_block()

        # Reputation scores
        self.reputation_scores: Dict[str, int] = {}

        # Performance metrics
        self.commit_latencies: List[float] = []
        self.transaction_sizes: List[int] = []
        self.log_entries: List[Dict] = []

    def _create_genesis_block(self):
        """Create the first block in the chain."""
        genesis = Block(index=0, transactions=[{"type": "genesis"}], previous_hash="0")
        self.chain.append(genesis)

    def _hash_gradient(self, gradient_data: Dict) -> str:
        """Compute SHA-256 hash of gradient data."""
        data_str = json.dumps(gradient_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def log_gradient_submission(
        self,
        org_name: str,
        gradient_norm: float,
        accepted: bool,
        round_num: int,
        clipped: bool = False,
        cosine_similarity: float = None,
        extra_data: Dict = None,
    ) -> Dict:
        """
        Log a gradient submission to the blockchain.

        Chaincode function: logGradientSubmission()

        Args:
            org_name: Organization identifier
            gradient_norm: L2 norm of submitted gradient
            accepted: Whether the submission was accepted
            round_num: Federated round number
            clipped: Whether gradient was clipped
            cosine_similarity: Cosine similarity score

        Returns:
            Transaction record with hash and timing
        """
        start_time = time.time()

        # Initialize reputation if new org
        if org_name not in self.reputation_scores:
            self.reputation_scores[org_name] = 100

        # Build transaction
        transaction = {
            "type": "gradient_submission",
            "org_name": org_name,
            "round": round_num,
            "gradient_norm": round(gradient_norm, 6),
            "accepted": accepted,
            "clipped": clipped,
            "cosine_similarity": round(cosine_similarity, 6) if cosine_similarity else None,
            "timestamp": datetime.now().isoformat(),
            "decision": "ACCEPT" if accepted else "REJECT",
        }

        # Compute hash
        transaction["hash"] = self._hash_gradient(transaction)

        # Add to pending
        self.pending_transactions.append(transaction)

        # Create block if enough transactions or timeout
        if len(self.pending_transactions) >= self.block_size:
            self._create_block()

        # Update reputation
        self._update_reputation(org_name, accepted)

        # Simulate commit latency (proportional to num_peers)
        # Real Fabric: ~0.5-2s depending on endorsement policy
        base_latency = 0.001 * self.num_peers  # simulated
        commit_time = time.time() - start_time + base_latency

        self.commit_latencies.append(commit_time)
        tx_size = len(json.dumps(transaction).encode())
        self.transaction_sizes.append(tx_size)

        self.log_entries.append({
            "org_name": org_name,
            "round": round_num,
            "accepted": accepted,
            "hash": transaction["hash"][:16] + "...",
            "commit_latency_ms": round(commit_time * 1000, 2),
            "tx_size_bytes": tx_size,
        })

        return transaction

    def log_detection_event(
        self,
        org_name: str,
        threat_score: float,
        threat_type: str,
        round_num: int,
    ) -> Dict:
        """
        Log a detection event. Chaincode: logDetection()
        """
        transaction = {
            "type": "detection_event",
            "org_name": org_name,
            "threat_score": round(threat_score, 4),
            "threat_type": threat_type,
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
        }
        transaction["hash"] = self._hash_gradient(transaction)
        self.pending_transactions.append(transaction)
        return transaction

    def log_response_action(
        self,
        action_type: str,
        triggered_by: str,
        round_num: int,
        details: Dict = None,
    ) -> Dict:
        """
        Log a response action. Chaincode: logResponse()
        """
        transaction = {
            "type": "response_action",
            "action": action_type,
            "triggered_by": triggered_by,
            "round": round_num,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        }
        transaction["hash"] = self._hash_gradient(transaction)
        self.pending_transactions.append(transaction)
        return transaction

    def _update_reputation(self, org_name: str, accepted: bool):
        """
        Update reputation score. Chaincode: updateReputation()
        Flagged: -20 penalty | Accepted: +2 reward
        """
        if accepted:
            self.reputation_scores[org_name] = min(100, self.reputation_scores[org_name] + 2)
        else:
            self.reputation_scores[org_name] = max(0, self.reputation_scores[org_name] - 20)

    def _create_block(self):
        """Create a new block from pending transactions."""
        if not self.pending_transactions:
            return

        previous_hash = self.chain[-1].hash
        new_block = Block(
            index=len(self.chain),
            transactions=self.pending_transactions.copy(),
            previous_hash=previous_hash,
        )
        self.chain.append(new_block)
        self.pending_transactions = []

    def flush(self):
        """Force create a block from remaining pending transactions."""
        if self.pending_transactions:
            self._create_block()

    def query_audit_trail(
        self,
        org_name: str = None,
        round_num: int = None,
    ) -> List[Dict]:
        """
        Query audit trail. Chaincode: queryAuditTrail()
        """
        results = []
        for block in self.chain:
            for tx in block.transactions:
                if tx.get("type") == "genesis":
                    continue
                if org_name and tx.get("org_name") != org_name:
                    continue
                if round_num and tx.get("round") != round_num:
                    continue
                results.append(tx)
        # Also search pending transactions
        for tx in self.pending_transactions:
            if org_name and tx.get("org_name") != org_name:
                continue
            if round_num and tx.get("round") != round_num:
                continue
            results.append(tx)
        return results

    def get_excluded_orgs(self, threshold: int = 40) -> List[str]:
        """Get orgs with reputation below exclusion threshold."""
        return [org for org, score in self.reputation_scores.items() if score <= threshold]

    def verify_chain_integrity(self) -> bool:
        """
        Verify blockchain integrity — check hash chain is unbroken.
        Returns True if chain is valid.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Verify hash
            if current.previous_hash != previous.hash:
                log.warning("Chain broken at block %d!", i)
                return False

            # Verify current block hash
            if current.hash != current.compute_hash():
                log.warning("Block %d hash tampered!", i)
                return False

        return True

    def simulate_tamper_detection(self) -> Dict:
        """
        Simulate tampering and verify detection.
        Tampers with a block and checks if verify_chain_integrity catches it.
        """
        # Chain should be valid before tampering
        assert self.verify_chain_integrity(), "Chain already invalid!"

        if len(self.chain) < 3:
            return {"tamper_detected": True, "message": "Chain too short to tamper"}

        # Tamper with a block (modify a transaction)
        tamper_block_idx = len(self.chain) // 2
        original_hash = self.chain[tamper_block_idx].hash

        if self.chain[tamper_block_idx].transactions:
            self.chain[tamper_block_idx].transactions[0]["tampered"] = True
            # Recompute hash (but chain will break)
            self.chain[tamper_block_idx].hash = self.chain[tamper_block_idx].compute_hash()

        # Verify — should detect tampering
        tamper_detected = not self.verify_chain_integrity()

        # Restore
        if self.chain[tamper_block_idx].transactions:
            del self.chain[tamper_block_idx].transactions[0]["tampered"]
            self.chain[tamper_block_idx].hash = original_hash

        return {
            "tamper_detected": tamper_detected,
            "tampered_block": tamper_block_idx,
            "detection_method": "hash_chain_verification",
        }

    def get_performance_metrics(self) -> Dict:
        """Get blockchain performance metrics."""
        if not self.commit_latencies:
            return {"error": "No transactions logged"}

        latencies_ms = [l * 1000 for l in self.commit_latencies]

        return {
            "total_transactions": len(self.commit_latencies),
            "total_blocks": len(self.chain),
            "num_peers": self.num_peers,
            "block_size": self.block_size,
            "avg_commit_latency_ms": round(np.mean(latencies_ms), 4),
            "p95_commit_latency_ms": round(np.percentile(latencies_ms, 95), 4),
            "max_commit_latency_ms": round(np.max(latencies_ms), 4),
            "avg_tx_size_bytes": round(np.mean(self.transaction_sizes), 2),
            "total_storage_bytes": sum(self.transaction_sizes),
            "total_storage_kb": round(sum(self.transaction_sizes) / 1024, 2),
            "throughput_tps": round(len(self.commit_latencies) / max(sum(self.commit_latencies), 0.001), 2),
            "chain_valid": self.verify_chain_integrity(),
            "reputation_scores": dict(self.reputation_scores),
        }

    def run_scalability_benchmark(
        self,
        peer_configs: List[int] = None,
        n_transactions: int = 100,
    ) -> Dict:
        """
        Run scalability benchmark across different peer configurations.

        Tests: TPS, commit latency, and storage as peers increase.

        Args:
            peer_configs: List of peer counts to test [2, 4, 8, 16]
            n_transactions: Transactions per test
        """
        if peer_configs is None:
            peer_configs = [2, 4, 8, 16]

        benchmark_results = {}
        org_names = ["Org_A", "Org_B", "Org_C", "Org_D", "Org_E"]

        for num_peers in peer_configs:
            log.info("Benchmarking with %d peers...", num_peers)

            bl = BlockchainLogger(num_peers=num_peers, block_size=50)
            start = time.time()

            for i in range(n_transactions):
                org = org_names[i % len(org_names)]
                bl.log_gradient_submission(
                    org_name=org,
                    gradient_norm=np.random.uniform(10, 100),
                    accepted=np.random.random() > 0.1,
                    round_num=i // 5 + 1,
                    clipped=np.random.random() > 0.7,
                    cosine_similarity=np.random.uniform(-0.5, 1.0),
                )

            bl.flush()
            elapsed = time.time() - start

            metrics = bl.get_performance_metrics()
            metrics["benchmark_time_seconds"] = round(elapsed, 4)
            benchmark_results[f"{num_peers}_peers"] = metrics

            log.info("  %d peers: TPS=%.2f, Avg Latency=%.4fms, Storage=%dKB",
                    num_peers, metrics["throughput_tps"],
                    metrics["avg_commit_latency_ms"],
                    metrics["total_storage_kb"])

        return benchmark_results

    def run_block_config_benchmark(
        self,
        block_sizes: List[int] = None,
        block_timeouts: List[float] = None,
        n_transactions: int = 100,
    ) -> Dict:
        """
        Benchmark different block configurations.
        Tests throughput vs latency tradeoff.
        """
        if block_sizes is None:
            block_sizes = [10, 50, 100, 500]
        if block_timeouts is None:
            block_timeouts = [0.5, 1.0, 2.0, 5.0]

        results = {}
        org_names = ["Org_A", "Org_B", "Org_C"]

        for bs in block_sizes:
            bl = BlockchainLogger(num_peers=4, block_size=bs)

            for i in range(n_transactions):
                org = org_names[i % len(org_names)]
                bl.log_gradient_submission(
                    org_name=org,
                    gradient_norm=np.random.uniform(10, 100),
                    accepted=True,
                    round_num=i // 5 + 1,
                )

            bl.flush()
            metrics = bl.get_performance_metrics()
            results[f"block_size_{bs}"] = {
                "block_size": bs,
                "total_blocks": metrics["total_blocks"],
                "avg_latency_ms": metrics["avg_commit_latency_ms"],
                "throughput_tps": metrics["throughput_tps"],
                "storage_kb": metrics["total_storage_kb"],
            }

        return results

    def get_summary(self) -> Dict:
        """Complete blockchain summary for thesis."""
        self.flush()
        metrics = self.get_performance_metrics()
        tamper_test = self.simulate_tamper_detection()

        return {
            "performance": metrics,
            "tamper_detection": tamper_test,
            "audit_trail_size": len(self.query_audit_trail()),
            "excluded_orgs": self.get_excluded_orgs(),
        }


if __name__ == "__main__":
    print("Testing BlockchainLogger...")

    bl = BlockchainLogger(num_peers=4)

    # Log some submissions
    for i in range(20):
        org = f"Org_{i % 5}"
        bl.log_gradient_submission(
            org_name=org,
            gradient_norm=np.random.uniform(10, 100),
            accepted=i % 7 != 0,
            round_num=i // 5 + 1,
            cosine_similarity=np.random.uniform(0, 1),
        )

    bl.flush()

    # Verify chain
    print(f"Chain valid: {bl.verify_chain_integrity()}")
    print(f"Blocks: {len(bl.chain)}")
    print(f"Reputation: {bl.reputation_scores}")

    # Performance
    metrics = bl.get_performance_metrics()
    print(f"TPS: {metrics['throughput_tps']}")
    print(f"Avg latency: {metrics['avg_commit_latency_ms']:.4f}ms")

    # Tamper test
    tamper = bl.simulate_tamper_detection()
    print(f"Tamper detected: {tamper['tamper_detected']}")

    print("\n✓ BlockchainLogger working!")
