"""
Week 8 Runner: Local IDS Training Pipeline
==========================================
Runs the complete Week 8 pipeline in one command:
1. Process CICIDS2017 data (or generate synthetic)
2. Partition data across 5 supply chain organizations (non-IID)
3. Train local CNN-LSTM models for each organization
4. Train centralized baseline for comparison
5. Evaluate and compare all models
6. Generate results summary

Usage:
    python run_week8.py
    python run_week8.py --epochs 10          # Quick test with fewer epochs
    python run_week8.py --num-orgs 3         # Fewer organizations
    python run_week8.py --skip-centralized   # Skip centralized baseline
"""

import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.cicids_loader import CICIDS2017Loader
from src.data.data_partitioner import DataPartitioner
from src.training.local_trainer import train_all_organizations, train_centralized_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Week 8: Local IDS Training Pipeline")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-orgs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for non-IID")
    parser.add_argument("--skip-centralized", action="store_true")
    args = parser.parse_args()

    total_start = time.time()

    print("=" * 70)
    print("  WEEK 8: LOCAL IDS TRAINING PIPELINE")
    print("  Federated Multi-Agent Learning — Aspect 02")
    print("=" * 70)

    # ---- Step 1: Process Dataset ----
    print("\n[STEP 1/5] Processing CICIDS2017 Dataset...")
    print("-" * 50)
    loader = CICIDS2017Loader(raw_dir=args.raw_dir, processed_dir=args.processed_dir)
    stats = loader.process()
    print(f"  ✓ {stats['total_samples']} samples, {stats['num_features']} features\n")

    # ---- Step 2: Partition Data ----
    print("[STEP 2/5] Partitioning Data Across Organizations...")
    print("-" * 50)
    features = np.load(Path(args.processed_dir) / "features.npy")
    labels = np.load(Path(args.processed_dir) / "binary_labels.npy")

    partitioner = DataPartitioner(
        num_orgs=args.num_orgs,
        strategy="dirichlet",
        alpha=args.alpha,
    )
    org_data = partitioner.partition(features, labels)
    partitioner.save_partition(org_data, output_dir=args.processed_dir)
    print(f"  ✓ Data partitioned across {args.num_orgs} organizations (α={args.alpha})\n")

    # ---- Step 3: Train Local Models ----
    print("[STEP 3/5] Training Local IDS Models...")
    print("-" * 50)
    local_results = train_all_organizations(
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
    )

    # ---- Step 4: Train Centralized Baseline ----
    if not args.skip_centralized:
        print("\n[STEP 4/5] Training Centralized Baseline...")
        print("-" * 50)
        centralized_results = train_centralized_baseline(
            processed_dir=args.processed_dir,
            results_dir=args.results_dir,
            epochs=args.epochs,
        )
    else:
        print("\n[STEP 4/5] Skipping Centralized Baseline")
        centralized_results = None

    # ---- Step 5: Final Summary ----
    print("\n[STEP 5/5] Generating Final Summary...")
    print("-" * 50)

    summary = {
        "week": 8,
        "aspect": "02 - Federated Multi-Agent Learning",
        "cr": "CR-01: Local IDS Training Pipeline",
        "dataset": {
            "name": "CICIDS2017",
            "total_samples": stats["total_samples"],
            "num_features": stats["num_features"],
            "type": "synthetic" if stats["total_samples"] == 50000 else "real",
        },
        "partition": {
            "strategy": "dirichlet",
            "alpha": args.alpha,
            "num_organizations": args.num_orgs,
        },
        "local_results": local_results,
        "centralized_baseline": centralized_results,
        "training_config": {
            "model": "CNN-LSTM",
            "epochs": args.epochs,
            "optimizer": "Adam",
            "learning_rate": 0.001,
        },
    }

    # Save
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "week8_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    total_time = time.time() - total_start

    # Final output
    print("\n" + "=" * 70)
    print("  WEEK 8 COMPLETE — LOCAL IDS TRAINING RESULTS")
    print("=" * 70)

    print(f"\n{'Organization':<30} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 70)
    for org, res in local_results.items():
        print(f"{org:<30} {res.get('accuracy',0):>8.4f} {res.get('precision',0):>8.4f} "
              f"{res.get('recall',0):>8.4f} {res.get('f1_score',0):>8.4f} {res.get('roc_auc',0):>8.4f}")

    if centralized_results:
        print("-" * 70)
        r = centralized_results
        print(f"{'Centralized Baseline':<30} {r.get('accuracy',0):>8.4f} {r.get('precision',0):>8.4f} "
              f"{r.get('recall',0):>8.4f} {r.get('f1_score',0):>8.4f} {r.get('roc_auc',0):>8.4f}")

    print("=" * 70)
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Results saved to: {results_dir / 'week8_summary.json'}")
    print("\n✓ Week 8 CR-01 Complete! Ready for Week 9 (Federated Learning)")


if __name__ == "__main__":
    main()
