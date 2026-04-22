"""
CICIDS2017 Dataset Loader
========================
Downloads, cleans, preprocesses, and prepares CICIDS2017 network traffic dataset
for CNN-LSTM based IDS training.

Reference: Sharafaldin et al., "Toward Generating a New Intrusion Detection Dataset 
and Intrusion Detection System", 2018.

Usage:
    python -m src.data.cicids_loader              # Download + preprocess
    python -m src.data.cicids_loader --info        # Show dataset statistics
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# CICIDS2017 Attack Label Mapping
# ============================================================
ATTACK_CATEGORY_MAP = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack \x96 Brute Force": "WebAttack",
    "Web Attack \x96 XSS": "WebAttack",
    "Web Attack \x96 Sql Injection": "WebAttack",
    "Web Attack – Brute Force": "WebAttack",
    "Web Attack – XSS": "WebAttack",
    "Web Attack – Sql Injection": "WebAttack",
    "Infiltration": "Infiltration",
    "PortScan": "PortScan",
    "Bot": "Botnet",
    "Heartbleed": "Heartbleed",
}

# Features to drop (high correlation or identifiers)
DROP_FEATURES = [
    "Flow ID", "Source IP", "Source Port", "Destination IP",
    "Destination Port", "Timestamp", "Protocol",
]

# Selected features for CNN-LSTM (top 30 by importance from literature)
SELECTED_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s",
    "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


class CICIDS2017Loader:
    """
    Complete pipeline for CICIDS2017 dataset:
    1. Load CSV files
    2. Clean (handle NaN, Inf, duplicates)
    3. Map labels to attack categories
    4. Feature selection and engineering
    5. Normalize features
    6. Save processed data
    """

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.feature_names: List[str] = []
        self.label_encoder: Dict[str, int] = {}
        self.stats: Dict = {}

    def load_csv_files(self, file_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load all CICIDS2017 CSV files and concatenate.
        If files not found, generates synthetic data for development.
        """
        csv_files = list(self.raw_dir.glob("*.csv"))

        if not csv_files:
            logger.warning("No CSV files found in %s. Generating synthetic CICIDS2017-like data...", self.raw_dir)
            return self._generate_synthetic_data()

        if file_list:
            csv_files = [self.raw_dir / f for f in file_list if (self.raw_dir / f).exists()]

        dfs = []
        for f in csv_files:
            logger.info("Loading %s...", f.name)
            try:
                df = pd.read_csv(f, encoding="utf-8", low_memory=False)
                df.columns = df.columns.str.strip()
                dfs.append(df)
                logger.info("  → %d rows, %d columns", len(df), len(df.columns))
            except Exception as e:
                logger.error("  → Failed to load %s: %s", f.name, e)

        if not dfs:
            logger.warning("No valid CSV files loaded. Generating synthetic data...")
            return self._generate_synthetic_data()

        combined = pd.concat(dfs, ignore_index=True)
        logger.info("Total loaded: %d rows, %d columns", len(combined), len(combined.columns))
        return combined

    def _generate_synthetic_data(self, n_samples: int = 50000) -> pd.DataFrame:
        """
        Generate synthetic CICIDS2017-like data for development/testing.
        Preserves realistic statistical properties.
        """
        logger.info("Generating %d synthetic samples...", n_samples)
        np.random.seed(42)

        # Generate features
        data = {}
        for feat in SELECTED_FEATURES:
            if "Flag" in feat or "Count" in feat:
                data[feat] = np.random.randint(0, 5, n_samples).astype(float)
            elif "Packets" in feat and "/" not in feat:
                data[feat] = np.abs(np.random.lognormal(3, 2, n_samples))
            elif "Length" in feat or "Bytes" in feat or "Size" in feat:
                data[feat] = np.abs(np.random.lognormal(5, 3, n_samples))
            elif "Duration" in feat or "IAT" in feat or "Active" in feat or "Idle" in feat:
                data[feat] = np.abs(np.random.exponential(1000, n_samples))
            elif "/s" in feat:
                data[feat] = np.abs(np.random.lognormal(2, 2, n_samples))
            elif "Ratio" in feat:
                data[feat] = np.random.uniform(0, 1, n_samples)
            elif "Win" in feat or "Init" in feat:
                data[feat] = np.random.randint(0, 65535, n_samples).astype(float)
            else:
                data[feat] = np.abs(np.random.normal(100, 50, n_samples))

        # Generate labels (realistic distribution: ~80% benign, ~20% attack)
        labels = np.random.choice(
            list(ATTACK_CATEGORY_MAP.keys()),
            size=n_samples,
            p=self._get_label_distribution()
        )
        data["Label"] = labels

        df = pd.DataFrame(data)
        logger.info("Synthetic data generated: %d rows, %d columns", len(df), len(df.columns))
        return df

    def _get_label_distribution(self) -> List[float]:
        """Realistic CICIDS2017 label distribution."""
        labels = list(ATTACK_CATEGORY_MAP.keys())
        n = len(labels)
        # BENIGN is ~80%, rest split among attacks
        probs = []
        for label in labels:
            if label == "BENIGN":
                probs.append(0.80)
            else:
                probs.append(0.20 / (n - 1))
        # Normalize
        total = sum(probs)
        return [p / total for p in probs]

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset:
        - Strip column names
        - Drop identifier columns
        - Handle NaN and Inf values
        - Remove duplicates
        - Convert to numeric
        """
        logger.info("Cleaning dataset...")
        initial_rows = len(df)

        # Strip column names
        df.columns = df.columns.str.strip()

        # Drop identifier columns
        drop_cols = [c for c in DROP_FEATURES if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            logger.info("  Dropped %d identifier columns", len(drop_cols))

        # Ensure Label column exists
        label_col = "Label" if "Label" in df.columns else None
        if label_col is None:
            for col in df.columns:
                if "label" in col.lower():
                    label_col = col
                    break

        if label_col and label_col != "Label":
            df = df.rename(columns={label_col: "Label"})

        # Separate labels
        labels = df["Label"].copy() if "Label" in df.columns else None
        feature_df = df.drop(columns=["Label"]) if labels is not None else df

        # Convert to numeric
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

        # Replace Inf with NaN, then fill NaN with 0
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        nan_count = feature_df.isna().sum().sum()
        if nan_count > 0:
            logger.info("  Replacing %d NaN/Inf values with 0", nan_count)
        feature_df = feature_df.fillna(0)

        # Remove duplicates
        if labels is not None:
            feature_df["Label"] = labels
        feature_df = feature_df.drop_duplicates()
        removed = initial_rows - len(feature_df)
        if removed > 0:
            logger.info("  Removed %d duplicate rows", removed)

        logger.info("  Clean data: %d rows, %d columns", len(feature_df), len(feature_df.columns))
        return feature_df

    def map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map raw CICIDS2017 labels to:
        - binary_label: 0 (BENIGN) / 1 (ATTACK)
        - attack_category: DoS, DDoS, BruteForce, etc.
        """
        logger.info("Mapping labels...")

        # Strip labels
        df["Label"] = df["Label"].str.strip()

        # Map to category
        df["attack_category"] = df["Label"].map(ATTACK_CATEGORY_MAP)
        unmapped = df["attack_category"].isna().sum()
        if unmapped > 0:
            unknown_labels = df[df["attack_category"].isna()]["Label"].unique()
            logger.warning("  %d rows with unknown labels: %s", unmapped, unknown_labels)
            df["attack_category"] = df["attack_category"].fillna("Unknown")

        # Binary label
        df["binary_label"] = (df["attack_category"] != "BENIGN").astype(int)

        # Multi-class encoding
        categories = sorted(df["attack_category"].unique())
        self.label_encoder = {cat: i for i, cat in enumerate(categories)}
        df["multi_label"] = df["attack_category"].map(self.label_encoder)

        # Log distribution
        dist = df["attack_category"].value_counts()
        logger.info("  Label distribution:")
        for cat, count in dist.items():
            logger.info("    %s: %d (%.1f%%)", cat, count, 100 * count / len(df))

        self.stats["label_distribution"] = dist.to_dict()
        return df

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant features for CNN-LSTM model."""
        logger.info("Selecting features...")

        available = [f for f in SELECTED_FEATURES if f in df.columns]
        missing = [f for f in SELECTED_FEATURES if f not in df.columns]

        if missing:
            logger.warning("  %d features not found: %s", len(missing), missing[:5])

        # If too few selected features found, use all numeric
        if len(available) < 10:
            logger.warning("  Too few selected features. Using all numeric columns.")
            label_cols = ["Label", "binary_label", "multi_label", "attack_category"]
            available = [c for c in df.columns if c not in label_cols and df[c].dtype in [np.float64, np.int64, np.float32]]

        self.feature_names = available
        logger.info("  Selected %d features", len(available))
        return df

    def normalize_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features using Min-Max scaling to [0, 1].
        Returns: (features, binary_labels, multi_labels)
        """
        logger.info("Normalizing features...")

        features = df[self.feature_names].values.astype(np.float32)

        # Min-Max normalization
        self.feature_min = features.min(axis=0)
        self.feature_max = features.max(axis=0)
        feature_range = self.feature_max - self.feature_min
        feature_range[feature_range == 0] = 1  # avoid division by zero

        features = (features - self.feature_min) / feature_range

        # Clip to [0, 1]
        features = np.clip(features, 0, 1)

        binary_labels = df["binary_label"].values.astype(np.int64)
        multi_labels = df["multi_label"].values.astype(np.int64)

        logger.info("  Features shape: %s", features.shape)
        logger.info("  Binary labels: %d benign, %d attack",
                     (binary_labels == 0).sum(), (binary_labels == 1).sum())

        return features, binary_labels, multi_labels

    def process(self) -> Dict:
        """
        Full processing pipeline:
        Load → Clean → Map Labels → Select Features → Normalize → Save
        """
        logger.info("=" * 60)
        logger.info("CICIDS2017 Processing Pipeline")
        logger.info("=" * 60)

        # Step 1: Load
        df = self.load_csv_files()

        # Step 2: Clean
        df = self.clean_data(df)

        # Step 3: Map labels
        df = self.map_labels(df)

        # Step 4: Select features
        df = self.select_features(df)

        # Step 5: Normalize
        features, binary_labels, multi_labels = self.normalize_features(df)

        # Step 6: Save
        self._save_processed(features, binary_labels, multi_labels)

        self.stats.update({
            "total_samples": len(features),
            "num_features": features.shape[1],
            "feature_names": self.feature_names,
            "label_encoder": self.label_encoder,
            "binary_class_distribution": {
                "benign": int((binary_labels == 0).sum()),
                "attack": int((binary_labels == 1).sum()),
            },
        })

        # Save stats
        stats_path = self.processed_dir / "dataset_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
        logger.info("Stats saved to %s", stats_path)

        logger.info("=" * 60)
        logger.info("Processing complete! %d samples, %d features", len(features), features.shape[1])
        logger.info("=" * 60)

        return self.stats

    def _save_processed(self, features: np.ndarray, binary_labels: np.ndarray, multi_labels: np.ndarray):
        """Save processed arrays as .npy files."""
        np.save(self.processed_dir / "features.npy", features)
        np.save(self.processed_dir / "binary_labels.npy", binary_labels)
        np.save(self.processed_dir / "multi_labels.npy", multi_labels)

        # Save normalization parameters for later use
        np.save(self.processed_dir / "feature_min.npy", self.feature_min)
        np.save(self.processed_dir / "feature_max.npy", self.feature_max)

        logger.info("Saved processed data to %s", self.processed_dir)

    def load_processed(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load already processed data."""
        features = np.load(self.processed_dir / "features.npy")
        binary_labels = np.load(self.processed_dir / "binary_labels.npy")
        multi_labels = np.load(self.processed_dir / "multi_labels.npy")

        with open(self.processed_dir / "dataset_stats.json", "r") as f:
            self.stats = json.load(f)

        self.feature_names = self.stats.get("feature_names", [])
        self.label_encoder = self.stats.get("label_encoder", {})

        logger.info("Loaded processed data: %d samples, %d features", len(features), features.shape[1])
        return features, binary_labels, multi_labels


def main():
    parser = argparse.ArgumentParser(description="CICIDS2017 Dataset Loader")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--processed-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--info", action="store_true", help="Show dataset statistics only")
    args = parser.parse_args()

    loader = CICIDS2017Loader(raw_dir=args.raw_dir, processed_dir=args.processed_dir)

    if args.info:
        try:
            features, binary_labels, multi_labels = loader.load_processed()
            print(f"\nDataset: {loader.stats.get('total_samples')} samples")
            print(f"Features: {loader.stats.get('num_features')}")
            print(f"Binary: {loader.stats.get('binary_class_distribution')}")
            print(f"Categories: {loader.stats.get('label_encoder')}")
        except FileNotFoundError:
            print("No processed data found. Run without --info first.")
    else:
        loader.process()


if __name__ == "__main__":
    main()
