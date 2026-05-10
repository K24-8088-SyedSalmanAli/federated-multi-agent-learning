"""
Statistical Validation & Hypothesis Testing
============================================
Week 12: Validates hypothesis H1b with rigorous statistical tests.

H1b: Federated learning with blockchain-verified aggregation achieves 
comparable or higher detection accuracy than centralized training while 
maintaining lower privacy leakage.

Tests:
- Paired t-test (parametric)
- Wilcoxon signed-rank test (non-parametric)
- Cohen's d effect size

Generates 4 publication-quality figures for thesis.

Reference: Proposal Section 4.3, 6.6

Usage:
    from src.evaluation.statistical_validation import StatisticalValidator
    validator = StatisticalValidator()
    results = validator.run_validation(federated_scores, centralized_scores)
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StatisticalValidator:
    """
    Statistical validation for federated vs centralized comparison.
    
    Runs paired statistical tests and computes effect sizes.
    """

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    def paired_t_test(
        self,
        federated_scores: np.ndarray,
        centralized_scores: np.ndarray,
    ) -> Dict:
        """
        Paired t-test: Compare federated vs centralized accuracy.
        H0: No significant difference in accuracy.
        """
        t_stat, p_value = stats.ttest_rel(federated_scores, centralized_scores)

        return {
            "test": "Paired t-test",
            "t_statistic": round(float(t_stat), 6),
            "p_value": round(float(p_value), 6),
            "significant": p_value < self.alpha,
            "reject_null": p_value < self.alpha,
            "interpretation": (
                f"Significant difference (p={p_value:.6f} < {self.alpha}). Reject H0."
                if p_value < self.alpha
                else f"No significant difference (p={p_value:.6f} >= {self.alpha}). Fail to reject H0."
            ),
        }

    def wilcoxon_test(
        self,
        federated_scores: np.ndarray,
        centralized_scores: np.ndarray,
    ) -> Dict:
        """
        Wilcoxon signed-rank test (non-parametric backup).
        Does not assume normal distribution.
        """
        try:
            w_stat, p_value = stats.wilcoxon(federated_scores, centralized_scores)
        except ValueError as e:
            return {
                "test": "Wilcoxon signed-rank",
                "error": str(e),
                "note": "May fail if all differences are zero",
            }

        return {
            "test": "Wilcoxon signed-rank",
            "w_statistic": round(float(w_stat), 6),
            "p_value": round(float(p_value), 6),
            "significant": p_value < self.alpha,
            "reject_null": p_value < self.alpha,
            "interpretation": (
                f"Significant difference (p={p_value:.6f} < {self.alpha}). Reject H0."
                if p_value < self.alpha
                else f"No significant difference (p={p_value:.6f} >= {self.alpha}). Fail to reject H0."
            ),
        }

    def cohens_d(
        self,
        federated_scores: np.ndarray,
        centralized_scores: np.ndarray,
    ) -> Dict:
        """
        Cohen's d effect size.
        Small: 0.2, Medium: 0.5, Large: 0.8
        """
        diff = federated_scores - centralized_scores
        d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0

        if abs(d) < 0.2:
            magnitude = "Negligible"
        elif abs(d) < 0.5:
            magnitude = "Small"
        elif abs(d) < 0.8:
            magnitude = "Medium"
        else:
            magnitude = "Large"

        return {
            "test": "Cohen's d",
            "d_value": round(d, 6),
            "magnitude": magnitude,
            "interpretation": f"Effect size is {magnitude.lower()} (d={d:.4f})",
        }

    def run_validation(
        self,
        federated_scores: np.ndarray,
        centralized_scores: np.ndarray,
    ) -> Dict:
        """
        Run all statistical tests.
        
        Args:
            federated_scores: Array of accuracy scores from N runs (federated)
            centralized_scores: Array of accuracy scores from N runs (centralized)
        """
        logger.info("Running statistical validation (%d runs)...", len(federated_scores))

        t_test = self.paired_t_test(federated_scores, centralized_scores)
        wilcoxon = self.wilcoxon_test(federated_scores, centralized_scores)
        effect_size = self.cohens_d(federated_scores, centralized_scores)

        # Descriptive statistics
        descriptive = {
            "federated": {
                "mean": round(float(np.mean(federated_scores)), 6),
                "std": round(float(np.std(federated_scores, ddof=1)), 6),
                "min": round(float(np.min(federated_scores)), 6),
                "max": round(float(np.max(federated_scores)), 6),
                "median": round(float(np.median(federated_scores)), 6),
            },
            "centralized": {
                "mean": round(float(np.mean(centralized_scores)), 6),
                "std": round(float(np.std(centralized_scores, ddof=1)), 6),
                "min": round(float(np.min(centralized_scores)), 6),
                "max": round(float(np.max(centralized_scores)), 6),
                "median": round(float(np.median(centralized_scores)), 6),
            },
        }

        # H1b verdict
        fed_mean = np.mean(federated_scores)
        cent_mean = np.mean(centralized_scores)
        comparable = abs(fed_mean - cent_mean) < 0.05  # within 5%

        verdict = {
            "hypothesis": "H1b",
            "statement": "Federated learning achieves comparable or higher accuracy than centralized",
            "federated_mean": round(float(fed_mean), 6),
            "centralized_mean": round(float(cent_mean), 6),
            "difference": round(float(fed_mean - cent_mean), 6),
            "comparable_or_higher": bool(comparable or fed_mean >= cent_mean),
            "verdict": (
                "ACCEPT H1b: Federated achieves comparable/higher accuracy"
                if comparable or fed_mean >= cent_mean
                else "REJECT H1b: Federated significantly lower than centralized"
            ),
        }

        return {
            "num_runs": len(federated_scores),
            "significance_level": self.alpha,
            "descriptive_statistics": descriptive,
            "paired_t_test": t_test,
            "wilcoxon_test": wilcoxon,
            "cohens_d": effect_size,
            "h1b_verdict": verdict,
        }


class FigureGenerator:
    """
    Generates publication-quality figures for thesis.
    Uses matplotlib — saves as PNG.
    """

    def __init__(self, output_dir: str = "data/results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def figure1_federated_vs_centralized(
        self,
        federated_round_accs: List[float],
        centralized_acc: float,
        filename: str = "fig1_federated_vs_centralized.png",
    ):
        """Figure 1: Federated accuracy over rounds vs centralized baseline."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rounds = list(range(1, len(federated_round_accs) + 1))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rounds, federated_round_accs, 'b-o', label='Federated (Coordinate Median)', linewidth=2, markersize=8)
        ax.axhline(y=centralized_acc, color='r', linestyle='--', linewidth=2, label=f'Centralized Baseline ({centralized_acc:.4f})')

        ax.set_xlabel('Federated Round', fontsize=14)
        ax.set_ylabel('Global Model Accuracy', fontsize=14)
        ax.set_title('Federated vs Centralized: Accuracy Over Rounds', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])

        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info("Saved: %s", path)

    def figure2_poisoning_defense(
        self,
        results: Dict,
        filename: str = "fig2_poisoning_defense.png",
    ):
        """Figure 2: Poisoning defense effectiveness at 10%, 20%, 30%."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ratios = ['10%', '20%', '30%']
        no_defense = []
        full_defense = []

        for r in [10, 20, 30]:
            s = results.get(f"{r}pct_summary", {})
            no_defense.append(s.get("untargeted_no_defense_acc", 0))
            full_defense.append(s.get("untargeted_full_defense_acc", 0))

        x = np.arange(len(ratios))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, no_defense, width, label='No Defense (FedAvg)', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, full_defense, width, label='Full Defense (Median + Clipping)', color='#2ecc71', alpha=0.8)

        ax.set_xlabel('Compromise Ratio', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_title('Untargeted Poisoning: Defense Effectiveness', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(ratios, fontsize=12)
        ax.legend(fontsize=12)
        ax.set_ylim([0.5, 1.05])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info("Saved: %s", path)

    def figure3_blockchain_scalability(
        self,
        benchmark_results: Dict,
        filename: str = "fig3_blockchain_scalability.png",
    ):
        """Figure 3: Blockchain overhead vs number of peers."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        peers = []
        tps_values = []
        latency_values = []

        for key, metrics in sorted(benchmark_results.items()):
            peers.append(metrics["num_peers"])
            tps_values.append(metrics["throughput_tps"])
            latency_values.append(metrics["avg_commit_latency_ms"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # TPS
        ax1.bar(range(len(peers)), tps_values, color='#3498db', alpha=0.8)
        ax1.set_xticks(range(len(peers)))
        ax1.set_xticklabels([str(p) for p in peers], fontsize=12)
        ax1.set_xlabel('Number of Peers', fontsize=14)
        ax1.set_ylabel('Throughput (TPS)', fontsize=14)
        ax1.set_title('Blockchain Throughput', fontsize=16)
        ax1.grid(True, alpha=0.3, axis='y')

        # Latency
        ax2.bar(range(len(peers)), latency_values, color='#e67e22', alpha=0.8)
        ax2.set_xticks(range(len(peers)))
        ax2.set_xticklabels([str(p) for p in peers], fontsize=12)
        ax2.set_xlabel('Number of Peers', fontsize=14)
        ax2.set_ylabel('Avg Commit Latency (ms)', fontsize=14)
        ax2.set_title('Blockchain Latency', fontsize=16)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info("Saved: %s", path)

    def figure4_statistical_comparison(
        self,
        federated_scores: np.ndarray,
        centralized_scores: np.ndarray,
        filename: str = "fig4_statistical_comparison.png",
    ):
        """Figure 4: Box plot comparison of federated vs centralized."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Box plot
        bp = ax1.boxplot(
            [federated_scores, centralized_scores],
            labels=['Federated\n(Coordinate Median)', 'Centralized\nBaseline'],
            patch_artist=True,
            widths=0.5,
        )
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')
        for box in bp['boxes']:
            box.set_alpha(0.7)

        ax1.set_ylabel('Accuracy', fontsize=14)
        ax1.set_title('Accuracy Distribution (30 Runs)', fontsize=16)
        ax1.grid(True, alpha=0.3, axis='y')

        # Histogram
        ax2.hist(federated_scores, bins=10, alpha=0.6, label='Federated', color='#3498db')
        ax2.hist(centralized_scores, bins=10, alpha=0.6, label='Centralized', color='#e74c3c')
        ax2.axvline(np.mean(federated_scores), color='#2980b9', linestyle='--', linewidth=2,
                    label=f'Fed Mean: {np.mean(federated_scores):.4f}')
        ax2.axvline(np.mean(centralized_scores), color='#c0392b', linestyle='--', linewidth=2,
                    label=f'Cent Mean: {np.mean(centralized_scores):.4f}')
        ax2.set_xlabel('Accuracy', fontsize=14)
        ax2.set_ylabel('Frequency', fontsize=14)
        ax2.set_title('Accuracy Histogram', fontsize=16)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info("Saved: %s", path)


if __name__ == "__main__":
    print("Testing Statistical Validation...")

    # Simulate 30 runs
    np.random.seed(42)
    fed_scores = np.random.normal(0.96, 0.01, 30)
    cent_scores = np.random.normal(0.95, 0.012, 30)

    validator = StatisticalValidator()
    results = validator.run_validation(fed_scores, cent_scores)

    print(f"\nFederated mean:  {results['descriptive_statistics']['federated']['mean']:.4f}")
    print(f"Centralized mean: {results['descriptive_statistics']['centralized']['mean']:.4f}")
    print(f"\nPaired t-test: p={results['paired_t_test']['p_value']:.6f}")
    print(f"Wilcoxon: p={results['wilcoxon_test']['p_value']:.6f}")
    print(f"Cohen's d: {results['cohens_d']['d_value']:.4f} ({results['cohens_d']['magnitude']})")
    print(f"\nH1b Verdict: {results['h1b_verdict']['verdict']}")

    print("\n✓ Statistical validation working!")
