"""
Evaluation Metrics for IDS
=========================
Computes all metrics needed for thesis evaluation:
Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.

Supports both binary and multi-class classification.

Usage:
    from src.evaluation.metrics import compute_all_metrics
    metrics = compute_all_metrics(y_true, y_pred, y_probs)
"""

import numpy as np
from typing import Dict, Optional
from collections import Counter


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall accuracy."""
    return float(np.mean(y_true == y_pred))


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """Precision = TP / (TP + FP)."""
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """Recall = TP / (TP + FN)."""
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """F1-Score = 2 * (precision * recall) / (precision + recall)."""
    precision = compute_precision(y_true, y_pred, pos_label)
    recall = compute_recall(y_true, y_pred, pos_label)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_roc_auc(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    ROC-AUC for binary classification.
    Manual implementation (no sklearn dependency).
    y_probs: probability of positive class (attack).
    """
    if y_probs.ndim == 2:
        # Multi-class probs → use column for positive class
        y_scores = y_probs[:, 1] if y_probs.shape[1] > 1 else y_probs[:, 0]
    else:
        y_scores = y_probs

    # Sort by predicted score descending
    sorted_indices = np.argsort(-y_scores)
    y_sorted = y_true[sorted_indices]
    scores_sorted = y_scores[sorted_indices]

    # Count positives and negatives
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5  # undefined, return 0.5

    # Compute TPR and FPR at each threshold
    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_pos
        fpr = fp / n_neg

        # Trapezoidal integration
        if fpr != prev_fpr:
            auc += (fpr - prev_fpr) * tpr
            prev_fpr = fpr

    return float(auc)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true)][int(pred)] += 1
    return cm


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (for ROC-AUC)

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "accuracy": compute_accuracy(y_true, y_pred),
        "precision": compute_precision(y_true, y_pred),
        "recall": compute_recall(y_true, y_pred),
        "f1_score": compute_f1(y_true, y_pred),
        "total_samples": int(len(y_true)),
        "class_distribution": {str(k): int(v) for k, v in Counter(y_true.tolist()).items()},
    }

    if y_probs is not None:
        metrics["roc_auc"] = compute_roc_auc(y_true, y_probs)

    # Confusion matrix
    num_classes = len(np.unique(y_true))
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    metrics["confusion_matrix"] = cm.tolist()

    if num_classes == 2:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return metrics


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n = 1000
    y_true = np.random.randint(0, 2, n)
    y_pred = y_true.copy()
    # Add some noise
    noise_idx = np.random.choice(n, size=50, replace=False)
    y_pred[noise_idx] = 1 - y_pred[noise_idx]
    y_probs = np.random.rand(n, 2)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

    metrics = compute_all_metrics(y_true, y_pred, y_probs)
    print("Test Metrics:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"  confusion_matrix:\n    {metrics['confusion_matrix']}")
