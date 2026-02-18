"""
Evaluation utilities for TESLearn.

Provides metrics and evaluation functions tailored for TES responder prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    return float(np.mean(y_true == y_pred))


def roc_auc_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute ROC AUC score for binary classification."""
    from sklearn.metrics import roc_auc_score as sk_roc_auc

    return sk_roc_auc(y_true, y_proba)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute precision for binary classification."""
    from sklearn.metrics import precision_score as sk_precision

    return sk_precision(y_true, y_pred, zero_division=0)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall for binary classification."""
    from sklearn.metrics import recall_score as sk_recall

    return sk_recall(y_true, y_pred, zero_division=0)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 score for binary classification."""
    from sklearn.metrics import f1_score as sk_f1

    return sk_f1(y_true, y_pred, zero_division=0)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix."""
    from sklearn.metrics import confusion_matrix as sk_confusion

    return sk_confusion(y_true, y_pred)


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """Generate classification report."""
    from sklearn.metrics import classification_report as sk_report

    return sk_report(y_true, y_pred, target_names=target_names, zero_division=0)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }

    def __str__(self) -> str:
        """String representation of metrics."""
        lines = [
            "Classification Metrics:",
            f"  Accuracy:  {self.accuracy:.4f}",
            f"  Precision: {self.precision:.4f}",
            f"  Recall:    {self.recall:.4f}",
            f"  F1 Score:  {self.f1:.4f}",
        ]
        if self.roc_auc is not None:
            lines.append(f"  ROC AUC:   {self.roc_auc:.4f}")
        return "\n".join(lines)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> ClassificationMetrics:
    """
    Evaluate classification predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : Optional[np.ndarray]
        Predicted probabilities (for ROC AUC)

    Returns
    -------
    metrics : ClassificationMetrics
        Computed metrics
    """
    metrics = ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        confusion_matrix=confusion_matrix(y_true, y_pred),
    )

    if y_proba is not None:
        try:
            metrics.roc_auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            pass

    return metrics
