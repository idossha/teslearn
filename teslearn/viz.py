"""
Visualization for TESLearn.

Brain visualization (glass brain) and model evaluation plots
(confusion matrix, ROC curve, feature importance, CV results).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _check_nilearn():
    """Raise a clear error if nilearn is not installed."""
    try:
        import nilearn  # noqa: F401
    except ImportError:
        raise ImportError(
            "nilearn is required for brain visualization.\n"
            "Install it with:  pip install nilearn>=0.10.0"
        )


def _parse_voxel_importance(
    feature_importance: Dict[str, float],
) -> tuple[list[tuple[int, int, int]], list[float]]:
    """Extract voxel coordinates and weights from feature importance dict.

    Parameters
    ----------
    feature_importance : Dict[str, float]
        Feature names mapped to importance scores.  Only keys matching
        ``voxel_X_Y_Z`` are used; others are silently skipped.

    Returns
    -------
    coords : list of (x, y, z)
    weights : list of float
    """
    pattern = re.compile(r"^voxel_(\d+)_(\d+)_(\d+)$")
    coords: list[tuple[int, int, int]] = []
    weights: list[float] = []

    for name, weight in feature_importance.items():
        m = pattern.match(name)
        if m is not None:
            coords.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
            weights.append(weight)

    return coords, weights


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def create_stat_map(
    feature_importance: Dict[str, float],
    reference_image: Any,
) -> Any:
    """Create a NIfTI stat map from voxel-level feature importance.

    Parses ``voxel_X_Y_Z`` keys, fills a 3-D volume with the corresponding
    weights, and returns a NIfTI image that shares the reference affine.

    Parameters
    ----------
    feature_importance : Dict[str, float]
        Output of ``pipeline.get_feature_importance()``.
    reference_image : nibabel.Nifti1Image
        Any NIfTI image whose shape and affine define the target space
        (e.g. one of the training images).

    Returns
    -------
    stat_map : nibabel.Nifti1Image
        3-D volume with voxel weights.
    """
    _check_nilearn()
    import nibabel as nib

    coords, weights = _parse_voxel_importance(feature_importance)
    if not coords:
        raise ValueError(
            "No voxel features found in feature_importance. "
            "Expected keys like 'voxel_X_Y_Z'."
        )

    shape = reference_image.shape[:3]
    data = np.zeros(shape, dtype=np.float64)

    for (x, y, z), w in zip(coords, weights):
        data[x, y, z] = w

    return nib.Nifti1Image(data, affine=reference_image.affine)


def plot_glass_brain(
    pipeline: Any,
    reference_image: Any,
    *,
    threshold: float = 0.01,
    title: Optional[str] = "Feature Importance",
    display_mode: str = "lyrz",
    colorbar: bool = True,
    output_file: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Plot model coefficients on a glass brain.

    Parameters
    ----------
    pipeline : TESPipeline
        A fitted pipeline (must expose ``get_feature_importance()``).
    reference_image : nibabel.Nifti1Image
        Image whose shape/affine define the voxel grid.
    threshold : float
        Values below this magnitude are not shown.
    title : str or None
        Plot title.
    display_mode : str
        Nilearn display mode (default ``"lyrz"``).
    colorbar : bool
        Show colour bar.
    output_file : str or None
        If given, save the figure to this path.
    **kwargs
        Forwarded to ``nilearn.plotting.plot_glass_brain``.

    Returns
    -------
    display : nilearn Display object
    """
    _check_nilearn()
    from nilearn.plotting import plot_glass_brain as _plot_glass_brain

    stat_map = create_stat_map(
        pipeline.get_feature_importance(), reference_image
    )

    return _plot_glass_brain(
        stat_map,
        threshold=threshold,
        title=title,
        display_mode=display_mode,
        colorbar=colorbar,
        output_file=output_file,
        **kwargs,
    )


# ------------------------------------------------------------------
# Evaluation plots
# ------------------------------------------------------------------


def plot_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    pipeline: Any,
    *,
    n_top: int = 10,
    n_bins: int = 5,
    labels: Tuple[str, str] = ("Non-responder", "Responder"),
) -> plt.Figure:
    """Plot a 3x2 evaluation dashboard.

    Panels: confusion matrix, ROC curve, precision-recall curve,
    calibration curve, feature importance, prediction distribution.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    pipeline : TESPipeline
        Fitted pipeline (used for feature importance).
    n_top : int
        Number of top features to show.
    n_bins : int
        Number of bins for the calibration curve.
    labels : tuple of str
        Class labels for the confusion matrix.

    Returns
    -------
    fig : matplotlib Figure
    """
    from sklearn.metrics import (
        confusion_matrix as sk_cm,
        roc_curve,
        auc,
        precision_recall_curve,
        average_precision_score,
    )
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    # --- Confusion matrix ---
    cm = sk_cm(y_true, y_pred)
    axes[0, 0].imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            axes[0, 0].text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color=color, fontsize=14,
            )
    axes[0, 0].set_xticks(range(len(labels)))
    axes[0, 0].set_yticks(range(len(labels)))
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_yticklabels(labels)
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")
    axes[0, 0].set_title("Confusion Matrix")

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0, 1].plot(
        fpr, tpr, color="darkorange", lw=2,
        label=f"ROC (AUC = {roc_auc:.3f})",
    )
    axes[0, 1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("ROC Curve")
    axes[0, 1].legend(loc="lower right")

    # --- Precision-Recall curve ---
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    prevalence = np.mean(y_true)
    axes[1, 0].plot(
        recall, precision, color="darkorange", lw=2,
        label=f"PR (AP = {ap:.3f})",
    )
    axes[1, 0].axhline(
        y=prevalence, color="navy", lw=2, linestyle="--",
        label=f"Baseline ({prevalence:.2f})",
    )
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision-Recall Curve")
    axes[1, 0].legend(loc="lower left")

    # --- Calibration / reliability curve ---
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform",
    )
    axes[1, 1].plot(
        prob_pred, prob_true, "s-", color="darkorange", lw=2, label="Model",
    )
    axes[1, 1].plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Perfectly calibrated",
    )
    axes[1, 1].set_xlim([0.0, 1.0])
    axes[1, 1].set_ylim([0.0, 1.05])
    axes[1, 1].set_xlabel("Mean Predicted Probability")
    axes[1, 1].set_ylabel("Fraction of Positives")
    axes[1, 1].set_title("Calibration Curve")
    axes[1, 1].legend(loc="lower right")

    # --- Feature importance ---
    importance = pipeline.get_feature_importance()
    names = list(importance.keys())
    values = list(importance.values())
    sorted_idx = np.argsort(np.abs(values))[-n_top:][::-1]
    axes[2, 0].barh(
        range(len(sorted_idx)),
        [values[i] for i in sorted_idx],
        color="steelblue",
    )
    axes[2, 0].set_yticks(range(len(sorted_idx)))
    axes[2, 0].set_yticklabels([names[i] for i in sorted_idx])
    axes[2, 0].set_xlabel("Coefficient Value")
    axes[2, 0].set_title(f"Feature Importance (Top {n_top})")
    axes[2, 0].invert_yaxis()

    # --- Prediction distribution ---
    axes[2, 1].hist(
        y_proba[y_true == 0], bins=10, alpha=0.5,
        label="Non-responders", color="red",
    )
    axes[2, 1].hist(
        y_proba[y_true == 1], bins=10, alpha=0.5,
        label="Responders", color="green",
    )
    axes[2, 1].set_xlabel("Predicted Probability")
    axes[2, 1].set_ylabel("Count")
    axes[2, 1].set_title("Prediction Distribution")
    axes[2, 1].legend()

    plt.tight_layout()
    return fig


def plot_cv_results(result: Any) -> plt.Figure:
    """Plot cross-validation accuracy and ROC AUC per fold.

    Parameters
    ----------
    result : TrainingResult
        Output of ``teslearn.train_model()``.

    Returns
    -------
    fig : matplotlib Figure
    """
    cv_scores = [r.accuracy for r in result.cv_results if r.accuracy is not None]
    cv_roc_aucs = [r.roc_auc for r in result.cv_results if r.roc_auc is not None]

    n_plots = 1 + (1 if cv_roc_aucs else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Accuracy
    folds = range(1, len(cv_scores) + 1)
    axes[0].bar(folds, cv_scores, color="steelblue", alpha=0.7)
    axes[0].axhline(
        y=np.mean(cv_scores), color="red", linestyle="--",
        label=f"Mean: {np.mean(cv_scores):.3f}",
    )
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Cross-Validation Accuracy")
    axes[0].set_ylim([0, 1])
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # ROC AUC
    if cv_roc_aucs:
        folds_auc = range(1, len(cv_roc_aucs) + 1)
        axes[1].bar(folds_auc, cv_roc_aucs, color="darkgreen", alpha=0.7)
        axes[1].axhline(
            y=np.mean(cv_roc_aucs), color="red", linestyle="--",
            label=f"Mean: {np.mean(cv_roc_aucs):.3f}",
        )
        axes[1].set_xlabel("Fold")
        axes[1].set_ylabel("ROC AUC")
        axes[1].set_title("Cross-Validation ROC AUC")
        axes[1].set_ylim([0, 1])
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig

