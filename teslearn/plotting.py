"""
Plotting utilities for TESLearn.

Custom plotting functions for visualizing model results, feature importance,
and diagnostics without relying on nilearn's plotting capabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


def plot_feature_importance(
    importance: Dict[str, float],
    output_path: Path,
    top_k: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Feature Importance",
    color_positive: str = "red",
    color_negative: str = "blue",
) -> None:
    """
    Plot feature importance as a horizontal bar chart.

    Parameters
    ----------
    importance : Dict[str, float]
        Feature names mapped to importance values
    output_path : Path
        Output path for figure
    top_k : int
        Number of top features to display
    figsize : Tuple[int, int]
        Figure size (width, height)
    title : str
        Plot title
    color_positive : str
        Color for positive weights
    color_negative : str
        Color for negative weights
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )
        raise

    # Sort features by absolute importance
    sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[
        :top_k
    ]

    # Separate names and values
    names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create bars
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values)

    # Color bars based on sign
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val > 0:
            bar.set_color(color_positive)
            bar.set_alpha(0.7)
        else:
            bar.set_color(color_negative)
            bar.set_alpha(0.7)

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()  # Top to bottom
    ax.set_xlabel("Coefficient Value", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.axvline(x=0, color="black", linewidth=0.5)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_positive, alpha=0.7, label="Positive (responders)"),
        Patch(facecolor=color_negative, alpha=0.7, label="Negative (non-responders)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Feature importance plot saved to {output_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 8),
) -> None:
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Predicted scores/probabilities
    output_path : Path
        Output path
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        logger.error("matplotlib and sklearn are required for plotting")
        raise

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    ax.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier"
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"ROC curve saved to {output_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 8),
) -> None:
    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Predicted scores/probabilities
    output_path : Path
        Output path
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError:
        logger.error("matplotlib and sklearn are required for plotting")
        raise

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AP = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Precision-Recall curve saved to {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
) -> None:
    """
    Plot confusion matrix as a heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    output_path : Path
        Output path
    labels : Optional[List[str]]
        Class labels
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    cmap : str
        Colormap name
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError:
        logger.error("matplotlib and sklearn are required for plotting")
        raise

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    if labels is None:
        labels = ["Non-Responder", "Responder"]

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrix saved to {output_path}")


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    title: str = "Prediction Distribution",
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot distribution of predicted probabilities by true class.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Predicted probabilities
    output_path : Path
        Output path
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting")
        raise

    fig, ax = plt.subplots(figsize=figsize)

    # Separate by true class
    non_responder_scores = y_score[y_true == 0]
    responder_scores = y_score[y_true == 1]

    # Plot histograms
    ax.hist(
        non_responder_scores,
        bins=30,
        alpha=0.5,
        label="Non-Responders",
        color="blue",
        edgecolor="black",
    )
    ax.hist(
        responder_scores,
        bins=30,
        alpha=0.5,
        label="Responders",
        color="red",
        edgecolor="black",
    )

    ax.set_xlabel("Predicted Probability", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Prediction distribution plot saved to {output_path}")


def plot_training_history(
    cv_results: List[Any],
    output_path: Path,
    metric: str = "roc_auc",
    title: str = "Cross-Validation Performance",
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot CV performance across folds.

    Parameters
    ----------
    cv_results : List[CVResult]
        Cross-validation results
    output_path : Path
        Output path
    metric : str
        Metric to plot ('roc_auc', 'accuracy', 'r2', etc.)
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting")
        raise

    # Extract metric values
    values = []
    for result in cv_results:
        val = getattr(result, metric, None)
        if val is not None:
            values.append(val)

    if not values:
        logger.warning(f"No values found for metric: {metric}")
        return

    fig, ax = plt.subplots(figsize=figsize)

    folds = range(1, len(values) + 1)
    ax.plot(folds, values, "o-", linewidth=2, markersize=8)
    ax.axhline(
        y=np.mean(values),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(values):.3f}",
    )

    ax.set_xlabel("CV Fold", fontsize=11)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Training history plot saved to {output_path}")


def create_figure_grid(
    figures: List[Path],
    output_path: Path,
    n_cols: int = 2,
    figsize: Tuple[int, int] = (16, 12),
) -> None:
    """
    Combine multiple figures into a grid.

    Parameters
    ----------
    figures : List[Path]
        Paths to figures to combine
    output_path : Path
        Output path for combined figure
    n_cols : int
        Number of columns
    figsize : Tuple[int, int]
        Figure size
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error(
            "PIL (Pillow) is required for figure grids. Install with: pip install Pillow"
        )
        raise

    if not figures:
        logger.warning("No figures to combine")
        return

    # Load images
    images = [Image.open(f) for f in figures if Path(f).exists()]

    if not images:
        logger.warning("No valid figures found")
        return

    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols

    # Get size of first image
    img_width, img_height = images[0].size

    # Create grid
    grid_width = img_width * n_cols
    grid_height = img_height * n_rows

    grid = Image.new("RGB", (grid_width, grid_height), "white")

    # Paste images
    for i, img in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, dpi=(150, 150))

    logger.info(f"Figure grid saved to {output_path}")
