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
    output_path: Optional[Path] = None,
    top_k: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Feature Importance",
    color_positive: str = "red",
    color_negative: str = "blue",
) -> Any:
    """
    Plot feature importance as a horizontal bar chart.

    Parameters
    ----------
    importance : Dict[str, float]
        Feature names mapped to importance values
    output_path : Optional[Path]
        Output path for figure. If None, figure is not saved.
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

    Returns
    -------
    fig : Any
        The matplotlib figure object for further manipulation or display
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

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
    legend_elements = [
        Patch(facecolor=color_positive, alpha=0.7, label="Positive (responders)"),
        Patch(facecolor=color_negative, alpha=0.7, label="Negative (non-responders)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {output_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 8),
) -> Any:
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Predicted scores/probabilities
    output_path : Optional[Path]
        Output path for figure. If None, figure is not saved.
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    fig : Any
        The matplotlib figure object for further manipulation or display
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

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

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved to {output_path}")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 8),
) -> Any:
    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Predicted scores/probabilities
    output_path : Optional[Path]
        Output path for figure. If None, figure is not saved.
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    fig : Any
        The matplotlib figure object for further manipulation or display
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

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

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Precision-Recall curve saved to {output_path}")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Path] = None,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
) -> Any:
    """
    Plot confusion matrix as a heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    output_path : Optional[Path]
        Output path for figure. If None, figure is not saved.
    labels : Optional[List[str]]
        Class labels
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    cmap : str
        Colormap name

    Returns
    -------
    fig : Any
        The matplotlib figure object for further manipulation or display
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

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

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {output_path}")

    return fig


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Prediction Distribution",
    figsize: Tuple[int, int] = (10, 6),
) -> Any:
    """
    Plot distribution of predicted probabilities by true class.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Predicted probabilities
    output_path : Optional[Path]
        Output path for figure. If None, figure is not saved.
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    fig : Any
        The matplotlib figure object for further manipulation or display
    """
    import matplotlib.pyplot as plt

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

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Prediction distribution plot saved to {output_path}")

    return fig


def plot_training_history(
    cv_results: List[Any],
    output_path: Optional[Path] = None,
    metric: str = "roc_auc",
    title: str = "Cross-Validation Performance",
    figsize: Tuple[int, int] = (10, 6),
) -> Any:
    """
    Plot CV performance across folds.

    Parameters
    ----------
    cv_results : List[CVResult]
        Cross-validation results
    output_path : Optional[Path]
        Output path for figure. If None, figure is not saved.
    metric : str
        Metric to plot ('roc_auc', 'accuracy', 'r2', etc.)
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    fig : Any
        The matplotlib figure object for further manipulation or display
    """
    import matplotlib.pyplot as plt

    # Extract metric values
    values = []
    for result in cv_results:
        val = getattr(result, metric, None)
        if val is not None:
            values.append(val)

    if not values:
        logger.warning(f"No values found for metric: {metric}")
        return None

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

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training history plot saved to {output_path}")

    return fig


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


def plot_intensity_response(
    voxel_values_responder: np.ndarray,
    voxel_values_non_responder: np.ndarray,
    subj_median_intensity: np.ndarray,
    subj_behavior: np.ndarray,
    subj_label: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Intensity vs Response",
    figsize: Tuple[int, int] = (14, 6),
    labels: Tuple[str, str] = ("Non-responder", "Responder"),
) -> Any:
    """Plot intensity-response figure (Albizu Fig.5 style).

    Creates a two-panel figure:
    1. Left panel: Histogram of voxel intensity distributions comparing
       responders vs non-responders
    2. Right panel: Scatter plot of behavioral response vs median intensity

    Parameters
    ----------
    voxel_values_responder : np.ndarray
        Voxel intensity values from responder subjects (all voxels concatenated)
    voxel_values_non_responder : np.ndarray
        Voxel intensity values from non-responder subjects (all voxels concatenated)
    subj_median_intensity : np.ndarray
        Median intensity per subject (n_subjects,)
    subj_behavior : np.ndarray
        Behavioral response value per subject (n_subjects,) - for classification
        this is typically predicted probability; for regression it's the target
    subj_label : np.ndarray
        Binary labels per subject (n_subjects,) - 0 for non-responder, 1 for responder
    output_path : Optional[Path]
        Output path for the figure. If None, figure is not saved.
    title : str
        Overall figure title
    figsize : Tuple[int, int]
        Figure size (width, height)
    labels : Tuple[str, str]
        Labels for the two groups (non-responder, responder)

    Returns
    -------
    fig : Any
        The matplotlib figure object for further manipulation or display
    """
    import matplotlib.pyplot as plt

    # Filter out non-finite values
    voxel_r = voxel_values_responder[np.isfinite(voxel_values_responder)]
    voxel_n = voxel_values_non_responder[np.isfinite(voxel_values_non_responder)]

    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Left panel: Intensity histogram ---
    bins = np.linspace(
        min(voxel_r.min() if len(voxel_r) else 0, voxel_n.min() if len(voxel_n) else 0),
        max(voxel_r.max() if len(voxel_r) else 1, voxel_n.max() if len(voxel_n) else 1),
        50,
    )

    # Compute histograms manually for better control (matching reference implementation)
    def _compute_hist(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts, _ = np.histogram(values, bins=bins)
        total = float(counts.sum())
        if total <= 0:
            pm = np.zeros_like(counts, dtype=float)
        else:
            pm = counts.astype(float) / total
        return pm, 0.5 * (bins[:-1] + bins[1:])

    pm_n, centers = _compute_hist(voxel_n) if len(voxel_n) > 0 else (None, None)
    pm_r, _ = _compute_hist(voxel_r) if len(voxel_r) > 0 else (None, None)

    if pm_n is not None:
        ax1.step(
            centers, pm_n, where="mid", color="#1f77b4", linewidth=2, label=labels[0]
        )
        ax1.fill_between(centers, 0, pm_n, step="mid", color="#1f77b4", alpha=0.25)
    if pm_r is not None:
        ax1.step(
            centers, pm_r, where="mid", color="#d62728", linewidth=2, label=labels[1]
        )
        ax1.fill_between(centers, 0, pm_r, step="mid", color="#d62728", alpha=0.25)

    ax1.set_xlabel("E-field Intensity (V/m)", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.set_title("Intensity Distribution", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # --- Right panel: Response vs median intensity ---
    mask_r = subj_label == 1
    mask_n = subj_label == 0

    if np.any(mask_n):
        ax2.scatter(
            subj_median_intensity[mask_n],
            subj_behavior[mask_n],
            c="blue",
            alpha=0.6,
            s=80,
            edgecolors="black",
            linewidths=0.5,
            label=labels[0],
        )
    if np.any(mask_r):
        ax2.scatter(
            subj_median_intensity[mask_r],
            subj_behavior[mask_r],
            c="red",
            alpha=0.6,
            s=80,
            edgecolors="black",
            linewidths=0.5,
            label=labels[1],
        )

    ax2.set_xlabel("Median E-field Intensity (V/m)", fontsize=11)
    ax2.set_ylabel("Behavioral Response", fontsize=11)
    ax2.set_title("Response vs Median Intensity", fontsize=12, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Intensity-response plot saved to {output_path}")

    return fig


def prepare_intensity_response_data(
    images: List[Any],
    y_true: np.ndarray,
    y_behavior: Optional[np.ndarray] = None,
    *,
    feature_mask: Optional[np.ndarray] = None,
    voxel_coords: Optional[List[Tuple[int, int, int]]] = None,
    atlas_path: Optional[Union[str, Path]] = None,
    roi_ids: Optional[List[int]] = None,
    resample_to_first: bool = True,
) -> Dict[str, np.ndarray]:
    """Prepare data for intensity-response visualization.

    Generic helper that works with both voxel-based and atlas ROI-based features.
    Extracts voxel intensities from selected regions/voxels for all subjects,
    then computes the arrays needed for plot_intensity_response().

    Parameters
    ----------
    images : List[Any]
        List of NIfTI images (one per subject)
    y_true : np.ndarray
        True binary labels (n_subjects,) - 0 for non-responder, 1 for responder
    y_behavior : Optional[np.ndarray]
        Behavioral response values (n_subjects,). If None, uses y_true.
        For classification, this is typically predicted probabilities.
        For regression, this is the target value.
    feature_mask : Optional[np.ndarray]
        Boolean 3D array same shape as images indicating which voxels to include.
        If provided, used directly to select voxels.
    voxel_coords : Optional[List[Tuple[int, int, int]]]
        List of (x, y, z) coordinates for voxel-based features.
        Alternative to feature_mask. Coordinates are in the space of the first image.
    atlas_path : Optional[Union[str, Path]]
        Path to atlas NIfTI for ROI-based features.
    roi_ids : Optional[List[int]]
        List of ROI IDs to include. Used with atlas_path.
    resample_to_first : bool
        If True, resample all images to match the first image's space.
        Set to False if images are already aligned.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys:
        - 'voxel_values_responder': All voxel intensities from responder subjects
        - 'voxel_values_non_responder': All voxel intensities from non-responder subjects
        - 'subj_median_intensity': Median intensity per subject
        - 'subj_behavior': Behavioral response per subject
        - 'subj_label': Binary label per subject

    Examples
    --------
    >>> # Voxel-based approach (e.g., from TTestSelector on voxels)
    >>> voxel_coords = [(10, 20, 30), (15, 25, 35), ...]  # Selected voxels
    >>> data = prepare_intensity_response_data(
    ...     images=train_images,
    ...     y_true=y_train,
    ...     y_behavior=y_proba,  # predicted probabilities
    ...     voxel_coords=voxel_coords,
    ... )

    >>> # Atlas ROI-based approach
    >>> data = prepare_intensity_response_data(
    ...     images=train_images,
    ...     y_true=y_train,
    ...     y_behavior=y_proba,
    ...     atlas_path="atlas.nii.gz",
    ...     roi_ids=[8, 9, 10, 11],  # Selected ROIs
    ... )
    """
    import nibabel as nib
    import nilearn.image as nii_img

    if len(images) != len(y_true):
        raise ValueError(
            f"Number of images ({len(images)}) must match y_true length ({len(y_true)})"
        )

    if y_behavior is None:
        y_behavior = y_true.astype(float)

    # Determine which voxels to extract
    if feature_mask is not None:
        # Use provided mask directly
        mask = feature_mask
        ref_img = images[0]
    elif voxel_coords is not None:
        # Create mask from voxel coordinates
        ref_img = images[0]
        ref_data = np.asanyarray(ref_img.dataobj)
        if ref_data.ndim == 4:
            ref_data = ref_data[:, :, :, 0]
        shape = ref_data.shape[:3]
        mask = np.zeros(shape, dtype=bool)
        for x, y, z in voxel_coords:
            if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                mask[x, y, z] = True
    elif atlas_path is not None and roi_ids is not None:
        # Create mask from atlas ROIs
        atlas_img = nib.load(str(atlas_path))
        atlas_data = np.asanyarray(atlas_img.dataobj).astype(np.int32)
        if atlas_data.ndim == 4:
            atlas_data = atlas_data[:, :, :, 0]
        mask = np.isin(atlas_data, np.asarray(roi_ids, dtype=np.int32))
        ref_img = atlas_img
    else:
        raise ValueError(
            "Must provide one of: feature_mask, voxel_coords, or (atlas_path + roi_ids)"
        )

    flat_idx = np.flatnonzero(mask.ravel())
    if flat_idx.size == 0:
        raise ValueError("No voxels selected by the provided mask/coordinates/ROIs")

    # Extract intensities from each subject
    voxel_r = []
    voxel_n = []
    subj_median_intensity = []

    for img, label in zip(images, y_true):
        if resample_to_first:
            aligned = nii_img.resample_to_img(img, ref_img, interpolation="continuous")
        else:
            aligned = img
        data = np.asanyarray(aligned.dataobj)
        if data.ndim == 4:
            data = data[:, :, :, 0]
        data_flat = data.astype(np.float32).ravel()
        vals = data_flat[flat_idx]
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            subj_median_intensity.append(np.nan)
        else:
            subj_median_intensity.append(float(np.median(vals)))
            if label == 1:
                voxel_r.append(vals)
            else:
                voxel_n.append(vals)

    return {
        "voxel_values_responder": np.concatenate(voxel_r)
        if voxel_r
        else np.array([], dtype=float),
        "voxel_values_non_responder": np.concatenate(voxel_n)
        if voxel_n
        else np.array([], dtype=float),
        "subj_median_intensity": np.asarray(subj_median_intensity, dtype=float),
        "subj_behavior": np.asarray(y_behavior, dtype=float),
        "subj_label": np.asarray(y_true, dtype=int),
    }


def plot_intensity_response_from_pipeline(
    images: List[Any],
    pipeline: Any,
    y_true: np.ndarray,
    output_path: Optional[Path] = None,
    y_behavior: Optional[np.ndarray] = None,
    atlas_path: Optional[Union[str, Path]] = None,
    title: str = "Intensity vs Response",
    figsize: Tuple[int, int] = (14, 6),
    labels: Tuple[str, str] = ("Non-responder", "Responder"),
) -> Any:
    """Create intensity-response plot from a fitted pipeline.

    Generic function that automatically detects feature type (voxel vs atlas ROI)
    from the pipeline's feature names and extracts appropriate voxel intensities.

    Parameters
    ----------
    images : List[Any]
        List of NIfTI images (one per subject)
    pipeline : Any
        Fitted TESPipeline with feature_extractor that has get_feature_names()
    y_true : np.ndarray
        True binary labels
    output_path : Optional[Path]
        Output path for the figure. If None, figure is not saved.
    y_behavior : Optional[np.ndarray]
        Behavioral response values. If None, uses predicted probabilities for
        classification or y_true for regression.
    atlas_path : Optional[Union[str, Path]]
        Path to atlas NIfTI. Required if pipeline uses atlas ROI features.
        Not needed for voxel-based features.
    title : str
        Figure title
    figsize : Tuple[int, int]
        Figure size
    labels : Tuple[str, str]
        Labels for the two groups

    Returns
    -------
    fig : Any
        The matplotlib figure object for further manipulation or display

    Examples
    --------
    >>> # Works with both voxel-based and atlas ROI-based pipelines
    >>> teslearn.plot_intensity_response_from_pipeline(
    ...     images=train_images,
    ...     pipeline=fitted_pipeline,
    ...     y_true=y_train,
    ...     y_behavior=y_proba,
    ...     output_path="intensity_response.png",
    ...     atlas_path="atlas.nii.gz",  # Required for atlas features
    ... )
    """
    # Get feature names from pipeline
    if not hasattr(pipeline, "feature_extractor"):
        raise ValueError("Pipeline must have a feature_extractor attribute")

    extractor = pipeline.feature_extractor
    if not hasattr(extractor, "get_feature_names"):
        raise ValueError("Feature extractor must have get_feature_names() method")

    feature_names = extractor.get_feature_names()

    # Detect feature type and prepare data
    if not feature_names:
        raise ValueError("No feature names found in pipeline")

    # Check if voxel-based features
    is_voxel = any(name.startswith("voxel_") for name in feature_names[:10])
    # Check if atlas ROI features
    is_atlas_roi = any("ROI_" in name and "__" in name for name in feature_names[:10])

    if is_voxel:
        # Parse voxel coordinates from feature names
        voxel_coords = []
        for name in feature_names:
            if name.startswith("voxel_"):
                parts = name.split("_")
                if len(parts) == 4:
                    try:
                        x, y, z = int(parts[1]), int(parts[2]), int(parts[3])
                        voxel_coords.append((x, y, z))
                    except ValueError:
                        continue
        if not voxel_coords:
            raise ValueError("Could not parse voxel coordinates from feature names")
        data = prepare_intensity_response_data(
            images=images,
            y_true=y_true,
            y_behavior=y_behavior,
            voxel_coords=voxel_coords,
        )
    elif is_atlas_roi:
        # Parse ROI IDs from feature names
        if atlas_path is None:
            raise ValueError(
                "atlas_path is required for atlas ROI features. "
                "Please provide the path to the atlas NIfTI file."
            )
        roi_ids = set()
        for name in feature_names:
            if name.startswith("ROI_") and "__" in name:
                left, _ = name.split("__", 1)
                try:
                    roi_id = int(left.replace("ROI_", ""))
                    roi_ids.add(roi_id)
                except ValueError:
                    continue
        if not roi_ids:
            raise ValueError("Could not parse ROI IDs from feature names")
        data = prepare_intensity_response_data(
            images=images,
            y_true=y_true,
            y_behavior=y_behavior,
            atlas_path=atlas_path,
            roi_ids=list(roi_ids),
        )
    else:
        raise ValueError(
            f"Could not detect feature type from feature names. "
            f"Expected voxel_X_Y_Z or ROI_X__stat patterns. Got: {feature_names[:5]}"
        )

    # Create the plot
    return plot_intensity_response(
        voxel_values_responder=data["voxel_values_responder"],
        voxel_values_non_responder=data["voxel_values_non_responder"],
        subj_median_intensity=data["subj_median_intensity"],
        subj_behavior=data["subj_behavior"],
        subj_label=data["subj_label"],
        output_path=output_path,
        title=title,
        figsize=figsize,
        labels=labels,
    )
