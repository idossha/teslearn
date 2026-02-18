"""
Model explanation and interpretability for TESLearn.

Provides tools for understanding model predictions through feature importance,
weight maps, and diagnostic visualizations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging

from .base import BaseExplainer, BaseModel
from .pipeline import TESPipeline

logger = logging.getLogger(__name__)


@dataclass
class ROIRankingResult:
    """Result from ROI importance ranking."""

    roi_rankings: List[Tuple[int, str, float]]  # (roi_id, roi_name, importance)
    top_responder_rois: List[Tuple[int, str, float]]  # Positive weights
    top_non_responder_rois: List[Tuple[int, str, float]]  # Negative weights
    statistics_breakdown: Dict[str, List[Tuple[int, str, float]]]  # Per-stat breakdown

    def get_summary(self, top_k: int = 10) -> str:
        """Get text summary of ROI rankings."""
        lines = [
            "=" * 70,
            "ROI Importance Ranking (Atlas-based)",
            "=" * 70,
            f"\nTop {top_k} ROIs Predicting Responders (positive weights):",
        ]

        for roi_id, roi_name, weight in self.top_responder_rois[:top_k]:
            lines.append(f"  {roi_id:3d}: {roi_name:30s} | weight = {weight:+.4f}")

        lines.append(
            f"\nTop {top_k} ROIs Predicting Non-Responders (negative weights):"
        )
        for roi_id, roi_name, weight in self.top_non_responder_rois[:top_k]:
            lines.append(f"  {roi_id:3d}: {roi_name:30s} | weight = {weight:+.4f}")

        # Add per-statistic breakdown if available
        if self.statistics_breakdown:
            lines.append("\n" + "-" * 70)
            lines.append("Breakdown by Feature Statistic:")
            lines.append("-" * 70)
            for stat, rankings in self.statistics_breakdown.items():
                lines.append(f"\n{stat.upper()}:")
                for roi_id, roi_name, weight in rankings[:5]:
                    lines.append(f"  {roi_id:3d}: {roi_name:25s} | {weight:+.4f}")

        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class ExplanationResult:
    """Result from model explanation."""

    feature_importance: Dict[str, float]
    top_positive: List[Tuple[str, float]]
    top_negative: List[Tuple[str, float]]
    weight_maps: Optional[List[Path]] = None
    figures: List[Path] = field(default_factory=list)

    def get_summary(self) -> str:
        """Get text summary of explanation."""
        lines = [
            "=" * 60,
            "Model Explanation Summary",
            "=" * 60,
            f"\nTop 10 Positive Features (responders):",
        ]

        for feat, imp in self.top_positive[:10]:
            lines.append(f"  {feat}: {imp:.4f}")

        lines.append(f"\nTop 10 Negative Features (non-responders):")
        for feat, imp in self.top_negative[:10]:
            lines.append(f"  {feat}: {imp:.4f}")

        lines.append("=" * 60)

        return "\n".join(lines)


class ModelExplainer(BaseExplainer):
    """
    Explain TESLearn models.

    Provides feature importance analysis and visualization for linear models.

    Parameters
    ----------
    pipeline : TESPipeline
        Fitted pipeline to explain
    atlas_path : Optional[Path]
        Path to atlas for creating weight maps

    Examples
    --------
    >>> explainer = ModelExplainer(pipeline, atlas_path='atlas.nii.gz')
    >>> explanation = explainer.explain()
    >>> print(explanation.get_summary())
    """

    def __init__(
        self,
        pipeline: TESPipeline,
        atlas_path: Optional[Path] = None,
    ):
        super().__init__(pipeline.model)
        self.pipeline = pipeline
        self.atlas_path = atlas_path
        self.feature_names = pipeline.feature_names_

    def explain(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> ExplanationResult:
        """
        Generate model explanation.

        Parameters
        ----------
        X : Optional[np.ndarray]
            Features (not used for global explanations)
        y : Optional[np.ndarray]
            Targets (not used for global explanations)

        Returns
        -------
        explanation : ExplanationResult
            Explanation results
        """
        # Get feature importance
        importance = self.pipeline.get_feature_importance()

        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Split into positive and negative
        positive = [(f, v) for f, v in sorted_features if v > 0]
        negative = [(f, v) for f, v in sorted_features if v < 0]

        return ExplanationResult(
            feature_importance=importance,
            top_positive=positive,
            top_negative=negative,
        )

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance as numpy array."""
        importance = self.pipeline.get_feature_importance()
        return np.array([importance.get(f, 0.0) for f in self.feature_names])

    def create_weight_map(
        self,
        output_path: Path,
        atlas_path: Optional[Path] = None,
    ) -> Path:
        """
        Create a NIfTI weight map from model coefficients.

        For atlas-based features, creates a map where each ROI is filled
        with its corresponding coefficient.

        Parameters
        ----------
        output_path : Path
            Output path for weight map
        atlas_path : Optional[Path]
            Atlas path (uses self.atlas_path if None)

        Returns
        -------
        output_path : Path
            Path to saved weight map
        """
        import nibabel as nib

        atlas_path = atlas_path or self.atlas_path
        if atlas_path is None:
            raise ValueError("Atlas path required to create weight map")

        atlas_path = Path(atlas_path)
        if not atlas_path.exists():
            raise FileNotFoundError(f"Atlas not found: {atlas_path}")

        # Load atlas
        atlas_img = nib.load(str(atlas_path))
        atlas_data = np.asanyarray(atlas_img.dataobj).astype(np.int32)

        # Create weight volume
        weight_data = np.zeros_like(atlas_data, dtype=np.float32)

        # Parse feature names and assign weights
        importance = self.pipeline.get_feature_importance()

        for feature_name, weight in importance.items():
            # Parse ROI ID from feature name
            # Format: ROI_<id>__<stat> or voxel_X_Y_Z
            if feature_name.startswith("ROI_") and "__" in feature_name:
                parts = feature_name.split("__")
                roi_part = parts[0]
                try:
                    roi_id = int(roi_part.replace("ROI_", ""))
                    # Fill ROI with weight
                    weight_data[atlas_data == roi_id] = weight
                except ValueError:
                    logger.warning(f"Could not parse ROI from {feature_name}")

        # Save weight map
        weight_img = nib.Nifti1Image(weight_data, atlas_img.affine, atlas_img.header)
        nib.save(weight_img, str(output_path))

        logger.info(f"Weight map saved to {output_path}")

        return output_path

    def plot_feature_importance(
        self,
        output_path: Path,
        top_k: int = 20,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Path:
        """
        Plot feature importance.

        Parameters
        ----------
        output_path : Path
            Output path for figure
        top_k : int
            Number of top features to show
        figsize : Tuple[int, int]
            Figure size

        Returns
        -------
        output_path : Path
            Path to saved figure
        """
        from .plotting import plot_feature_importance as plot_imp

        importance = self.pipeline.get_feature_importance()

        plot_imp(
            importance=importance,
            output_path=output_path,
            top_k=top_k,
            figsize=figsize,
            title="Feature Importance (Model Coefficients)",
        )

        return output_path

    def rank_roi_importance(
        self,
        atlas_labels: Optional[Dict[int, str]] = None,
        aggregate_statistics: bool = True,
    ) -> ROIRankingResult:
        """
        Rank atlas ROIs by importance based on model weights.

        This aggregates feature weights across different statistics (mean, max, etc.)
        to provide an ROI-level interpretation of model importance.

        Parameters
        ----------
        atlas_labels : Optional[Dict[int, str]]
            Mapping from ROI ID to ROI name. If None, uses "ROI_<id>" format.
        aggregate_statistics : bool
            If True, aggregates weights across statistics per ROI. If False,
            keeps statistics separate in the breakdown.

        Returns
        -------
        roi_ranking : ROIRankingResult
            ROI importance rankings
        """
        importance = self.pipeline.get_feature_importance()

        # Parse ROI features: format "ROI_<id>__<stat>"
        roi_weights: Dict[int, Dict[str, float]] = {}

        for feature_name, weight in importance.items():
            if feature_name.startswith("ROI_") and "__" in feature_name:
                parts = feature_name.split("__")
                roi_part = parts[0]
                stat = parts[1] if len(parts) > 1 else "unknown"

                try:
                    roi_id = int(roi_part.replace("ROI_", ""))
                    if roi_id not in roi_weights:
                        roi_weights[roi_id] = {}
                    roi_weights[roi_id][stat] = weight
                except ValueError:
                    logger.warning(f"Could not parse ROI from {feature_name}")

        if not roi_weights:
            logger.warning("No atlas ROI features found in model")
            return ROIRankingResult(
                roi_rankings=[],
                top_responder_rois=[],
                top_non_responder_rois=[],
                statistics_breakdown={},
            )

        # Aggregate weights per ROI
        roi_importance: Dict[int, float] = {}
        for roi_id, stat_weights in roi_weights.items():
            if aggregate_statistics:
                # Sum absolute weights across statistics for total importance
                roi_importance[roi_id] = sum(stat_weights.values())
            else:
                # Use mean weight
                roi_importance[roi_id] = np.mean(list(stat_weights.values()))

        # Sort by importance
        sorted_rois = sorted(
            roi_importance.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # Build rankings with names
        roi_rankings = []
        for roi_id, weight in sorted_rois:
            roi_name = (
                atlas_labels.get(roi_id, f"ROI_{roi_id}")
                if atlas_labels
                else f"ROI_{roi_id}"
            )
            roi_rankings.append((roi_id, roi_name, weight))

        # Split by sign
        positive = [(r, n, w) for r, n, w in roi_rankings if w > 0]
        negative = [(r, n, w) for r, n, w in roi_rankings if w < 0]

        # Sort positive by weight (descending), negative by absolute weight
        positive_sorted = sorted(positive, key=lambda x: x[2], reverse=True)
        negative_sorted = sorted(negative, key=lambda x: abs(x[2]), reverse=True)

        # Build statistics breakdown
        statistics_breakdown: Dict[str, List[Tuple[int, str, float]]] = {}
        all_stats = set()
        for stat_weights in roi_weights.values():
            all_stats.update(stat_weights.keys())

        for stat in sorted(all_stats):
            stat_rankings = []
            for roi_id, stat_weights in roi_weights.items():
                if stat in stat_weights:
                    weight = stat_weights[stat]
                    roi_name = (
                        atlas_labels.get(roi_id, f"ROI_{roi_id}")
                        if atlas_labels
                        else f"ROI_{roi_id}"
                    )
                    stat_rankings.append((roi_id, roi_name, weight))
            # Sort by absolute weight
            stat_rankings.sort(key=lambda x: abs(x[2]), reverse=True)
            statistics_breakdown[stat] = stat_rankings

        return ROIRankingResult(
            roi_rankings=roi_rankings,
            top_responder_rois=positive_sorted,
            top_non_responder_rois=negative_sorted,
            statistics_breakdown=statistics_breakdown,
        )

    def plot_roi_rankings(
        self,
        output_path: Path,
        atlas_labels: Optional[Dict[int, str]] = None,
        top_k: int = 15,
        figsize: Tuple[int, int] = (12, 10),
    ) -> Path:
        """
        Plot ROI importance rankings.

        Creates a bar chart showing top ROIs for responders and non-responders.

        Parameters
        ----------
        output_path : Path
            Output path for figure
        atlas_labels : Optional[Dict[int, str]]
            Mapping from ROI ID to ROI name
        top_k : int
            Number of top ROIs to show per group
        figsize : Tuple[int, int]
            Figure size

        Returns
        -------
        output_path : Path
            Path to saved figure
        """
        import matplotlib.pyplot as plt

        roi_ranking = self.rank_roi_importance(atlas_labels=atlas_labels)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Top responder ROIs
        responder_rois = roi_ranking.top_responder_rois[:top_k]
        if responder_rois:
            labels = [f"{r}: {n[:20]}" for r, n, w in responder_rois]
            values = [w for r, n, w in responder_rois]
            y_pos = np.arange(len(labels))
            ax1.barh(y_pos, values, color="red", alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(labels, fontsize=8)
            ax1.invert_yaxis()
            ax1.set_xlabel("Weight (log-odds)", fontsize=10)
            ax1.set_title(
                f"Top {len(responder_rois)} ROIs Predicting Responders", fontsize=11
            )
            ax1.grid(axis="x", alpha=0.3)
        else:
            ax1.text(
                0.5,
                0.5,
                "No positive ROI weights",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )

        # Top non-responder ROIs
        non_responder_rois = roi_ranking.top_non_responder_rois[:top_k]
        if non_responder_rois:
            labels = [f"{r}: {n[:20]}" for r, n, w in non_responder_rois]
            values = [
                abs(w) for r, n, w in non_responder_rois
            ]  # Use absolute for display
            y_pos = np.arange(len(labels))
            ax2.barh(y_pos, values, color="blue", alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels, fontsize=8)
            ax2.invert_yaxis()
            ax2.set_xlabel("|Weight| (log-odds)", fontsize=10)
            ax2.set_title(
                f"Top {len(non_responder_rois)} ROIs Predicting Non-Responders",
                fontsize=11,
            )
            ax2.grid(axis="x", alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "No negative ROI weights",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"ROI rankings plot saved to {output_path}")
        return output_path


def explain_model(
    pipeline: TESPipeline,
    atlas_path: Optional[Path] = None,
    create_weight_maps: bool = False,
    output_dir: Optional[Path] = None,
) -> ExplanationResult:
    """
    High-level function to explain a model.

    Parameters
    ----------
    pipeline : TESPipeline
        Fitted pipeline
    atlas_path : Optional[Path]
        Path to atlas
    create_weight_maps : bool
        Whether to create NIfTI weight maps
    output_dir : Optional[Path]
        Directory to save outputs

    Returns
    -------
    explanation : ExplanationResult
        Explanation results

    Examples
    --------
    >>> explanation = explain_model(
    ...     pipeline,
    ...     atlas_path='atlas.nii.gz',
    ...     output_dir='./explanations'
    ... )
    >>> print(explanation.get_summary())
    """
    explainer = ModelExplainer(pipeline, atlas_path=atlas_path)
    explanation = explainer.explain()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save feature importance plot
        try:
            fig_path = output_dir / "feature_importance.png"
            explainer.plot_feature_importance(fig_path)
            explanation.figures.append(fig_path)
        except Exception as e:
            logger.warning(f"Could not create feature importance plot: {e}")

        # Create weight maps
        if create_weight_maps and atlas_path:
            try:
                weight_map_path = output_dir / "weight_map.nii.gz"
                explainer.create_weight_map(weight_map_path, atlas_path)
                explanation.weight_maps = [weight_map_path]
            except Exception as e:
                logger.warning(f"Could not create weight map: {e}")

    return explanation
