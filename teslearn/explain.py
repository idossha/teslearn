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
