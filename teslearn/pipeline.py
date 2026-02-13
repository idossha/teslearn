"""
ML Pipeline for TESLearn.

Integrates feature extraction, selection, scaling, and modeling into a
unified pipeline with a scikit-learn compatible interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import logging

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler

from .base import BaseFeatureExtractor, BaseFeatureSelector, BaseModel

logger = logging.getLogger(__name__)


class TESPipeline(BaseEstimator):
    """
    End-to-end pipeline for TES responder prediction.

    Combines feature extraction, optional feature selection, standardization,
    and classification/regression into a single pipeline.

    Parameters
    ----------
    feature_extractor : BaseFeatureExtractor
        Extractor to transform images to features
    model : BaseModel
        Classification or regression model
    feature_selector : Optional[BaseFeatureSelector]
        Optional feature selection step
    use_scaling : bool
        Whether to standardize features (default: True)

    Examples
    --------
    >>> from teslearn.features import AtlasFeatureExtractor
    >>> from teslearn.models import LogisticRegressionModel
    >>> from teslearn.selection import TTestSelector

    >>> # Build pipeline
    >>> extractor = AtlasFeatureExtractor(atlas_path='atlas.nii.gz')
    >>> selector = TTestSelector(p_threshold=0.001)
    >>> model = LogisticRegressionModel(C=1.0)

    >>> pipeline = TESPipeline(
    ...     feature_extractor=extractor,
    ...     feature_selector=selector,
    ...     model=model,
    ... )

    >>> # Train
    >>> pipeline.fit(images, y)

    >>> # Predict
    >>> proba = pipeline.predict_proba(test_images)

    >>> # Get feature importance
    >>> importance = pipeline.get_feature_importance()
    """

    def __init__(
        self,
        feature_extractor: BaseFeatureExtractor,
        model: BaseModel,
        feature_selector: Optional[BaseFeatureSelector] = None,
        use_scaling: bool = True,
    ):
        self.feature_extractor = feature_extractor
        self.model = model
        self.feature_selector = feature_selector
        self.use_scaling = use_scaling

        self._scaler: Optional[StandardScaler] = None
        self._is_fitted = False
        self.feature_names_: Optional[List[str]] = None

    def fit(
        self,
        images: List[Any],
        y: np.ndarray,
    ) -> "TESPipeline":
        """
        Fit the entire pipeline.

        Parameters
        ----------
        images : List[NiftiImageLike]
            Training images
        y : np.ndarray
            Training targets

        Returns
        -------
        self : TESPipeline
            Fitted pipeline
        """
        logger.info("Fitting TESPipeline...")

        # Step 1: Extract features
        logger.info("Extracting features...")
        X = self.feature_extractor.fit_transform(images, y)
        self.feature_names_ = self.feature_extractor.get_feature_names()
        logger.info(f"Extracted {X.shape[1]} features")

        # Step 2: Feature selection
        if self.feature_selector is not None:
            logger.info("Selecting features...")
            X = self.feature_selector.fit_transform(X, y)
            self.feature_names_ = [
                self.feature_names_[i] for i in self.feature_selector.selected_indices_
            ]
            logger.info(f"Selected {X.shape[1]} features")

        # Step 3: Scaling
        if self.use_scaling:
            logger.info("Standardizing features...")
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        # Step 4: Fit model
        logger.info("Training model...")
        self.model.fit(X, y)

        self._is_fitted = True
        logger.info("Pipeline fitted successfully")

        return self

    def transform(self, images: List[Any]) -> np.ndarray:
        """
        Transform images to features (without prediction).

        Parameters
        ----------
        images : List[NiftiImageLike]
            Input images

        Returns
        -------
        X : np.ndarray
            Transformed features
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        # Extract features
        X = self.feature_extractor.transform(images)

        # Feature selection
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)

        # Scaling
        if self._scaler is not None:
            X = self._scaler.transform(X)

        return X

    def predict(self, images: List[Any]) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        images : List[NiftiImageLike]
            Input images

        Returns
        -------
        predictions : np.ndarray
            Class predictions (or continuous predictions for regression)
        """
        X = self.transform(images)
        return self.model.predict(X)

    def predict_proba(self, images: List[Any]) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        images : List[NiftiImageLike]
            Input images

        Returns
        -------
        probabilities : np.ndarray
            Class probabilities
        """
        X = self.transform(images)
        return self.model.predict_proba(X)

    def score(self, images: List[Any], y: np.ndarray) -> float:
        """
        Compute accuracy score (classification) or R^2 (regression).

        Parameters
        ----------
        images : List[NiftiImageLike]
            Test images
        y : np.ndarray
            True targets

        Returns
        -------
        score : float
            Performance score
        """
        from sklearn.metrics import accuracy_score, r2_score

        predictions = self.predict(images)

        if self.model.is_classifier:
            return accuracy_score(y, predictions)
        else:
            return r2_score(y, predictions)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance as a dictionary.

        Returns
        -------
        importance : Dict[str, float]
            Feature names mapped to importance scores
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted first")

        importance = {}

        # Get coefficients or feature importance from model
        if hasattr(self.model, "coef_") and self.model.coef_ is not None:
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = coef[0]
            for name, value in zip(self.feature_names_, coef):
                importance[name] = float(value)
        elif hasattr(self.model, "get_feature_importance"):
            # For models like RandomForest
            fi = self.model.get_feature_importance()
            for name, value in zip(self.feature_names_, fi):
                importance[name] = float(value)
        else:
            logger.warning("Model does not provide feature importance")

        return importance

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return {
            "feature_extractor": self.feature_extractor,
            "model": self.model,
            "feature_selector": self.feature_selector,
            "use_scaling": self.use_scaling,
        }

    def set_params(self, **params) -> "TESPipeline":
        """Set pipeline parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        self._is_fitted = False
        return self

    @property
    def is_classifier(self) -> bool:
        """Whether this is a classification pipeline."""
        return self.model.is_classifier

    def __sklearn_is_fitted__(self):
        """Check if fitted (for sklearn compatibility)."""
        return self._is_fitted
