"""
Abstract base classes for TESLearn.

This module provides the foundation for extensibility through abstract base classes
that define interfaces for feature extraction, selection, models, and validation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
import numpy as np
from pathlib import Path


class NiftiImageLike(Protocol):
    """Protocol for nibabel NIfTI image-like objects."""

    @property
    def shape(self) -> Tuple[int, ...]: ...

    @property
    def affine(self) -> np.ndarray: ...

    @property
    def dataobj(self) -> np.ndarray: ...


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extraction from E-field images.

    Feature extractors transform 3D E-field intensity maps into feature vectors
    suitable for machine learning.

    Examples
    --------
    >>> extractor = AtlasFeatureExtractor(atlas_path='path/to/atlas.nii.gz')
    >>> features = extractor.transform(efield_images)
    """

    def __init__(self, **kwargs):
        self._is_fitted = False
        self.feature_names_: Optional[List[str]] = None

    @abstractmethod
    def fit(
        self, images: List[NiftiImageLike], y: Optional[np.ndarray] = None
    ) -> "BaseFeatureExtractor":
        """
        Fit the extractor to the data.

        Parameters
        ----------
        images : List[NiftiImageLike]
            List of NIfTI images containing E-field intensity values
        y : Optional[np.ndarray]
            Target values (may be used for supervised feature extraction)

        Returns
        -------
        self : BaseFeatureExtractor
            Fitted extractor
        """
        pass

    @abstractmethod
    def transform(self, images: List[NiftiImageLike]) -> np.ndarray:
        """
        Transform images to feature matrix.

        Parameters
        ----------
        images : List[NiftiImageLike]
            List of NIfTI images

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        """
        pass

    def fit_transform(
        self, images: List[NiftiImageLike], y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(images, y).transform(images)

    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        if self.feature_names_ is None:
            raise RuntimeError(
                "Feature extractor must be fitted before getting feature names"
            )
        return self.feature_names_

    @property
    def n_features(self) -> int:
        """Number of features extracted."""
        if self.feature_names_ is None:
            return 0
        return len(self.feature_names_)


class BaseFeatureSelector(ABC):
    """
    Abstract base class for feature selection.

    Feature selectors reduce the dimensionality of feature matrices by selecting
    the most informative features based on various criteria.
    """

    def __init__(self, **kwargs):
        self._is_fitted = False
        self.selected_indices_: Optional[np.ndarray] = None
        self.selected_feature_names_: Optional[List[str]] = None
        self.support_: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseFeatureSelector":
        """
        Fit the selector to the data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target vector of shape (n_samples,)

        Returns
        -------
        self : BaseFeatureSelector
            Fitted selector
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Select features from X.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        X_selected : np.ndarray
            Feature matrix with only selected features
        """
        self._check_is_fitted()
        if self.selected_indices_ is None:
            raise RuntimeError("No features selected")
        return X[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _check_is_fitted(self):
        """Check if selector has been fitted."""
        if not self._is_fitted:
            raise RuntimeError("Feature selector must be fitted before transform")

    def get_support(self) -> np.ndarray:
        """Get boolean mask of selected features."""
        self._check_is_fitted()
        return self.support_


class BaseModel(ABC):
    """
    Abstract base class for ML models.

    Wraps scikit-learn compatible estimators with a consistent interface
    for training, prediction, and model introspection.
    """

    def __init__(self, **kwargs):
        self._is_fitted = False
        self.model_: Optional[Any] = None
        self.classes_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """
        Fit the model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector

        Returns
        -------
        self : BaseModel
            Fitted model
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        y_pred : np.ndarray
            Predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        proba : np.ndarray
            Class probabilities of shape (n_samples, n_classes)
        """
        pass

    def _check_is_fitted(self):
        """Check if model has been fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {}

    def set_params(self, **params) -> "BaseModel":
        """Set model parameters."""
        return self

    @property
    def is_classifier(self) -> bool:
        """Whether this is a classification model."""
        return True


class BaseValidator(ABC):
    """
    Abstract base class for cross-validation strategies.

    Validators split data into training and validation folds while ensuring
    proper stratification for classification tasks.
    """

    def __init__(self, n_splits: int = 5, random_state: Optional[int] = None, **kwargs):
        self.n_splits = n_splits
        self.random_state = random_state

    @abstractmethod
    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate train/test indices.

        Yields
        ------
        train_index : np.ndarray
            Training indices
        test_index : np.ndarray
            Test indices
        """
        pass

    def get_n_splits(
        self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None
    ) -> int:
        """Get number of splits."""
        return self.n_splits


class BaseExplainer(ABC):
    """
    Abstract base class for model explanation/explainability.

    Explainers provide insights into model predictions through feature
    importance, weight maps, and diagnostic plots.
    """

    def __init__(self, model: BaseModel, **kwargs):
        self.model = model
        self.feature_names: Optional[List[str]] = None

    @abstractmethod
    def explain(
        self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate model explanation.

        Parameters
        ----------
        X : Optional[np.ndarray]
            Feature matrix (for local explanations)
        y : Optional[np.ndarray]
            Target values

        Returns
        -------
        explanation : Dict[str, Any]
            Dictionary containing explanation artifacts
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.

        Returns
        -------
        importance : np.ndarray
            Feature importance scores
        """
        pass
