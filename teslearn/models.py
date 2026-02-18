"""
Machine learning models for TESLearn.

Provides classification and regression models with a consistent interface
for responder prediction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression with L2 regularization (default).

    This is the default model for TESLearn classification tasks, providing
    good baseline performance with interpretable coefficients.

    Parameters
    ----------
    C : float
        Inverse of regularization strength (default: 1.0)
    penalty : str
        Penalty type: 'l1', 'l2', 'elasticnet' (default: 'l2')
    solver : str
        Optimization solver (default: 'lbfgs')
    max_iter : int
        Maximum iterations (default: 1000)
    class_weight : Union[str, Dict, None]
        Class weight strategy (default: 'balanced')
    random_state : Optional[int]
        Random seed for reproducibility
    l1_ratio : Optional[float]
        Elastic-net mixing parameter (only for elasticnet)

    Examples
    --------
    >>> model = LogisticRegressionModel(C=1.0, penalty='l2')
    >>> model.fit(X_train, y_train)
    >>> proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: Union[str, Dict, None] = "balanced",
        random_state: Optional[int] = 42,
        l1_ratio: Optional[float] = None,
        tol: float = 1e-4,
    ):
        super().__init__()
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.l1_ratio = l1_ratio
        self.tol = tol

        self._build_model()

    def _build_model(self):
        """Build the sklearn model."""
        from sklearn.linear_model import LogisticRegression

        # Validate solver/penalty combination
        if self.penalty == "l1" and self.solver not in ["liblinear", "saga"]:
            self.solver = "liblinear"
        elif self.penalty == "elasticnet" and self.solver != "saga":
            self.solver = "saga"

        # Build kwargs based on sklearn version
        # penalty parameter is deprecated in sklearn 1.8+, use l1_ratio instead
        kwargs = {
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "tol": self.tol,
        }

        # Handle penalty/l1_ratio based on sklearn version
        try:
            # Try new API (sklearn >= 1.8) - penalty is deprecated
            import sklearn

            sklearn_version = tuple(map(int, sklearn.__version__.split(".")[:2]))
            if sklearn_version >= (1, 8):
                # New API: use l1_ratio instead of penalty
                if self.penalty == "l1":
                    kwargs["l1_ratio"] = 1.0
                elif self.penalty == "l2":
                    kwargs["l1_ratio"] = 0.0
                elif self.penalty == "elasticnet":
                    kwargs["l1_ratio"] = (
                        self.l1_ratio if self.l1_ratio is not None else 0.5
                    )
                elif self.penalty is None:
                    kwargs["C"] = float("inf")
            else:
                # Old API: use penalty parameter
                kwargs["penalty"] = self.penalty
                if self.penalty == "elasticnet" and self.l1_ratio is not None:
                    kwargs["l1_ratio"] = self.l1_ratio
        except (ImportError, AttributeError):
            # Fallback to old API
            kwargs["penalty"] = self.penalty
            if self.penalty == "elasticnet" and self.l1_ratio is not None:
                kwargs["l1_ratio"] = self.l1_ratio

        self.model_ = LogisticRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        """Fit the model."""
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.coef_ = self.model_.coef_
        self._is_fitted = True
        logger.info(
            f"LogisticRegression fitted: {len(self.classes_)} classes, {X.shape[1]} features"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions."""
        self._check_is_fitted()
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self._check_is_fitted()
        return self.model_.predict_proba(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "l1_ratio": self.l1_ratio,
            "tol": self.tol,
        }

    def set_params(self, **params) -> "LogisticRegressionModel":
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._build_model()
        if self._is_fitted:
            logger.warning("Model parameters changed - refitting required")
            self._is_fitted = False
        return self

    @property
    def is_classifier(self) -> bool:
        return True


class SVMModel(BaseModel):
    """
    Support Vector Machine classifier.

    Provides SVM with various kernels for non-linear classification.
    Note: SVM doesn't provide coefficient-based feature importance,
    use `coef_` only for linear kernel.

    Parameters
    ----------
    C : float
        Regularization parameter (default: 1.0)
    kernel : str
        Kernel type: 'linear', 'rbf', 'poly', 'sigmoid' (default: 'rbf')
    gamma : Union[str, float]
        Kernel coefficient (default: 'scale')
    class_weight : Union[str, Dict, None]
        Class weight strategy (default: 'balanced')
    probability : bool
        Enable probability estimates (default: True)
    random_state : Optional[int]
        Random seed

    Examples
    --------
    >>> model = SVMModel(kernel='rbf', C=1.0)
    >>> model.fit(X_train, y_train)
    >>> proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        class_weight: Union[str, Dict, None] = "balanced",
        probability: bool = True,
        random_state: Optional[int] = 42,
    ):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.class_weight = class_weight
        self.probability = probability
        self.random_state = random_state

        self._build_model()

    def _build_model(self):
        """Build the sklearn model."""
        from sklearn.svm import SVC

        self.model_ = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            class_weight=self.class_weight,
            probability=self.probability,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMModel":
        """Fit the model."""
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_

        # Only linear kernel has coef_
        if self.kernel == "linear":
            self.coef_ = self.model_.coef_
        else:
            self.coef_ = None

        self._is_fitted = True
        logger.info(f"SVM fitted: {self.kernel} kernel, {len(self.classes_)} classes")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions."""
        self._check_is_fitted()
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self._check_is_fitted()
        if not self.probability:
            raise RuntimeError("probability=False, cannot predict_proba")
        return self.model_.predict_proba(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "C": self.C,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "class_weight": self.class_weight,
            "probability": self.probability,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "SVMModel":
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._build_model()
        if self._is_fitted:
            logger.warning("Model parameters changed - refitting required")
            self._is_fitted = False
        return self

    @property
    def is_classifier(self) -> bool:
        return True


class ElasticNetModel(BaseModel):
    """
    Elastic Net regression model.

    For continuous target variables (e.g., predicting response magnitude).
    Combines L1 and L2 regularization.

    Parameters
    ----------
    alpha : float
        Regularization strength (default: 1.0)
    l1_ratio : float
        L1/L2 mixing (0=Ridge, 1=Lasso, default: 0.5)
    max_iter : int
        Maximum iterations (default: 1000)
    random_state : Optional[int]
        Random seed

    Examples
    --------
    >>> model = ElasticNetModel(alpha=0.5, l1_ratio=0.3)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        random_state: Optional[int] = 42,
        tol: float = 1e-4,
    ):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol

        self._build_model()

    def _build_model(self):
        """Build the sklearn model."""
        from sklearn.linear_model import ElasticNet

        self.model_ = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            random_state=self.random_state,
            tol=self.tol,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetModel":
        """Fit the model."""
        self.model_.fit(X, y)
        self.coef_ = self.model_.coef_
        self._is_fitted = True
        logger.info(f"ElasticNet fitted: {X.shape[1]} features, alpha={self.alpha}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self._check_is_fitted()
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Not applicable for regression."""
        raise NotImplementedError("ElasticNetModel is a regressor, not classifier")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "tol": self.tol,
        }

    def set_params(self, **params) -> "ElasticNetModel":
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._build_model()
        if self._is_fitted:
            logger.warning("Model parameters changed - refitting required")
            self._is_fitted = False
        return self

    @property
    def is_classifier(self) -> bool:
        return False


class RandomForestModel(BaseModel):
    """
    Random Forest classifier.

    Ensemble method that may capture non-linear relationships.
    Provides feature importance based on impurity reduction.

    Parameters
    ----------
    n_estimators : int
        Number of trees (default: 100)
    max_depth : Optional[int]
        Maximum tree depth
    class_weight : Union[str, Dict, None]
        Class weight strategy (default: 'balanced')
    random_state : Optional[int]
        Random seed

    Examples
    --------
    >>> model = RandomForestModel(n_estimators=100, max_depth=10)
    >>> model.fit(X_train, y_train)
    >>> proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        class_weight: Union[str, Dict, None] = "balanced",
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._build_model()

    def _build_model(self):
        """Build the sklearn model."""
        from sklearn.ensemble import RandomForestClassifier

        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Fit the model."""
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.coef_ = None  # Use feature_importances_ instead
        self._is_fitted = True
        logger.info(f"RandomForest fitted: {self.n_estimators} trees")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self._check_is_fitted()
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        self._check_is_fitted()
        return self.model_.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from the forest."""
        self._check_is_fitted()
        return self.model_.feature_importances_

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def set_params(self, **params) -> "RandomForestModel":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._build_model()
        if self._is_fitted:
            self._is_fitted = False
        return self

    @property
    def is_classifier(self) -> bool:
        return True


def get_model(model_name: str, **kwargs) -> BaseModel:
    """
    Factory function to get a model by name.

    Parameters
    ----------
    model_name : str
        One of: 'logistic', 'svm', 'elasticnet', 'rf'
    **kwargs
        Model-specific parameters

    Returns
    -------
    model : BaseModel
        Instantiated model
    """
    models = {
        "logistic": LogisticRegressionModel,
        "logistic_regression": LogisticRegressionModel,
        "lr": LogisticRegressionModel,
        "svm": SVMModel,
        "elasticnet": ElasticNetModel,
        "en": ElasticNetModel,
        "rf": RandomForestModel,
        "random_forest": RandomForestModel,
    }

    model_name = model_name.lower()
    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: {list(models.keys())}"
        )

    return models[model_name](**kwargs)
