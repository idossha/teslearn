"""
Cross-validation utilities for TESLearn.

Provides robust cross-validation strategies tailored for small sample sizes
common in TES studies.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, Tuple
import numpy as np
import logging

from .base import BaseValidator

logger = logging.getLogger(__name__)


class StratifiedKFoldValidator(BaseValidator):
    """
    Stratified K-Fold cross-validator.

    Maintains class distribution across folds, essential for classification
    with imbalanced data.

    Parameters
    ----------
    n_splits : int
        Number of folds (default: 5)
    shuffle : bool
        Whether to shuffle before splitting (default: True)
    random_state : Optional[int]
        Random seed

    Examples
    --------
    >>> validator = StratifiedKFoldValidator(n_splits=5)
    >>> for train_idx, test_idx in validator.split(X, y):
    ...     model.fit(X[train_idx], y[train_idx])
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_splits=n_splits, random_state=random_state)
        self.shuffle = shuffle

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.

        Yields
        ------
        train_index : np.ndarray
            Training indices
        test_index : np.ndarray
            Test indices
        """
        from sklearn.model_selection import StratifiedKFold

        if y is None:
            raise ValueError("StratifiedKFold requires y")

        # Adjust n_splits if necessary
        n_splits = min(self.n_splits, self._compute_max_splits(y))

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        yield from cv.split(X, y)

    def _compute_max_splits(self, y: np.ndarray) -> int:
        """Compute maximum safe number of splits."""
        classes, counts = np.unique(y, return_counts=True)
        min_class_size = np.min(counts)
        # Need at least 1 sample per class per fold
        return min(self.n_splits, min_class_size)


class KFoldValidator(BaseValidator):
    """
    K-Fold cross-validator for regression.

    Simple K-fold splitting without stratification, appropriate for
    continuous targets.

    Parameters
    ----------
    n_splits : int
        Number of folds (default: 5)
    shuffle : bool
        Whether to shuffle (default: True)
    random_state : Optional[int]
        Random seed

    Examples
    --------
    >>> validator = KFoldValidator(n_splits=5)
    >>> for train_idx, test_idx in validator.split(X, y):
    ...     model.fit(X[train_idx], y[train_idx])
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_splits=n_splits, random_state=random_state)
        self.shuffle = shuffle

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices."""
        from sklearn.model_selection import KFold

        n_splits = min(self.n_splits, len(X))

        cv = KFold(
            n_splits=n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        yield from cv.split(X)


class LeaveOneOutValidator(BaseValidator):
    """
    Leave-One-Out cross-validator.

    Useful for very small datasets where every sample is valuable.

    Parameters
    ----------
    random_state : Optional[int]
        Not used, present for API compatibility

    Examples
    --------
    >>> validator = LeaveOneOutValidator()
    >>> predictions = []
    >>> for train_idx, test_idx in validator.split(X, y):
    ...     model.fit(X[train_idx], y[train_idx])
    ...     predictions.append(model.predict(X[test_idx]))
    """

    def __init__(self, random_state: Optional[int] = None):
        super().__init__(n_splits=0, random_state=random_state)

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices."""
        from sklearn.model_selection import LeaveOneOut

        cv = LeaveOneOut()
        self.n_splits = cv.get_n_splits(X)

        yield from cv.split(X)

    def get_n_splits(
        self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None
    ) -> int:
        """Get number of splits."""
        if X is not None:
            return len(X)
        return 0


class NestedCrossValidator:
    """
    Nested cross-validation for unbiased performance estimation.

    Outer loop: model evaluation
    Inner loop: hyperparameter tuning

    Parameters
    ----------
    outer_validator : BaseValidator
        Validator for outer loop
    inner_validator : BaseValidator
        Validator for inner loop

    Examples
    --------
    >>> outer = StratifiedKFoldValidator(n_splits=5)
    >>> inner = StratifiedKFoldValidator(n_splits=3)
    >>> nested = NestedCrossValidator(outer, inner)
    >>> for fold_idx, (X_tr, X_te, y_tr, y_te) in enumerate(nested.split(X, y)):
    ...     # Inner CV for hyperparameter selection
    ...     best_params = grid_search(X_tr, y_tr, inner)
    ...     # Train final model
    ...     model.fit(X_tr, y_tr, **best_params)
    ...     # Evaluate
    ...     score = model.score(X_te, y_te)
    """

    def __init__(
        self,
        outer_validator: BaseValidator,
        inner_validator: BaseValidator,
    ):
        self.outer_validator = outer_validator
        self.inner_validator = inner_validator

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate nested CV splits.

        Yields
        ------
        fold_idx : int
            Current fold index
        X_train : np.ndarray
            Training features
        X_test : np.ndarray
            Test features
        y_train : np.ndarray
            Training targets
        y_test : np.ndarray
            Test targets
        """
        for fold_idx, (train_idx, test_idx) in enumerate(
            self.outer_validator.split(X, y), 1
        ):
            yield fold_idx, X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def get_n_splits(self) -> int:
        """Get number of outer splits."""
        return self.outer_validator.get_n_splits()


def create_validator(
    validator_type: str = "stratified_kfold",
    n_splits: int = 5,
    random_state: Optional[int] = 42,
    **kwargs,
) -> BaseValidator:
    """
    Factory function to create validators.

    Parameters
    ----------
    validator_type : str
        Type of validator: 'stratified_kfold', 'kfold', 'loo'
    n_splits : int
        Number of splits
    random_state : Optional[int]
        Random seed
    **kwargs
        Additional validator-specific parameters

    Returns
    -------
    validator : BaseValidator
        Configured validator
    """
    validators = {
        "stratified_kfold": StratifiedKFoldValidator,
        "skf": StratifiedKFoldValidator,
        "kfold": KFoldValidator,
        "kf": KFoldValidator,
        "loo": LeaveOneOutValidator,
        "leave_one_out": LeaveOneOutValidator,
    }

    validator_type = validator_type.lower()
    if validator_type not in validators:
        raise ValueError(f"Unknown validator: {validator_type}")

    if validator_type in ["loo", "leave_one_out"]:
        return validators[validator_type](random_state=random_state)

    return validators[validator_type](
        n_splits=n_splits, random_state=random_state, **kwargs
    )
