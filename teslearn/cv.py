"""
Cross-validation utilities for TESLearn.

Provides robust cross-validation strategies tailored for small sample sizes
common in TES studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    ) -> Iterator[
        Tuple[
            int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ]
    ]:
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
        train_idx : np.ndarray
            Training indices
        test_idx : np.ndarray
            Test indices
        """
        for fold_idx, (train_idx, test_idx) in enumerate(
            self.outer_validator.split(X, y), 1
        ):
            yield (
                fold_idx,
                X[train_idx],
                X[test_idx],
                y[train_idx],
                y[test_idx],
                train_idx,
                test_idx,
            )

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


@dataclass
class PermutationTestResult:
    """Result from permutation testing."""
    
    observed_score: float
    permuted_scores: np.ndarray
    p_value: float
    n_permutations: int
    score_name: str
    
    def get_summary(self) -> str:
        """Get text summary of permutation test."""
        mean_perm = np.mean(self.permuted_scores)
        std_perm = np.std(self.permuted_scores)
        
        lines = [
            "=" * 60,
            "Permutation Test Results",
            "=" * 60,
            f"Score metric: {self.score_name}",
            f"Observed score: {self.observed_score:.4f}",
            f"Permutation distribution: mean={mean_perm:.4f}, std={std_perm:.4f}",
            f"Number of permutations: {self.n_permutations}",
            f"P-value: {self.p_value:.4f} {'(significant)' if self.p_value < 0.05 else '(not significant)'}"",
            "=" * 60,
        ]
        return "\n".join(lines)


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    validator: BaseValidator,
    n_permutations: int = 1000,
    scoring: str = "roc_auc",
    random_state: Optional[int] = 42,
    n_jobs: int = 1,
    verbose: bool = True,
) -> PermutationTestResult:
    """
    Perform permutation testing to assess model significance.
    
    Permutation testing creates a null distribution by randomly shuffling labels
    and re-training the model. The p-value is the proportion of permuted scores
    that are better than or equal to the observed score.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    model : Any
        Model object with fit() and predict/predict_proba() methods
    validator : BaseValidator
        Cross-validation strategy
    n_permutations : int
        Number of permutations to run (default: 1000)
    scoring : str
        Scoring metric: 'roc_auc', 'accuracy', 'f1', etc.
    random_state : Optional[int]
        Random seed for reproducibility
    n_jobs : int
        Number of parallel jobs (-1 uses all cores, default: 1)
    verbose : bool
        Whether to show progress
        
    Returns
    -------
    result : PermutationTestResult
        Permutation test results
        
    Examples
    --------
    >>> validator = StratifiedKFoldValidator(n_splits=5)
    >>> model = LogisticRegression()
    >>> result = permutation_test(X, y, model, validator, n_permutations=1000)
    >>> print(result.get_summary())
    >>> if result.p_value < 0.05:
    ...     print("Model performance is statistically significant!")
    """
    from sklearn.metrics import get_scorer, roc_auc_score, accuracy_score, f1_score
    from joblib import Parallel, delayed
    import time
    
    rng = np.random.RandomState(random_state)
    
    # Compute observed score
    logger.info("Computing observed score...")
    observed_predictions = []
    observed_true = []
    
    for train_idx, test_idx in validator.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)
        
        if scoring == "roc_auc":
            try:
                y_pred = model_clone.predict_proba(X_test)[:, 1]
            except AttributeError:
                y_pred = model_clone.predict(X_test)
        else:
            y_pred = model_clone.predict(X_test)
        
        observed_predictions.extend(y_pred if hasattr(y_pred, '__iter__') else [y_pred])
        observed_true.extend(y_test)
    
    # Calculate observed score
    y_true_arr = np.array(observed_true)
    y_pred_arr = np.array(observed_predictions)
    
    if scoring == "roc_auc":
        observed_score = roc_auc_score(y_true_arr, y_pred_arr)
    elif scoring == "accuracy":
        observed_score = accuracy_score(y_true_arr, (y_pred_arr > 0.5).astype(int))
    elif scoring == "f1":
        observed_score = f1_score(y_true_arr, (y_pred_arr > 0.5).astype(int))
    else:
        scorer = get_scorer(scoring)
        observed_score = scorer._score_func(y_true_arr, y_pred_arr)
    
    logger.info(f"Observed {scoring}: {observed_score:.4f}")
    
    # Run permutations
    def _single_permutation(seed: int) -> float:
        """Run a single permutation and return score."""
        rng_perm = np.random.RandomState(seed)
        y_perm = rng_perm.permutation(y)
        
        perm_predictions = []
        perm_true = []
        
        for train_idx, test_idx in validator.split(X, y_perm):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_perm[train_idx], y_perm[test_idx]
            
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train, y_train)
            
            if scoring == "roc_auc":
                try:
                    y_pred = model_clone.predict_proba(X_test)[:, 1]
                except AttributeError:
                    y_pred = model_clone.predict(X_test)
            else:
                y_pred = model_clone.predict(X_test)
            
            perm_predictions.extend(y_pred if hasattr(y_pred, '__iter__') else [y_pred])
            perm_true.extend(y_test)
        
        y_true_perm = np.array(perm_true)
        y_pred_perm = np.array(perm_predictions)
        
        if scoring == "roc_auc":
            return roc_auc_score(y_true_perm, y_pred_perm)
        elif scoring == "accuracy":
            return accuracy_score(y_true_perm, (y_pred_perm > 0.5).astype(int))
        elif scoring == "f1":
            return f1_score(y_true_perm, (y_pred_perm > 0.5).astype(int))
        else:
            scorer = get_scorer(scoring)
            return scorer._score_func(y_true_perm, y_pred_perm)
    
    # Run permutations (parallel or sequential)
    logger.info(f"Running {n_permutations} permutations...")
    start_time = time.time()
    
    if n_jobs == 1:
        permuted_scores = np.array([_single_permutation(rng.randint(0, 2**31)) 
                                     for _ in range(n_permutations)])
    else:
        n_jobs_eff = n_jobs if n_jobs > 0 else -1
        seeds = rng.randint(0, 2**31, size=n_permutations)
        permuted_scores = np.array(
            Parallel(n_jobs=n_jobs_eff)(
                delayed(_single_permutation)(seed) for seed in seeds
            )
        )
    
    elapsed = time.time() - start_time
    logger.info(f"Permutations completed in {elapsed:.1f}s")
    
    # Calculate p-value (one-sided: proportion of permutations >= observed)
    p_value = (np.sum(permuted_scores >= observed_score) + 1) / (n_permutations + 1)
    
    logger.info(f"P-value: {p_value:.4f}")
    
    return PermutationTestResult(
        observed_score=observed_score,
        permuted_scores=permuted_scores,
        p_value=p_value,
        n_permutations=n_permutations,
        score_name=scoring,
    )


def plot_permutation_test(
    result: PermutationTestResult,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 6),
) -> Path:
    """
    Plot permutation test results.
    
    Creates a histogram of permuted scores with observed score marked.
    
    Parameters
    ----------
    result : PermutationTestResult
        Result from permutation_test()
    output_path : Path
        Output path for figure
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    output_path : Path
        Path to saved figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram of permuted scores
    ax.hist(result.permuted_scores, bins=30, alpha=0.7, color='gray', 
            edgecolor='black', label='Permuted scores')
    
    # Mark observed score
    ax.axvline(result.observed_score, color='red', linewidth=2, 
               linestyle='--', label=f'Observed: {result.observed_score:.3f}')
    
    ax.set_xlabel(result.score_name.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Permutation Test (p={result.p_value:.4f}, n={result.n_permutations})', 
                 fontsize=12)
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Permutation test plot saved to {output_path}")
    return output_path
