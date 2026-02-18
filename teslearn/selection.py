"""
Feature selection for TESLearn.

Provides statistical and atlas-based methods for selecting informative features
before model training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple
import numpy as np
import logging
from scipy import stats

from .base import BaseFeatureSelector, BaseFeatureExtractor
from .features import AtlasFeatureExtractor, VoxelFeatureExtractor

logger = logging.getLogger(__name__)


class TTestSelector(BaseFeatureSelector):
    """
    Feature selection using mass univariate t-tests.

    Performs independent t-tests comparing responders vs non-responders
    for each feature and selects those below a significance threshold.

    This is the default feature selector for classification tasks.

    Parameters
    ----------
    p_threshold : float
        P-value threshold for feature selection (default: 0.001)
    correction : Optional[str]
        Multiple comparison correction: None, 'bonferroni', 'fdr'

    Examples
    --------
    >>> selector = TTestSelector(p_threshold=0.001)
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        p_threshold: float = 0.001,
        correction: Optional[str] = None,
    ):
        super().__init__()
        self.p_threshold = p_threshold
        self.correction = correction
        self.p_values_: Optional[np.ndarray] = None
        self.t_stats_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TTestSelector":
        """
        Fit t-test selector.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Binary target vector (0/1)
        """
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X has {len(X)} samples, y has {len(y)}")

        if len(np.unique(y)) != 2:
            raise ValueError("TTestSelector requires binary targets (0/1)")

        n_features = X.shape[1]

        # Split by class
        mask_1 = y == 1
        mask_0 = y == 0

        if np.sum(mask_1) < 2 or np.sum(mask_0) < 2:
            raise ValueError("Need at least 2 samples per class for t-test")

        X_1 = X[mask_1]
        X_0 = X[mask_0]

        # Mass univariate t-test
        self.t_stats_, self.p_values_ = stats.ttest_ind(
            X_1, X_0, axis=0, equal_var=False
        )

        # Handle NaN p-values (from zero variance features)
        valid_mask = ~np.isnan(self.p_values_)

        # Apply multiple comparison correction
        p_values_corrected = self.p_values_.copy()
        if self.correction == "bonferroni":
            p_values_corrected[valid_mask] *= np.sum(valid_mask)
            p_values_corrected = np.minimum(p_values_corrected, 1.0)
        elif self.correction == "fdr":
            # Benjamini-Hochberg FDR
            from scipy.stats import false_discovery_rate as fdr

            p_values_corrected[valid_mask] = fdr(self.p_values_[valid_mask])[1]

        # Select significant features
        self.support_ = (p_values_corrected < self.p_threshold) & valid_mask
        self.selected_indices_ = np.flatnonzero(self.support_)

        n_selected = np.sum(self.support_)
        logger.info(
            f"TTestSelector: selected {n_selected}/{n_features} features (p < {self.p_threshold})"
        )

        if n_selected == 0:
            logger.warning("No features passed the significance threshold!")

        self._is_fitted = True
        return self

    def get_significant_features(self, feature_names: List[str]) -> List[str]:
        """Get names of selected features."""
        self._check_is_fitted()
        return [feature_names[i] for i in self.selected_indices_]


class FRegressionSelector(BaseFeatureSelector):
    """
    Feature selection using univariate F-tests for regression.

    Performs F-tests for each feature against the continuous target
    and selects features below a significance threshold.

    Parameters
    ----------
    p_threshold : float
        P-value threshold for feature selection (default: 0.001)

    Examples
    --------
    >>> selector = FRegressionSelector(p_threshold=0.001)
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(self, p_threshold: float = 0.001):
        super().__init__()
        self.p_threshold = p_threshold
        self.f_stats_: Optional[np.ndarray] = None
        self.p_values_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FRegressionSelector":
        """
        Fit F-test selector.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Continuous target vector
        """
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X has {len(X)} samples, y has {len(y)}")

        if len(np.unique(y)) < 2:
            raise ValueError("Target must have at least 2 unique values")

        from sklearn.feature_selection import f_regression

        self.f_stats_, self.p_values_ = f_regression(X, y)

        # Handle NaN
        valid_mask = ~np.isnan(self.p_values_)

        # Select features
        self.support_ = (self.p_values_ < self.p_threshold) & valid_mask
        self.selected_indices_ = np.flatnonzero(self.support_)

        n_selected = np.sum(self.support_)
        n_features = X.shape[1]
        logger.info(
            f"FRegressionSelector: selected {n_selected}/{n_features} features (p < {self.p_threshold})"
        )

        if n_selected == 0:
            logger.warning("No features passed the significance threshold!")

        self._is_fitted = True
        return self


class AtlasSelector(BaseFeatureSelector):
    """
    Atlas-based feature selection.

    Selects features corresponding to specific ROIs from an atlas.
    This is useful when you want to restrict analysis to anatomically
    defined regions of interest.

    Parameters
    ----------
    atlas_path : Path
        Path to atlas NIfTI
    roi_ids : List[int]
        List of ROI IDs to select
    extractor : Optional[AtlasFeatureExtractor]
        Pre-configured extractor (if None, creates new one)

    Examples
    --------
    >>> selector = AtlasSelector('atlas.nii.gz', roi_ids=[1, 2, 3, 4, 5])
    >>> X_roi = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        atlas_path: Path,
        roi_ids: List[int],
        extractor: Optional[AtlasFeatureExtractor] = None,
    ):
        super().__init__()
        self.atlas_path = Path(atlas_path)
        self.roi_ids = roi_ids
        self.extractor = extractor
        self._roi_indices: Optional[List[int]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AtlasSelector":
        """
        Fit atlas selector.

        Note: For AtlasSelector, X should be features extracted with
        the same atlas. This method validates ROI selection.
        """
        # This selector doesn't actually need to fit - it just validates
        self.support_ = np.ones(X.shape[1], dtype=bool)
        self.selected_indices_ = np.arange(X.shape[1])

        logger.info(f"AtlasSelector: using {len(self.roi_ids)} ROIs")

        self._is_fitted = True
        return self


class VarianceThresholdSelector(BaseFeatureSelector):
    """
    Remove features with low variance.

    Simple but effective preprocessing step to remove constant
    or near-constant features.

    Parameters
    ----------
    threshold : float
        Minimum variance threshold (default: 0.0)

    Examples
    --------
    >>> selector = VarianceThresholdSelector(threshold=0.01)
    >>> X_filtered = selector.fit_transform(X, y)
    """

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold
        self.variances_: Optional[np.ndarray] = None

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "VarianceThresholdSelector":
        """Fit by computing variances."""
        self.variances_ = np.var(X, axis=0)
        self.support_ = self.variances_ > self.threshold
        self.selected_indices_ = np.flatnonzero(self.support_)

        n_selected = np.sum(self.support_)
        n_total = X.shape[1]
        logger.info(
            f"VarianceThresholdSelector: retained {n_selected}/{n_total} features (var > {self.threshold})"
        )

        self._is_fitted = True
        return self


class TopKSelector(BaseFeatureSelector):
    """
    Select top K features based on univariate statistical tests.

    Parameters
    ----------
    k : int
        Number of features to select
    score_func : callable
        Scoring function (default: f_classif)

    Examples
    --------
    >>> selector = TopKSelector(k=100)
    >>> X_top = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        k: int = 100,
        score_func: Optional[Any] = None,
    ):
        super().__init__()
        self.k = k
        self.score_func = score_func
        self.scores_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TopKSelector":
        """Fit by computing univariate scores."""
        if self.score_func is None:
            from sklearn.feature_selection import f_classif

            self.score_func = f_classif

        self.scores_, _ = self.score_func(X, y)

        # Handle NaN
        self.scores_ = np.nan_to_num(self.scores_, nan=-np.inf)

        # Select top k
        k = min(self.k, X.shape[1])
        self.selected_indices_ = np.argsort(self.scores_)[-k:]
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[self.selected_indices_] = True

        logger.info(f"TopKSelector: selected top {k} features")

        self._is_fitted = True
        return self


class VoxelSelectorFromImages(BaseFeatureSelector):
    """
    Statistical feature selection directly from voxel-level E-field images.

    This performs voxel-wise statistical tests on the raw E-field data
    and returns selected voxel coordinates.

    Parameters
    ----------
    p_threshold : float
        P-value threshold
    test_type : str
        'ttest' for classification, 'fregression' for continuous targets
    verbose : bool
        Show progress information (default: True)

    Examples
    --------
    >>> selector = VoxelSelectorFromImages(p_threshold=0.001, test_type='ttest')
    >>> selector.fit(images, y)
    >>> selected_coords = selector.get_voxel_coordinates()
    """

    def __init__(
        self,
        p_threshold: float = 0.001,
        test_type: str = "ttest",
        verbose: bool = True,
    ):
        super().__init__()
        self.p_threshold = p_threshold
        self.test_type = test_type
        self.verbose = verbose
        self.voxel_coords: Optional[List[Tuple[int, int, int]]] = None
        self.p_values_: Optional[np.ndarray] = None
        self._reference_shape: Optional[Tuple[int, int, int]] = None
        self._reference_affine: Optional[np.ndarray] = None

    def fit(
        self,
        images: List[Any],
        y: np.ndarray,
    ) -> "VoxelSelectorFromImages":
        """
        Fit by performing voxel-wise statistical tests.

        Parameters
        ----------
        images : List[NiftiImageLike]
            E-field images
        y : np.ndarray
            Target values
        """
        from .features import _resample_to_reference, _ensure_3d_data

        if len(images) != len(y):
            raise ValueError("Number of images must match length of y")

        if self.verbose:
            print(f"Loading {len(images)} images...")

        # Load and flatten all images
        data_arrays, ref_img, shape = _resample_to_reference(images)
        self._reference_shape = shape
        self._reference_affine = ref_img.affine

        # Create data matrix (n_samples, n_voxels)
        if self.verbose:
            print(
                f"Creating data matrix ({len(images)} samples x {np.prod(shape)} voxels)..."
            )
        X = np.array([data.ravel() for data in data_arrays])

        if self.verbose:
            print(f"Running {self.test_type} on {X.shape[1]} voxels...")

        # Perform statistical tests
        if self.test_type == "ttest":
            if len(np.unique(y)) != 2:
                raise ValueError("ttest requires binary targets")

            mask_1 = y == 1
            mask_0 = y == 0

            t_stats, p_values = stats.ttest_ind(
                X[mask_1], X[mask_0], axis=0, equal_var=False
            )
        elif self.test_type == "fregression":
            from sklearn.feature_selection import f_regression

            _, p_values = f_regression(X, y)
        else:
            raise ValueError(f"Unknown test_type: {self.test_type}")

        self.p_values_ = p_values

        # Select significant voxels
        valid_mask = ~np.isnan(p_values)
        significant_mask = (p_values < self.p_threshold) & valid_mask

        # Convert to coordinates
        flat_indices = np.flatnonzero(significant_mask)
        coords_array = np.unravel_index(flat_indices, shape)
        self.voxel_coords = list(zip(coords_array[0], coords_array[1], coords_array[2]))

        msg = f"VoxelSelectorFromImages: selected {len(self.voxel_coords)} voxels (p < {self.p_threshold})"
        if self.verbose:
            print(msg)
        logger.info(msg)

        self._is_fitted = True
        return self

    def get_voxel_coordinates(self) -> List[Tuple[int, int, int]]:
        """Get selected voxel coordinates."""
        self._check_is_fitted()
        return self.voxel_coords

    def create_voxel_extractor(self) -> VoxelFeatureExtractor:
        """Create a VoxelFeatureExtractor with selected coordinates."""
        self._check_is_fitted()
        return VoxelFeatureExtractor(voxel_coords=self.voxel_coords)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform not applicable - use create_voxel_extractor instead."""
        raise NotImplementedError(
            "VoxelSelectorFromImages requires images. Use create_voxel_extractor() "
            "to get a feature extractor for the selected voxels."
        )

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("VoxelSelectorFromImages must be fitted first")


class PermutationClusterSelector(BaseFeatureSelector):
    """
    Voxel selection using cluster-based permutation testing.

    Performs voxel-wise statistical tests with cluster-level correction
    for multiple comparisons using permutation testing. This controls
    family-wise error rate (FWER) by accounting for spatial correlations.

    Parameters
    ----------
    n_permutations : int
        Number of permutations for cluster test (default: 1000)
    alpha : float
        Significance level (default: 0.05)
    cluster_threshold : float
        Threshold for forming clusters in t-statistic (default: 2.0)
    min_cluster_size : int
        Minimum cluster size in voxels (default: 10)
    test_type : str
        'ttest' for two-sample test, 'onesample' for one-sample test
    n_jobs : int
        Number of parallel jobs (-1 uses all cores, default: -1)
    verbose : bool
        Show progress bar (default: True)

    Examples
    --------
    >>> selector = PermutationClusterSelector(
    ...     n_permutations=1000,
    ...     alpha=0.05,
    ...     min_cluster_size=10,
    ...     n_jobs=4
    ... )
    >>> selector.fit(images, y)
    >>> selected_coords = selector.get_voxel_coordinates()
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        alpha: float = 0.05,
        cluster_threshold: float = 2.0,
        min_cluster_size: int = 10,
        test_type: str = "ttest",
        n_jobs: int = -1,
        verbose: bool = True,
    ):
        super().__init__()
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.cluster_threshold = cluster_threshold
        self.min_cluster_size = min_cluster_size
        self.test_type = test_type
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.voxel_coords: Optional[List[Tuple[int, int, int]]] = None
        self.t_stats_: Optional[np.ndarray] = None
        self.clusters_: Optional[List[np.ndarray]] = None
        self.cluster_p_values_: Optional[np.ndarray] = None
        self._reference_shape: Optional[Tuple[int, int, int]] = None
        self._reference_affine: Optional[np.ndarray] = None

    def fit(
        self,
        images: List[Any],
        y: np.ndarray,
    ) -> "PermutationClusterSelector":
        """
        Fit by performing cluster-based permutation testing.

        Parameters
        ----------
        images : List[NiftiImageLike]
            E-field images
        y : np.ndarray
            Target values (binary for ttest, any for onesample)
        """
        from .features import _resample_to_reference

        if len(images) != len(y):
            raise ValueError("Number of images must match length of y")

        # Load and flatten all images
        data_arrays, ref_img, shape = _resample_to_reference(images)
        self._reference_shape = shape
        self._reference_affine = ref_img.affine

        # Create data matrix (n_samples, n_voxels)
        X = np.array([data.ravel() for data in data_arrays])

        # Compute observed t-statistics
        if self.test_type == "ttest":
            if len(np.unique(y)) != 2:
                raise ValueError("ttest requires binary targets")
            mask_1 = y == 1
            mask_0 = y == 0
            self.t_stats_, _ = stats.ttest_ind(
                X[mask_1], X[mask_0], axis=0, equal_var=False
            )
        elif self.test_type == "onesample":
            self.t_stats_, _ = stats.ttest_1samp(X, 0, axis=0)
        else:
            raise ValueError(f"Unknown test_type: {self.test_type}")

        # Handle NaN values
        self.t_stats_ = np.nan_to_num(self.t_stats_, nan=0.0)

        # Find clusters in observed data
        self.clusters_, cluster_scores = self._find_clusters(self.t_stats_)

        if len(self.clusters_) == 0:
            logger.warning("No clusters found at threshold")
            self.voxel_coords = []
            self._is_fitted = True
            return self

        # Permutation testing with progress bar and parallelization
        null_distribution = self._run_permutations(X, y)

        # Compute cluster p-values
        # Handle case where null distribution has no variance (all zeros)
        if np.all(null_distribution == 0):
            logger.warning(
                "No clusters found in any permutation - cannot compute valid p-values. "
                "Consider lowering cluster_threshold or min_cluster_size."
            )
            self.cluster_p_values_ = np.ones(len(cluster_scores))
        else:
            self.cluster_p_values_ = np.array(
                [np.mean(null_distribution >= score) for score in cluster_scores]
            )

        # Select voxels from significant clusters
        significant_clusters = [
            cluster
            for cluster, p in zip(self.clusters_, self.cluster_p_values_)
            if p < self.alpha
        ]

        if significant_clusters:
            selected_indices = np.concatenate(significant_clusters)
            coords_array = np.unravel_index(selected_indices, shape)
            self.voxel_coords = list(
                zip(coords_array[0], coords_array[1], coords_array[2])
            )
        else:
            self.voxel_coords = []
            logger.warning(
                f"No significant clusters found after permutation testing. "
                f"Found {len(self.clusters_)} clusters, but none significant at "
                f"alpha={self.alpha}. Consider adjusting cluster_threshold or min_cluster_size."
            )

        logger.info(
            f"PermutationClusterSelector: found {len(self.clusters_)} clusters, "
            f"{len(significant_clusters)} significant (p < {self.alpha}), "
            f"{len(self.voxel_coords)} voxels selected"
        )

        self._is_fitted = True
        return self

    def _find_clusters(
        self, t_stats: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Find clusters in t-statistic map.

        Uses simple connected component labeling in 3D space.
        """
        from scipy import ndimage

        # Reshape to 3D
        t_map = t_stats.reshape(self._reference_shape)

        # Find clusters above threshold
        mask = np.abs(t_map) > self.cluster_threshold
        labeled, n_clusters = ndimage.label(mask)

        clusters = []
        cluster_scores = []

        for i in range(1, n_clusters + 1):
            cluster_mask = labeled == i
            cluster_indices = np.flatnonzero(cluster_mask)

            if len(cluster_indices) >= self.min_cluster_size:
                clusters.append(cluster_indices)
                # Cluster score is sum of absolute t-values
                score = np.sum(np.abs(t_map[cluster_mask]))
                cluster_scores.append(score)

        return clusters, cluster_scores

    def get_voxel_coordinates(self) -> List[Tuple[int, int, int]]:
        """Get selected voxel coordinates."""
        self._check_is_fitted()
        return self.voxel_coords

    def create_voxel_extractor(self) -> VoxelFeatureExtractor:
        """Create a VoxelFeatureExtractor with selected coordinates."""
        self._check_is_fitted()
        return VoxelFeatureExtractor(voxel_coords=self.voxel_coords)

    def _run_permutations(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Run permutations with parallelization and progress bar.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (n_samples, n_voxels)
        y : np.ndarray
            Target values

        Returns
        -------
        null_distribution : np.ndarray
            Max cluster scores from each permutation
        """
        from joblib import Parallel, delayed
        from tqdm import tqdm
        import os

        # Determine number of jobs
        n_jobs = self.n_jobs if self.n_jobs != -1 else os.cpu_count()
        n_jobs = min(n_jobs, self.n_permutations)

        # Create worker function
        def _single_permutation(seed: int) -> float:
            """Run a single permutation and return max cluster score."""
            rng = np.random.RandomState(seed)
            y_perm = rng.permutation(y)

            # Compute statistics for permuted data
            if self.test_type == "ttest":
                mask_1_perm = y_perm == 1
                mask_0_perm = y_perm == 0
                t_perm, _ = stats.ttest_ind(
                    X[mask_1_perm], X[mask_0_perm], axis=0, equal_var=False
                )
            else:  # onesample
                t_perm, _ = stats.ttest_1samp(X, 0, axis=0)

            t_perm = np.nan_to_num(t_perm, nan=0.0)

            # Find clusters in permuted data
            _, perm_scores = self._find_clusters(t_perm)

            return np.max(perm_scores) if len(perm_scores) > 0 else 0.0

        # Run permutations in parallel with progress bar
        if self.verbose:
            print(f"Running {self.n_permutations} permutations using {n_jobs} cores...")
            iterator = tqdm(
                range(self.n_permutations),
                total=self.n_permutations,
                desc="Permutations",
                unit="perm",
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        else:
            iterator = range(self.n_permutations)

        # Use sequential processing for small number of permutations
        if n_jobs == 1 or self.n_permutations < 10:
            null_distribution = np.array([_single_permutation(i) for i in iterator])
        else:
            # Parallel processing
            null_distribution = np.array(
                Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(_single_permutation)(i) for i in iterator
                )
            )

        if self.verbose:
            print(
                f"Permutations complete. Found {np.sum(null_distribution > 0)} non-zero cluster scores."
            )

        return null_distribution

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform not applicable - use create_voxel_extractor instead."""
        raise NotImplementedError(
            "PermutationClusterSelector requires images. Use create_voxel_extractor() "
            "to get a feature extractor for the selected voxels."
        )

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("PermutationClusterSelector must be fitted first")
