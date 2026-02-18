"""
Feature extraction from E-field images.

This module provides various methods to extract features from 3D E-field intensity
maps for machine learning classification/regression.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import logging

from .base import BaseFeatureExtractor, NiftiImageLike

logger = logging.getLogger(__name__)


def _ensure_3d_data(data: np.ndarray) -> np.ndarray:
    """Ensure data is 3D (squeeze trailing dimensions if 4D with singleton)."""
    if data.ndim == 4:
        if data.shape[-1] == 1:
            return data[..., 0]
        return data[..., -1]
    if data.ndim > 4:
        raise ValueError(f"Cannot handle {data.ndim}D images, expected 3D or 4D")
    return data


def _resample_to_reference(
    images: Sequence[NiftiImageLike], reference: Optional[NiftiImageLike] = None
) -> Tuple[List[np.ndarray], NiftiImageLike, Tuple[int, int, int]]:
    """
    Resample all images to a common reference space.

    Returns
    -------
    data_arrays : List[np.ndarray]
        List of resampled 3D arrays
    reference : NiftiImageLike
        Reference image
    shape : Tuple[int, int, int]
        Shape of 3D volume
    """
    import nibabel as nib
    from nibabel.processing import resample_to_output

    if not images:
        raise ValueError("No images provided")

    ref = reference or images[0]

    # Load reference data
    ref_data = np.asanyarray(ref.dataobj)
    ref_data_3d = _ensure_3d_data(ref_data)
    shape = ref_data_3d.shape

    data_arrays = []

    for img in images:
        # Resample to reference
        if img is ref:
            data = ref_data_3d
        else:
            # Simple resampling using nibabel
            resampled = nib.processing.resample_from_to(img, ref, order=1)
            data = np.asanyarray(resampled.dataobj)
            data = _ensure_3d_data(data)

        data_arrays.append(data)

    return data_arrays, ref, shape


def _flatten_voxels(data: np.ndarray) -> np.ndarray:
    """Flatten 3D array to 1D."""
    return data.ravel()


class AtlasFeatureExtractor(BaseFeatureExtractor):
    """
    Extract ROI-based features using an anatomical atlas.

    For each atlas ROI, computes various statistics of the E-field intensity
    distribution within that region.

    Parameters
    ----------
    atlas_path : Union[str, Path]
        Path to atlas NIfTI file in MNI space
    statistics : List[str]
        List of statistics to compute: 'mean', 'max', 'top10mean', etc.
    top_percentile : float
        Percentile for 'top' statistics (default: 90.0)

    Examples
    --------
    >>> extractor = AtlasFeatureExtractor(
    ...     atlas_path='path/to/atlas.nii.gz',
    ...     statistics=['mean', 'max']
    ... )
    >>> X = extractor.fit_transform(efield_images)
    """

    def __init__(
        self,
        atlas_path: Union[str, Path],
        statistics: Optional[List[str]] = None,
        top_percentile: float = 90.0,
    ):
        super().__init__()
        self.atlas_path = Path(atlas_path)
        self.statistics = statistics or ["mean"]
        self.top_percentile = top_percentile
        self._atlas_data: Optional[np.ndarray] = None
        self._roi_ids: Optional[List[int]] = None

    def fit(
        self, images: List[NiftiImageLike], y: Optional[np.ndarray] = None
    ) -> "AtlasFeatureExtractor":
        """
        Load atlas and identify ROIs.

        Parameters
        ----------
        images : List[NiftiImageLike]
            E-field images (used to verify atlas alignment)
        y : Optional[np.ndarray]
            Not used, present for API compatibility
        """
        import nibabel as nib
        from nibabel.processing import resample_from_to

        if not self.atlas_path.exists():
            raise FileNotFoundError(f"Atlas not found: {self.atlas_path}")

        # Load atlas and resample to image space (nearest-neighbour for labels)
        atlas_img = nib.load(str(self.atlas_path))
        if images:
            ref = images[0]
            atlas_img = resample_from_to(
                atlas_img, (ref.shape[:3], ref.affine), order=0
            )
        self._atlas_data = _ensure_3d_data(np.asanyarray(atlas_img.dataobj)).astype(
            np.int32
        )

        # Get unique ROI IDs (excluding background 0)
        unique_ids = np.unique(self._atlas_data)
        self._roi_ids = sorted([int(x) for x in unique_ids if x != 0])

        # Generate feature names
        self.feature_names_ = [
            f"ROI_{roi_id}__{stat}"
            for roi_id in self._roi_ids
            for stat in self.statistics
        ]

        self._is_fitted = True
        logger.info(
            f"Atlas loaded with {len(self._roi_ids)} ROIs, {len(self.feature_names_)} features"
        )

        return self

    def transform(self, images: List[NiftiImageLike]) -> np.ndarray:
        """
        Extract ROI features from images.

        Parameters
        ----------
        images : List[NiftiImageLike]
            E-field images

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape (n_images, n_features)
        """
        self._check_is_fitted()

        # Resample images to atlas space
        data_arrays, _, shape = _resample_to_reference(images)

        n_samples = len(images)
        n_rois = len(self._roi_ids)
        n_stats = len(self.statistics)

        X = np.zeros((n_samples, n_rois * n_stats), dtype=np.float32)

        # Precompute ROI masks
        flat_atlas = self._atlas_data.ravel()

        for sample_idx, data in enumerate(data_arrays):
            flat_data = data.ravel()

            feature_idx = 0
            for roi_id in self._roi_ids:
                # Get voxels in this ROI
                roi_mask = flat_atlas == roi_id
                roi_values = flat_data[roi_mask]

                if len(roi_values) == 0:
                    # Empty ROI
                    for _ in self.statistics:
                        X[sample_idx, feature_idx] = 0.0
                        feature_idx += 1
                    continue

                # Compute statistics
                for stat in self.statistics:
                    if stat == "mean":
                        value = np.mean(roi_values)
                    elif stat == "max":
                        value = np.max(roi_values)
                    elif stat == "min":
                        value = np.min(roi_values)
                    elif stat == "std":
                        value = np.std(roi_values)
                    elif stat == "median":
                        value = np.median(roi_values)
                    elif stat == "sum":
                        value = np.sum(roi_values)
                    elif stat.startswith("top"):
                        # e.g., 'top10mean' -> top 10% mean
                        percentile = self.top_percentile
                        threshold = np.percentile(roi_values, percentile)
                        top_values = roi_values[roi_values >= threshold]
                        value = (
                            np.mean(top_values)
                            if len(top_values) > 0
                            else np.mean(roi_values)
                        )
                    else:
                        value = np.mean(roi_values)

                    X[sample_idx, feature_idx] = value
                    feature_idx += 1

        return X

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("AtlasFeatureExtractor must be fitted before transform")


class VoxelFeatureExtractor(BaseFeatureExtractor):
    """
    Extract features from specific voxel coordinates.

    This is useful when you have pre-selected voxels (e.g., from statistical
    feature selection) and want to extract E-field values at those locations.

    Parameters
    ----------
    voxel_coords : Optional[List[Tuple[int, int, int]]]
        List of (x, y, z) voxel coordinates. If None, all non-zero voxels are used.

    Examples
    --------
    >>> # Extract from specific coordinates
    >>> coords = [(45, 50, 30), (46, 50, 30)]
    >>> extractor = VoxelFeatureExtractor(voxel_coords=coords)
    >>> X = extractor.fit_transform(efield_images)
    """

    def __init__(
        self,
        voxel_coords: Optional[List[Tuple[int, int, int]]] = None,
        mask: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.voxel_coords = voxel_coords
        self.mask = mask
        self._flat_indices: Optional[np.ndarray] = None
        self._reference_shape: Optional[Tuple[int, int, int]] = None

    def fit(
        self, images: List[NiftiImageLike], y: Optional[np.ndarray] = None
    ) -> "VoxelFeatureExtractor":
        """
        Determine voxel indices from coordinates or mask.

        Parameters
        ----------
        images : List[NiftiImageLike]
            Reference images for determining shape
        y : Optional[np.ndarray]
            Not used
        """
        if not images:
            raise ValueError("No images provided")

        # Get reference shape
        ref_data = np.asanyarray(images[0].dataobj)
        ref_data_3d = _ensure_3d_data(ref_data)
        self._reference_shape = ref_data_3d.shape

        if self.voxel_coords is not None:
            # Convert coordinates to flat indices
            self._flat_indices = np.array(
                [
                    np.ravel_multi_index(coord, self._reference_shape)
                    for coord in self.voxel_coords
                ]
            )
            self.feature_names_ = [
                f"voxel_{x}_{y}_{z}" for x, y, z in self.voxel_coords
            ]
        elif self.mask is not None:
            # Use mask to select voxels
            self._flat_indices = np.flatnonzero(self.mask)
            coords = np.unravel_index(self._flat_indices, self._reference_shape)
            self.feature_names_ = [
                f"voxel_{x}_{y}_{z}" for x, y, z in zip(coords[0], coords[1], coords[2])
            ]
        else:
            # Use all non-zero voxels from first image as template
            self._flat_indices = np.flatnonzero(ref_data_3d != 0)
            coords = np.unravel_index(self._flat_indices, self._reference_shape)
            self.feature_names_ = [
                f"voxel_{x}_{y}_{z}" for x, y, z in zip(coords[0], coords[1], coords[2])
            ]

        self._is_fitted = True
        logger.info(
            f"VoxelFeatureExtractor fitted with {len(self.feature_names_)} voxels"
        )

        return self

    def transform(self, images: List[NiftiImageLike]) -> np.ndarray:
        """
        Extract voxel values from images.

        Parameters
        ----------
        images : List[NiftiImageLike]
            E-field images

        Returns
        -------
        X : np.ndarray
            Feature matrix (n_images, n_voxels)
        """
        self._check_is_fitted()

        # Resample images to reference
        data_arrays, _, _ = _resample_to_reference(images)

        n_samples = len(images)
        n_features = len(self._flat_indices)

        X = np.zeros((n_samples, n_features), dtype=np.float32)

        for i, data in enumerate(data_arrays):
            flat_data = data.ravel()
            X[i, :] = flat_data[self._flat_indices]

        return X

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("VoxelFeatureExtractor must be fitted before transform")

    def get_voxel_coordinates(self) -> List[Tuple[int, int, int]]:
        """Get list of voxel coordinates."""
        self._check_is_fitted()
        coords = np.unravel_index(self._flat_indices, self._reference_shape)
        return list(zip(coords[0], coords[1], coords[2]))


class WholeBrainFeatureExtractor(BaseFeatureExtractor):
    """
    Extract all voxels as features (flattened 3D brain).

    WARNING: This creates very high-dimensional feature spaces and is
    typically only useful when combined with dimensionality reduction.

    Parameters
    ----------
    mask : Optional[np.ndarray]
        Brain mask to restrict voxels

    Examples
    --------
    >>> extractor = WholeBrainFeatureExtractor()
    >>> X = extractor.fit_transform(efield_images)  # Very large X!
    """

    def __init__(self, mask: Optional[np.ndarray] = None):
        super().__init__()
        self.mask = mask
        self._n_voxels: int = 0

    def fit(
        self, images: List[NiftiImageLike], y: Optional[np.ndarray] = None
    ) -> "WholeBrainFeatureExtractor":
        """Fit by determining number of voxels."""
        if not images:
            raise ValueError("No images provided")

        ref_data = np.asanyarray(images[0].dataobj)
        ref_data_3d = _ensure_3d_data(ref_data)

        if self.mask is not None:
            self._n_voxels = int(np.sum(self.mask))
        else:
            self._n_voxels = np.prod(ref_data_3d.shape)

        self.feature_names_ = [f"voxel_{i}" for i in range(self._n_voxels)]
        self._is_fitted = True

        logger.warning(
            f"WholeBrainFeatureExtractor creates {self._n_voxels} features - use with caution!"
        )

        return self

    def transform(self, images: List[NiftiImageLike]) -> np.ndarray:
        """Extract all voxels."""
        self._check_is_fitted()

        data_arrays, _, _ = _resample_to_reference(images)

        if self.mask is not None:
            X = np.array([data[self.mask] for data in data_arrays])
        else:
            X = np.array([data.ravel() for data in data_arrays])

        return X

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "WholeBrainFeatureExtractor must be fitted before transform"
            )


class MetadataFeatureExtractor(BaseFeatureExtractor):
    """
    Extract metadata features from Subject objects.

    This extractor pulls scalar values from subject metadata (like n_stimulation_protocols)
    and includes them as features in the model.

    Parameters
    ----------
    metadata_fields : List[str]
        List of metadata field names to extract as features
    subjects : List[Subject]
        List of Subject objects containing the metadata

    Examples
    --------
    >>> extractor = MetadataFeatureExtractor(
    ...     metadata_fields=['n_stimulation_protocols'],
    ...     subjects=dataset.subjects
    ... )
    >>> X_metadata = extractor.fit_transform(efield_images)  # images are ignored
    """

    def __init__(
        self,
        metadata_fields: List[str],
        subjects: List[Any],
    ):
        super().__init__()
        self.metadata_fields = metadata_fields
        self.subjects = subjects

    def fit(
        self, images: List[NiftiImageLike], y: Optional[np.ndarray] = None
    ) -> "MetadataFeatureExtractor":
        """
        Fit the extractor (validates subjects and fields).

        Parameters
        ----------
        images : List[NiftiImageLike]
            Ignored, present for API compatibility
        y : Optional[np.ndarray]
            Not used, present for API compatibility
        """
        if not self.subjects:
            raise ValueError("No subjects provided")

        # Validate that metadata fields exist and are numeric
        for field in self.metadata_fields:
            values = self._get_field_values(field)
            if all(v is None for v in values):
                logger.warning(f"Metadata field '{field}' is None for all subjects")

        self.feature_names_ = self.metadata_fields
        self._is_fitted = True
        logger.info(
            f"MetadataFeatureExtractor fitted with {len(self.metadata_fields)} fields"
        )

        return self

    def transform(self, images: List[NiftiImageLike]) -> np.ndarray:
        """
        Extract metadata features.

        Parameters
        ----------
        images : List[NiftiImageLike]
            Used to determine expected number of samples.

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape (n_subjects, n_fields)
        """
        self._check_is_fitted()

        n_samples = len(images)
        if len(self.subjects) != n_samples:
            raise ValueError(
                f"Number of subjects ({len(self.subjects)}) does not match "
                f"number of images ({n_samples}). Call set_subjects() with "
                f"the correct subject list before transform."
            )
        n_features = len(self.metadata_fields)
        X = np.zeros((n_samples, n_features), dtype=np.float32)

        for col_idx, field in enumerate(self.metadata_fields):
            values = self._get_field_values(field)
            for row_idx, value in enumerate(values):
                if value is not None:
                    X[row_idx, col_idx] = float(value)

        return X

    def _get_field_values(self, field: str) -> List[Optional[float]]:
        """Get values for a metadata field from all subjects."""
        values = []
        for subject in self.subjects:
            if hasattr(subject, field):
                value = getattr(subject, field)
                values.append(value)
            elif field in subject.metadata:
                value = subject.metadata[field]
                # Try to convert to numeric
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = None
                values.append(value)
            else:
                values.append(None)
        return values

    def set_subjects(self, subjects: List[Any]) -> None:
        """
        Update the subjects used for metadata extraction.

        Call this before ``transform`` when predicting on a different split
        (e.g. validation or test subjects).

        Parameters
        ----------
        subjects : List[Subject]
            New list of subjects whose metadata will be extracted.
        """
        self.subjects = subjects

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "MetadataFeatureExtractor must be fitted before transform"
            )


class SelectedFeatureExtractor(BaseFeatureExtractor):
    """
    Wraps a fitted extractor and keeps only specified feature columns.

    Use this to bake a pre-fitted feature selection into the extraction step
    so the pipeline needs no separate ``feature_selector``.

    Parameters
    ----------
    extractor : BaseFeatureExtractor
        A **fitted** extractor (e.g. ``AtlasFeatureExtractor``).
    selected_indices : array-like of int
        Column indices to keep from the wrapped extractor's output.

    Examples
    --------
    >>> atlas = AtlasFeatureExtractor(atlas_path='atlas.nii.gz', statistics=['mean'])
    >>> atlas.fit(train_images)
    >>> X = atlas.transform(train_images)
    >>>
    >>> selector = TTestSelector(p_threshold=0.05)
    >>> selector.fit(X, y_train)
    >>>
    >>> selected = SelectedFeatureExtractor(atlas, selector.selected_indices_)
    >>> selected.fit(train_images)          # lightweight — just resolves names
    >>> X_sel = selected.transform(images)  # returns only selected columns
    """

    def __init__(
        self,
        extractor: BaseFeatureExtractor,
        selected_indices: Any,
    ):
        super().__init__()
        self.extractor = extractor
        self.selected_indices = np.asarray(selected_indices)

    def fit(
        self, images: List[NiftiImageLike], y: Optional[np.ndarray] = None
    ) -> "SelectedFeatureExtractor":
        """Resolve feature names from the wrapped extractor."""
        all_names = self.extractor.get_feature_names()
        self.feature_names_ = [all_names[i] for i in self.selected_indices]
        self._is_fitted = True
        return self

    def transform(self, images: List[NiftiImageLike]) -> np.ndarray:
        """Extract all features then keep only selected columns."""
        self._check_is_fitted()
        X = self.extractor.transform(images)
        return X[:, self.selected_indices]

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "SelectedFeatureExtractor must be fitted before transform"
            )


class CompositeFeatureExtractor(BaseFeatureExtractor):
    """
    Combine multiple feature extractors.

    Concatenates features from multiple extractors into a single feature matrix.

    Parameters
    ----------
    extractors : List[BaseFeatureExtractor]
        List of feature extractors to combine
    add_prefix : bool
        If True, prefix each extractor's feature names with ``ext{i}_`` to
        avoid collisions.  Default is False.

    Examples
    --------
    >>> extractor1 = AtlasFeatureExtractor(atlas_path='atlas1.nii.gz')
    >>> extractor2 = AtlasFeatureExtractor(atlas_path='atlas2.nii.gz')
    >>> composite = CompositeFeatureExtractor([extractor1, extractor2])
    >>> X = composite.fit_transform(images)
    """

    def __init__(
        self,
        extractors: List[BaseFeatureExtractor],
        add_prefix: bool = False,
    ):
        super().__init__()
        self.extractors = extractors
        self.add_prefix = add_prefix

    def fit(
        self, images: List[NiftiImageLike], y: Optional[np.ndarray] = None
    ) -> "CompositeFeatureExtractor":
        """Fit all extractors."""
        for extractor in self.extractors:
            extractor.fit(images, y)

        # Combine feature names
        self.feature_names_ = []
        for i, extractor in enumerate(self.extractors):
            names = extractor.get_feature_names()
            if self.add_prefix:
                names = [f"ext{i}_{name}" for name in names]
            self.feature_names_.extend(names)

        self._is_fitted = True
        return self

    def transform(self, images: List[NiftiImageLike]) -> np.ndarray:
        """Transform with all extractors and concatenate."""
        self._check_is_fitted()

        feature_blocks = [ext.transform(images) for ext in self.extractors]
        return np.hstack(feature_blocks)

    def set_subjects(self, subjects: List[Any]) -> None:
        """
        Update subjects on any child ``MetadataFeatureExtractor`` instances.

        Parameters
        ----------
        subjects : List[Subject]
            New subjects for metadata extraction.
        """
        for ext in self.extractors:
            if hasattr(ext, "set_subjects"):
                ext.set_subjects(subjects)

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "CompositeFeatureExtractor must be fitted before transform"
            )


class GlobalMeanExtractor(BaseFeatureExtractor):
    """
    Extract global mean E-field intensity from each image.

    Computes the mean of non-zero voxels across the entire brain volume,
    providing a single scalar feature representing overall stimulation intensity.

    Parameters
    ----------
    threshold : float
        Minimum value to consider (default: 0.0). Voxels below this are excluded.

    Examples
    --------
    >>> extractor = GlobalMeanExtractor()
    >>> X = extractor.fit_transform(efield_images)
    >>> print(f"Mean intensity: {X[0, 0]:.4f}")
    """

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold

    def fit(
        self, images: List[NiftiImageLike], y: Optional[np.ndarray] = None
    ) -> "GlobalMeanExtractor":
        """
        Fit the extractor.

        Parameters
        ----------
        images : List[NiftiImageLike]
            E-field images (used to validate)
        y : Optional[np.ndarray]
            Not used, present for API compatibility
        """
        if not images:
            raise ValueError("No images provided")

        self.feature_names_ = ["global_mean_efield"]
        self._is_fitted = True
        logger.info("GlobalMeanExtractor fitted")
        return self

    def transform(self, images: List[NiftiImageLike]) -> np.ndarray:
        """
        Extract global mean E-field from images.

        Parameters
        ----------
        images : List[NiftiImageLike]
            E-field images

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape (n_images, 1)
        """
        self._check_is_fitted()

        means = []
        for img in images:
            data = np.asanyarray(img.dataobj)
            data = _ensure_3d_data(data)

            # Mean of voxels above threshold (exclude background)
            valid_data = data[data > self.threshold]
            mean_val = np.mean(valid_data) if len(valid_data) > 0 else 0.0
            means.append(mean_val)

        return np.array(means).reshape(-1, 1)

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("GlobalMeanExtractor must be fitted before transform")
