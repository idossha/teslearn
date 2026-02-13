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

        if not self.atlas_path.exists():
            raise FileNotFoundError(f"Atlas not found: {self.atlas_path}")

        # Load atlas
        atlas_img = nib.load(str(self.atlas_path))
        self._atlas_data = np.asanyarray(atlas_img.dataobj).astype(np.int32)

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


class CompositeFeatureExtractor(BaseFeatureExtractor):
    """
    Combine multiple feature extractors.

    Concatenates features from multiple extractors into a single feature matrix.

    Parameters
    ----------
    extractors : List[BaseFeatureExtractor]
        List of feature extractors to combine

    Examples
    --------
    >>> extractor1 = AtlasFeatureExtractor(atlas_path='atlas1.nii.gz')
    >>> extractor2 = AtlasFeatureExtractor(atlas_path='atlas2.nii.gz')
    >>> composite = CompositeFeatureExtractor([extractor1, extractor2])
    >>> X = composite.fit_transform(images)
    """

    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super().__init__()
        self.extractors = extractors

    def fit(
        self, images: List[NiftiImageLike], y: Optional[np.ndarray] = None
    ) -> "CompositeFeatureExtractor":
        """Fit all extractors."""
        for extractor in self.extractors:
            extractor.fit(images, y)

        # Combine feature names
        self.feature_names_ = []
        for i, extractor in enumerate(self.extractors):
            prefix = f"ext{i}_"
            self.feature_names_.extend(
                [f"{prefix}{name}" for name in extractor.get_feature_names()]
            )

        self._is_fitted = True
        return self

    def transform(self, images: List[NiftiImageLike]) -> np.ndarray:
        """Transform with all extractors and concatenate."""
        self._check_is_fitted()

        feature_blocks = [ext.transform(images) for ext in self.extractors]
        return np.hstack(feature_blocks)

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "CompositeFeatureExtractor must be fitted before transform"
            )
