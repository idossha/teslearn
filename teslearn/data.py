"""
Data management for TESLearn.

Handles loading and management of E-field NIfTI images, subject metadata,
and target variables for responder prediction.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Subject:
    """
    Represents a subject in the study.

    Attributes
    ----------
    subject_id : str
        Unique subject identifier
    simulation_name : str
        Name of the stimulation simulation (e.g., montage configuration)
    condition : Optional[str]
        Experimental condition (e.g., 'active', 'sham')
    target : Optional[float]
        Target variable (0/1 for classification, continuous for regression)
    efield_path : Optional[Path]
        Path to the E-field NIfTI file
    metadata : Dict[str, Any]
        Additional subject-level metadata
    """

    subject_id: str
    simulation_name: str
    condition: Optional[str] = None
    target: Optional[float] = None
    efield_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_sham(self, sham_value: str = "sham") -> bool:
        """Check if this subject is a sham condition."""
        if self.condition is None:
            return False
        return self.condition.strip().lower() == sham_value.strip().lower()

    @property
    def has_efield(self) -> bool:
        """Check if subject has an associated E-field file."""
        return self.efield_path is not None and self.efield_path.exists()


@dataclass
class Dataset:
    """
    Dataset for TES responder prediction.

    Contains subjects with their E-field images and target variables.

    Attributes
    ----------
    subjects : List[Subject]
        List of subjects in the dataset
    task : str
        ML task type ('classification' or 'regression')
    target_col : str
        Name of the target column
    """

    subjects: List[Subject]
    task: str = "classification"
    target_col: str = "response"

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Subject:
        return self.subjects[idx]

    @property
    def n_subjects(self) -> int:
        """Total number of subjects."""
        return len(self.subjects)

    @property
    def n_active(self) -> int:
        """Number of active (non-sham) subjects."""
        return sum(1 for s in self.subjects if not s.is_sham())

    @property
    def n_sham(self) -> int:
        """Number of sham subjects."""
        return sum(1 for s in self.subjects if s.is_sham())

    @property
    def has_efield_count(self) -> int:
        """Number of subjects with E-field data."""
        return sum(1 for s in self.subjects if s.has_efield)

    def get_targets(self) -> np.ndarray:
        """Get target array for all subjects with targets."""
        targets = [s.target for s in self.subjects if s.target is not None]
        if self.task == "classification":
            return np.array(targets, dtype=int)
        return np.array(targets, dtype=float)

    def get_efield_paths(self) -> List[Optional[Path]]:
        """Get list of E-field paths for all subjects."""
        return [s.efield_path for s in self.subjects]

    def filter_active(self) -> Dataset:
        """Return dataset with only active (non-sham) subjects."""
        return Dataset(
            subjects=[s for s in self.subjects if not s.is_sham()],
            task=self.task,
            target_col=self.target_col,
        )

    def filter_with_efield(self) -> Dataset:
        """Return dataset with only subjects that have E-field data."""
        return Dataset(
            subjects=[s for s in self.subjects if s.has_efield],
            task=self.task,
            target_col=self.target_col,
        )

    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution for classification tasks."""
        if self.task != "classification":
            raise ValueError("Class distribution only available for classification")
        targets = self.get_targets()
        unique, counts = np.unique(targets, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def validate(self) -> List[str]:
        """Validate dataset and return list of issues."""
        issues = []

        if not self.subjects:
            issues.append("No subjects in dataset")
            return issues

        # Check for duplicate subject IDs
        ids = [s.subject_id for s in self.subjects]
        if len(ids) != len(set(ids)):
            issues.append("Duplicate subject IDs found")

        # Check targets
        if self.task == "classification":
            targets = self.get_targets()
            if not all(t in [0, 1] for t in targets):
                issues.append("Classification targets must be 0 or 1")

        # Check E-field files
        missing_files = [
            s.subject_id for s in self.subjects if not s.is_sham() and not s.has_efield
        ]
        if missing_files:
            issues.append(f"Missing E-field files for subjects: {missing_files[:5]}...")

        return issues


def resolve_efield_path(
    subject_id: str,
    simulation_name: str,
    base_dir: Optional[Path] = None,
    pattern: str = "{subject_id}/{simulation_name}/efield.nii.gz",
) -> Path:
    """
    Resolve E-field file path from subject ID and simulation name.

    Parameters
    ----------
    subject_id : str
        Subject identifier
    simulation_name : str
        Simulation/montage name
    base_dir : Optional[Path]
        Base directory for derivatives
    pattern : str
        Path pattern with {subject_id} and {simulation_name} placeholders

    Returns
    -------
    path : Path
        Resolved path to E-field NIfTI file
    """
    path_str = pattern.format(subject_id=subject_id, simulation_name=simulation_name)
    if base_dir is not None:
        return base_dir / path_str
    return Path(path_str)


def load_dataset_from_csv(
    csv_path: Union[str, Path],
    efield_base_dir: Optional[Union[str, Path]] = None,
    target_col: str = "response",
    condition_col: Optional[str] = None,
    sham_value: str = "sham",
    task: str = "classification",
    efield_pattern: str = "{subject_id}/{simulation_name}/efield.nii.gz",
    require_target: bool = True,
) -> Dataset:
    """
    Load dataset from CSV file.

    Expected CSV columns:
    - subject_id: Subject identifier
    - simulation_name: Stimulation configuration name
    - {target_col}: Target variable (required if require_target=True)
    - {condition_col}: Experimental condition (optional)

    Parameters
    ----------
    csv_path : Union[str, Path]
        Path to CSV file
    efield_base_dir : Optional[Union[str, Path]]
        Base directory for E-field files
    target_col : str
        Name of target column
    condition_col : Optional[str]
        Name of condition column
    sham_value : str
        Value indicating sham condition
    task : str
        'classification' or 'regression'
    efield_pattern : str
        Pattern for resolving E-field paths
    require_target : bool
        Whether target column is required

    Returns
    -------
    dataset : Dataset
        Loaded dataset

    Raises
    ------
    FileNotFoundError
        If CSV file not found
    ValueError
        If required columns missing or invalid data
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if efield_base_dir is not None:
        efield_base_dir = Path(efield_base_dir)

    subjects = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError("CSV has no header row")

        # Check required columns
        required = {"subject_id", "simulation_name"}
        if require_target:
            required.add(target_col)
        if condition_col:
            required.add(condition_col)

        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        for i, row in enumerate(reader, start=2):
            subject_id = row.get("subject_id", "").strip()
            simulation_name = row.get("simulation_name", "").strip()

            if not subject_id:
                raise ValueError(f"Empty subject_id on line {i}")

            # Get condition
            condition = None
            if condition_col:
                condition = row.get(condition_col, "").strip() or None

            # Get target
            target = None
            if require_target or target_col in row:
                target_str = row.get(target_col, "").strip()
                if target_str:
                    if task == "classification":
                        try:
                            target = float(int(target_str))
                            if target not in [0, 1]:
                                raise ValueError(f"Target must be 0 or 1, got {target}")
                        except ValueError as e:
                            raise ValueError(
                                f"Invalid target on line {i}: {target_str}"
                            ) from e
                    else:
                        try:
                            target = float(target_str)
                        except ValueError as e:
                            raise ValueError(
                                f"Invalid target on line {i}: {target_str}"
                            ) from e

            # Resolve E-field path (allow empty for sham)
            is_sham = (
                condition and condition.strip().lower() == sham_value.strip().lower()
            )
            if is_sham:
                efield_path = None
            else:
                efield_path = resolve_efield_path(
                    subject_id=subject_id,
                    simulation_name=simulation_name,
                    base_dir=efield_base_dir,
                    pattern=efield_pattern,
                )

            subject = Subject(
                subject_id=subject_id,
                simulation_name=simulation_name,
                condition=condition,
                target=target,
                efield_path=efield_path,
                metadata={k: v for k, v in row.items() if k not in required},
            )
            subjects.append(subject)

    if not subjects:
        raise ValueError("No subjects loaded from CSV")

    logger.info(f"Loaded {len(subjects)} subjects from {csv_path}")

    return Dataset(subjects=subjects, task=task, target_col=target_col)


class NiftiLoader:
    """
    Lazy loader for NIfTI images.

    Loads images on-demand and caches them to avoid repeated I/O.
    """

    def __init__(self, cache_size: int = 10):
        self._cache: Dict[Path, Any] = {}
        self._cache_size = cache_size
        self._cache_order: List[Path] = []

    def load(self, path: Path) -> Any:
        """Load NIfTI image, using cache if available."""
        import nibabel as nib

        path = Path(path)

        if path in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(path)
            self._cache_order.append(path)
            return self._cache[path]

        if not path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {path}")

        img = nib.load(str(path))

        # Add to cache
        self._cache[path] = img
        self._cache_order.append(path)

        # Evict oldest if cache is full
        if len(self._cache) > self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        return img

    def load_dataset_images(self, dataset: Dataset) -> Tuple[List[Any], List[int]]:
        """
        Load all E-field images for a dataset.

        Returns
        -------
        images : List[Any]
            List of loaded NIfTI images
        indices : List[int]
            Indices of subjects that were successfully loaded
        """
        images = []
        indices = []

        for i, subject in enumerate(dataset.subjects):
            if subject.is_sham() or not subject.has_efield:
                continue

            try:
                img = self.load(subject.efield_path)
                images.append(img)
                indices.append(i)
            except FileNotFoundError:
                logger.warning(
                    f"Could not load E-field for subject {subject.subject_id}"
                )
                continue

        return images, indices

    def clear_cache(self):
        """Clear the image cache."""
        self._cache.clear()
        self._cache_order.clear()
