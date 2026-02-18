"""
Data splitting utilities for TESLearn.

Provides train/test splitting functionality that integrates with TESLearn's
Dataset and Subject objects.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np
import logging

from .data import Dataset, Subject

logger = logging.getLogger(__name__)


def train_test_split(
    dataset: Dataset,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    stratify: bool = True,
    random_state: int = 42,
    shuffle: bool = True,
) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
    """
    Split a dataset into train/test or train/val/test sets.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split
    test_size : float
        Fraction of data to use for test set (default: 0.2)
    val_size : Optional[float]
        Fraction of remaining data to use for validation (default: None)
    stratify : bool
        Whether to stratify by target class (default: True)
    random_state : int
        Random seed for reproducibility
    shuffle : bool
        Whether to shuffle before splitting (default: True)

    Returns
    -------
    train_dataset : Dataset
        Training dataset
    val_dataset : Dataset (if val_size provided)
        Validation dataset
    test_dataset : Dataset
        Test dataset

    Examples
    --------
    >>> # Simple train/test split
    >>> train_dataset, test_dataset = train_test_split(
    ...     dataset, test_size=0.2, stratify=True, random_state=42
    ... )

    >>> # Train/val/test split
    >>> train_dataset, val_dataset, test_dataset = train_test_split(
    ...     dataset, test_size=0.15, val_size=0.15, stratify=True, random_state=42
    ... )
    """
    subjects = dataset.subjects.copy()
    n_total = len(subjects)

    if shuffle:
        rng = np.random.RandomState(random_state)
        indices = np.arange(n_total)
        rng.shuffle(indices)
        subjects = [subjects[i] for i in indices]

    # Get stratification targets if needed
    if stratify and dataset.task == "classification":
        targets = np.array([s.target for s in subjects if s.target is not None])
    else:
        targets = None

    # Calculate split indices
    n_test = int(n_total * test_size)
    n_test = max(1, n_test)  # At least 1 sample

    if val_size is not None:
        n_val = int(n_total * val_size)
        n_val = max(1, n_val)
        n_train = n_total - n_val - n_test
    else:
        n_val = 0
        n_train = n_total - n_test

    if stratify and targets is not None and len(targets) == n_total:
        # Stratified split
        from sklearn.model_selection import train_test_split as sk_split

        indices = np.arange(n_total)

        if val_size is not None:
            # Three-way split
            trainval_idx, test_idx = sk_split(
                indices,
                test_size=test_size,
                stratify=targets,
                random_state=random_state,
                shuffle=shuffle,
            )
            # Split trainval into train and val
            val_ratio = val_size / (1 - test_size)
            train_idx, val_idx = sk_split(
                trainval_idx,
                test_size=val_ratio,
                stratify=targets[trainval_idx],
                random_state=random_state,
                shuffle=shuffle,
            )

            train_subjects = [subjects[i] for i in train_idx]
            val_subjects = [subjects[i] for i in val_idx]
            test_subjects = [subjects[i] for i in test_idx]

            train_dataset = Dataset(
                subjects=train_subjects,
                task=dataset.task,
                target_col=dataset.target_col,
            )
            val_dataset = Dataset(
                subjects=val_subjects,
                task=dataset.task,
                target_col=dataset.target_col,
            )
            test_dataset = Dataset(
                subjects=test_subjects,
                task=dataset.task,
                target_col=dataset.target_col,
            )

            logger.info(
                f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val, "
                f"{len(test_dataset)} test"
            )

            return train_dataset, val_dataset, test_dataset
        else:
            # Two-way split
            train_idx, test_idx = sk_split(
                indices,
                test_size=test_size,
                stratify=targets,
                random_state=random_state,
                shuffle=shuffle,
            )

            train_subjects = [subjects[i] for i in train_idx]
            test_subjects = [subjects[i] for i in test_idx]

            train_dataset = Dataset(
                subjects=train_subjects,
                task=dataset.task,
                target_col=dataset.target_col,
            )
            test_dataset = Dataset(
                subjects=test_subjects,
                task=dataset.task,
                target_col=dataset.target_col,
            )

            logger.info(
                f"Split dataset: {len(train_dataset)} train, {len(test_dataset)} test"
            )

            return train_dataset, test_dataset
    else:
        # Simple split without stratification
        if val_size is not None:
            train_subjects = subjects[:n_train]
            val_subjects = subjects[n_train : n_train + n_val]
            test_subjects = subjects[n_train + n_val :]

            train_dataset = Dataset(
                subjects=train_subjects,
                task=dataset.task,
                target_col=dataset.target_col,
            )
            val_dataset = Dataset(
                subjects=val_subjects,
                task=dataset.task,
                target_col=dataset.target_col,
            )
            test_dataset = Dataset(
                subjects=test_subjects,
                task=dataset.task,
                target_col=dataset.target_col,
            )

            logger.info(
                f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val, "
                f"{len(test_dataset)} test"
            )

            return train_dataset, val_dataset, test_dataset
        else:
            train_subjects = subjects[:n_train]
            test_subjects = subjects[n_train:]

            train_dataset = Dataset(
                subjects=train_subjects,
                task=dataset.task,
                target_col=dataset.target_col,
            )
            test_dataset = Dataset(
                subjects=test_subjects,
                task=dataset.task,
                target_col=dataset.target_col,
            )

            logger.info(
                f"Split dataset: {len(train_dataset)} train, {len(test_dataset)} test"
            )

            return train_dataset, test_dataset
