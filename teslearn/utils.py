"""
Utility functions for TESLearn.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for TESLearn.

    Parameters
    ----------
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    log_file : Optional[Path]
        Optional file to log to
    format_string : Optional[str]
        Custom format string

    Returns
    -------
    logger : logging.Logger
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get TESLearn logger
    teslearn_logger = logging.getLogger("teslearn")
    teslearn_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    teslearn_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(logging.Formatter(format_string))
    teslearn_logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(logging.Formatter(format_string))
        teslearn_logger.addHandler(file_handler)

    return teslearn_logger


def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    task: str = "classification",
) -> None:
    """
    Validate input data for ML.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    task : str
        'classification' or 'regression'

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same number of samples: {X.shape[0]} != {y.shape[0]}"
        )

    if task == "classification":
        unique = np.unique(y)
        if len(unique) != 2:
            raise ValueError(
                f"Classification requires exactly 2 classes, got {len(unique)}"
            )
        if not all(v in [0, 1] for v in unique):
            raise ValueError("Classification targets must be 0 or 1")
    else:
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("Regression targets must be numeric")


def check_nifti_alignment(
    img1: Any,
    img2: Any,
    tolerance: float = 1e-3,
) -> bool:
    """
    Check if two NIfTI images are aligned.

    Parameters
    ----------
    img1, img2 : NiftiImageLike
        Images to compare
    tolerance : float
        Tolerance for affine comparison

    Returns
    -------
    aligned : bool
        Whether images are aligned
    """
    # Check shape
    if img1.shape != img2.shape:
        return False

    # Check affine
    if not np.allclose(img1.affine, img2.affine, atol=tolerance):
        return False

    return True


def get_nifti_info(img: Any) -> Dict[str, Any]:
    """
    Get information about a NIfTI image.

    Parameters
    ----------
    img : NiftiImageLike
        Input image

    Returns
    -------
    info : Dict[str, Any]
        Dictionary with image information
    """
    data = np.asanyarray(img.dataobj)

    return {
        "shape": img.shape,
        "affine": img.affine,
        "dtype": data.dtype,
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "n_voxels": int(np.prod(data.shape)),
        "n_nonzero": int(np.count_nonzero(data)),
    }


def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Safe division that handles division by zero.

    Parameters
    ----------
    a : np.ndarray
        Numerator
    b : np.ndarray
        Denominator
    fill_value : float
        Value to use when b is 0

    Returns
    -------
    result : np.ndarray
        a / b with zeros replaced by fill_value
    """
    result = np.divide(a, b, out=np.full_like(a, fill_value, dtype=float), where=b != 0)
    return result


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights.

    Parameters
    ----------
    y : np.ndarray
        Target vector

    Returns
    -------
    weights : Dict[int, float]
        Class weights
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(unique)

    weights = {}
    for cls, count in zip(unique, counts):
        weights[int(cls)] = total / (n_classes * count)

    return weights


def memory_usage_mb(arr: np.ndarray) -> float:
    """
    Get memory usage of array in MB.

    Parameters
    ----------
    arr : np.ndarray
        Input array

    Returns
    -------
    mb : float
        Memory usage in megabytes
    """
    return arr.nbytes / (1024 * 1024)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks.

    Parameters
    ----------
    lst : List[Any]
        List to split
    chunk_size : int
        Size of each chunk

    Returns
    -------
    chunks : List[List[Any]]
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def format_time(seconds: float) -> str:
    """
    Format seconds as human-readable time string.

    Parameters
    ----------
    seconds : float
        Time in seconds

    Returns
    -------
    formatted : str
        Formatted string like "1h 23m 45s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_progress(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    bar_length: int = 50,
) -> None:
    """
    Print a progress bar.

    Parameters
    ----------
    current : int
        Current progress
    total : int
        Total items
    prefix : str
        Prefix string
    suffix : str
        Suffix string
    bar_length : int
        Length of progress bar
    """
    percent = 100 * (current / float(total))
    filled_length = int(bar_length * current // total)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    print(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}", end="")

    if current == total:
        print()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists.

    Parameters
    ----------
    path : Union[str, Path]
        Directory path

    Returns
    -------
    path : Path
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
