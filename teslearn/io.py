"""
I/O utilities for TESLearn.

Functions for saving and loading models, results, and data.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

from .pipeline import TESPipeline

logger = logging.getLogger(__name__)


class ModelIO:
    """
    Handle saving and loading of TESLearn models.

    Uses pickle/joblib for model serialization with metadata.
    """

    @staticmethod
    def save(
        pipeline: TESPipeline,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a fitted pipeline.

        Parameters
        ----------
        pipeline : TESPipeline
            Fitted pipeline to save
        filepath : Union[str, Path]
            Output file path (.pkl or .joblib)
        metadata : Optional[Dict]
            Additional metadata to save
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create bundle
        bundle = {
            "pipeline": pipeline,
            "metadata": metadata or {},
            "version": "0.1.0",
        }

        # Save
        if filepath.suffix == ".joblib":
            try:
                import joblib

                joblib.dump(bundle, filepath)
            except ImportError:
                logger.warning("joblib not available, using pickle instead")
                with open(filepath.with_suffix(".pkl"), "wb") as f:
                    pickle.dump(bundle, f)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(bundle, f)

        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: Union[str, Path]) -> tuple[TESPipeline, Dict[str, Any]]:
        """
        Load a saved pipeline.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to saved model

        Returns
        -------
        pipeline : TESPipeline
            Loaded pipeline
        metadata : Dict[str, Any]
            Associated metadata
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load
        if filepath.suffix == ".joblib":
            try:
                import joblib

                bundle = joblib.load(filepath)
            except ImportError:
                raise ImportError("joblib required to load .joblib files")
        else:
            with open(filepath, "rb") as f:
                bundle = pickle.load(f)

        pipeline = bundle["pipeline"]
        metadata = bundle.get("metadata", {})

        logger.info(f"Model loaded from {filepath}")

        return pipeline, metadata


def save_json(
    data: Dict[str, Any],
    filepath: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Save data to JSON file.

    Parameters
    ----------
    data : Dict[str, Any]
        Data to save
    filepath : Union[str, Path]
        Output path
    indent : int
        JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays and other types
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        elif hasattr(obj, "item"):
            return obj.item()
        return obj

    with open(filepath, "w") as f:
        json.dump(convert(data), f, indent=indent)

    logger.info(f"JSON saved to {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to JSON file

    Returns
    -------
    data : Dict[str, Any]
        Loaded data
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    return data


def save_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    prefix: str = "results",
) -> Dict[str, Path]:
    """
    Save multiple results to output directory.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary of results to save
    output_dir : Union[str, Path]
        Output directory
    prefix : str
        Prefix for filenames

    Returns
    -------
    paths : Dict[str, Path]
        Paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    for key, value in results.items():
        # Determine file type
        if isinstance(value, TESPipeline):
            path = output_dir / f"{prefix}_{key}.pkl"
            ModelIO.save(value, path)
            saved_paths[key] = path
        elif isinstance(value, dict):
            path = output_dir / f"{prefix}_{key}.json"
            save_json(value, path)
            saved_paths[key] = path
        elif isinstance(value, str) and value.startswith("figure:"):
            # Skip figures, they're handled separately
            continue
        else:
            # Generic pickle
            path = output_dir / f"{prefix}_{key}.pkl"
            with open(path, "wb") as f:
                pickle.dump(value, f)
            saved_paths[key] = path

    return saved_paths
