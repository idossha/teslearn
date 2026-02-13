"""
Prediction utilities for TESLearn.

Functions for making predictions with trained models on new data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import logging

from .pipeline import TESPipeline
from .data import Dataset, Subject

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from prediction."""

    subject_ids: List[str]
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    simulation_names: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "subject_id": self.subject_ids,
            "prediction": self.predictions.tolist(),
        }

        if self.simulation_names:
            result["simulation_name"] = self.simulation_names

        if self.probabilities is not None:
            if self.probabilities.ndim == 2:
                result["probability_class_0"] = self.probabilities[:, 0].tolist()
                result["probability_class_1"] = self.probabilities[:, 1].tolist()
            else:
                result["probability"] = self.probabilities.tolist()

        return result

    def to_csv(self, filepath: Union[str, Path]) -> None:
        """Save predictions to CSV."""
        import csv

        filepath = Path(filepath)

        with open(filepath, "w", newline="") as f:
            fieldnames = ["subject_id", "simulation_name", "prediction"]

            if self.probabilities is not None:
                if self.probabilities.ndim == 2:
                    fieldnames.extend(["probability_class_0", "probability_class_1"])
                else:
                    fieldnames.append("probability")

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, subj_id in enumerate(self.subject_ids):
                row = {
                    "subject_id": subj_id,
                    "simulation_name": self.simulation_names[i]
                    if self.simulation_names
                    else "",
                    "prediction": self.predictions[i],
                }

                if self.probabilities is not None:
                    if self.probabilities.ndim == 2:
                        row["probability_class_0"] = self.probabilities[i, 0]
                        row["probability_class_1"] = self.probabilities[i, 1]
                    else:
                        row["probability"] = self.probabilities[i]

                writer.writerow(row)

        logger.info(f"Predictions saved to {filepath}")


def predict(
    pipeline: TESPipeline,
    images: List[Any],
    subject_ids: Optional[List[str]] = None,
    simulation_names: Optional[List[str]] = None,
) -> PredictionResult:
    """
    Make predictions with a trained pipeline.

    Parameters
    ----------
    pipeline : TESPipeline
        Fitted pipeline
    images : List[NiftiImageLike]
        Images to predict on
    subject_ids : Optional[List[str]]
        Subject identifiers
    simulation_names : Optional[List[str]]
        Simulation names

    Returns
    -------
    result : PredictionResult
        Prediction results

    Examples
    --------
    >>> result = predict(pipeline, test_images, subject_ids=test_ids)
    >>> print(result.predictions)
    >>> result.to_csv('predictions.csv')
    """
    if not pipeline._is_fitted:
        raise RuntimeError("Pipeline must be fitted before prediction")

    # Make predictions
    predictions = pipeline.predict(images)

    # Get probabilities if available
    probabilities = None
    if pipeline.is_classifier:
        try:
            probabilities = pipeline.predict_proba(images)
        except Exception as e:
            logger.warning(f"Could not compute probabilities: {e}")

    # Generate IDs if not provided
    if subject_ids is None:
        subject_ids = [f"subject_{i}" for i in range(len(images))]

    if simulation_names is None:
        simulation_names = [f"sim_{i}" for i in range(len(images))]

    return PredictionResult(
        subject_ids=subject_ids,
        predictions=predictions,
        probabilities=probabilities,
        simulation_names=simulation_names,
    )


def predict_proba(
    pipeline: TESPipeline,
    images: List[Any],
    subject_ids: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Predict class probabilities.

    Parameters
    ----------
    pipeline : TESPipeline
        Fitted pipeline
    images : List[NiftiImageLike]
        Images to predict on
    subject_ids : Optional[List[str]]
        Subject identifiers

    Returns
    -------
    probabilities : np.ndarray
        Class probabilities

    Examples
    --------
    >>> proba = predict_proba(pipeline, test_images)
    >>> responder_probability = proba[:, 1]
    """
    if not pipeline.is_classifier:
        raise ValueError("predict_proba only available for classification")

    return pipeline.predict_proba(images)


def predict_dataset(
    pipeline: TESPipeline,
    dataset: Dataset,
    loader: Optional[Any] = None,
) -> PredictionResult:
    """
    Make predictions on a dataset.

    Parameters
    ----------
    pipeline : TESPipeline
        Fitted pipeline
    dataset : Dataset
        Dataset to predict on
    loader : Optional[NiftiLoader]
        Image loader (creates new one if None)

    Returns
    -------
    result : PredictionResult
        Prediction results
    """
    from .data import NiftiLoader

    if loader is None:
        loader = NiftiLoader()

    # Load images
    images, indices = loader.load_dataset_images(dataset)

    # Get metadata for loaded subjects
    subjects = [dataset.subjects[i] for i in indices]
    subject_ids = [s.subject_id for s in subjects]
    simulation_names = [s.simulation_name for s in subjects]

    # Predict
    return predict(pipeline, images, subject_ids, simulation_names)


def evaluate(
    pipeline: TESPipeline,
    images: List[Any],
    y_true: np.ndarray,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Parameters
    ----------
    pipeline : TESPipeline
        Fitted pipeline
    images : List[NiftiImageLike]
        Test images
    y_true : np.ndarray
        True targets
    metrics : Optional[List[str]]
        Metrics to compute

    Returns
    -------
    scores : Dict[str, float]
        Dictionary of metric scores

    Examples
    --------
    >>> scores = evaluate(pipeline, test_images, y_test)
    >>> print(f"ROC AUC: {scores['roc_auc']:.3f}")
    """
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        average_precision_score,
        r2_score,
        mean_squared_error,
        mean_absolute_error,
        confusion_matrix,
    )

    predictions = pipeline.predict(images)
    scores = {}

    if pipeline.is_classifier:
        # Classification metrics
        scores["accuracy"] = accuracy_score(y_true, predictions)
        scores["precision"] = precision_score(y_true, predictions, zero_division=0)
        scores["recall"] = recall_score(y_true, predictions, zero_division=0)
        scores["f1"] = f1_score(y_true, predictions, zero_division=0)

        # Probabilistic metrics
        try:
            proba = pipeline.predict_proba(images)
            if proba.ndim == 2 and proba.shape[1] == 2:
                scores["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
                scores["pr_auc"] = average_precision_score(y_true, proba[:, 1])
            else:
                scores["roc_auc"] = roc_auc_score(y_true, proba)
        except Exception as e:
            logger.warning(f"Could not compute probabilistic metrics: {e}")

        # Confusion matrix
        cm = confusion_matrix(y_true, predictions)
        scores["tn"] = int(cm[0, 0])
        scores["fp"] = int(cm[0, 1])
        scores["fn"] = int(cm[1, 0])
        scores["tp"] = int(cm[1, 1])
    else:
        # Regression metrics
        scores["r2"] = r2_score(y_true, predictions)
        scores["mse"] = mean_squared_error(y_true, predictions)
        scores["rmse"] = np.sqrt(scores["mse"])
        scores["mae"] = mean_absolute_error(y_true, predictions)

    return scores
