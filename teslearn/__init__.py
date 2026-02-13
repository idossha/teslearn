"""
TESLearn: Machine Learning for Transcranial Electrical Stimulation Responder Prediction

A modular, extensible library for predicting responders to transcranial electrical
stimulation using electric field intensity maps in MNI space.
"""

__version__ = "0.1.0"

# Core imports for easy access
from .base import (
    BaseFeatureExtractor,
    BaseFeatureSelector,
    BaseModel,
    BaseValidator,
)
from .data import Subject, Dataset, load_dataset_from_csv
from .features import (
    AtlasFeatureExtractor,
    VoxelFeatureExtractor,
)
from .selection import (
    TTestSelector,
    FRegressionSelector,
    AtlasSelector,
)
from .models import (
    LogisticRegressionModel,
    SVMModel,
)
from .pipeline import TESPipeline
from .train import train_model, cross_validate
from .predict import predict, predict_proba
from .explain import explain_model, ModelExplainer

__all__ = [
    # Version
    "__version__",
    # Base classes
    "BaseFeatureExtractor",
    "BaseFeatureSelector",
    "BaseModel",
    "BaseValidator",
    # Data
    "Subject",
    "Dataset",
    "load_dataset_from_csv",
    # Features
    "AtlasFeatureExtractor",
    "VoxelFeatureExtractor",
    # Selection
    "TTestSelector",
    "FRegressionSelector",
    "AtlasSelector",
    # Models
    "LogisticRegressionModel",
    "SVMModel",
    # Pipeline
    "TESPipeline",
    # Training
    "train_model",
    "cross_validate",
    # Prediction
    "predict",
    "predict_proba",
    # Explanation
    "explain_model",
    "ModelExplainer",
]
