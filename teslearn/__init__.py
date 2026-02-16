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
from .cv import LeaveOneOutValidator
from .data import Subject, Dataset, load_dataset_from_csv, NiftiLoader, load_nifti_BIDS
from .features import (
    AtlasFeatureExtractor,
    VoxelFeatureExtractor,
    MetadataFeatureExtractor,
    SelectedFeatureExtractor,
    CompositeFeatureExtractor,
)
from .selection import (
    TTestSelector,
    FRegressionSelector,
    AtlasSelector,
    VoxelSelectorFromImages,
)
from .models import (
    LogisticRegressionModel,
    SVMModel,
)
from .pipeline import TESPipeline
from .train import train_model, cross_validate
from .predict import predict, predict_proba
from .explain import explain_model, ModelExplainer
from .split import train_test_split
from .viz import create_stat_map, plot_glass_brain, plot_evaluation, plot_cv_results
from .metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

__all__ = [
    # Version
    "__version__",
    # Base classes
    "BaseFeatureExtractor",
    "BaseFeatureSelector",
    "BaseModel",
    "BaseValidator",
    "LeaveOneOutValidator",
    # Data
    "Subject",
    "Dataset",
    "load_dataset_from_csv",
    "NiftiLoader",
    "load_nifti_BIDS",
    # Features
    "AtlasFeatureExtractor",
    "VoxelFeatureExtractor",
    "MetadataFeatureExtractor",
    "SelectedFeatureExtractor",
    "CompositeFeatureExtractor",
    # Selection
    "TTestSelector",
    "FRegressionSelector",
    "AtlasSelector",
    "VoxelSelectorFromImages",
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
    # Split
    "train_test_split",
    # Visualization
    "create_stat_map",
    "plot_glass_brain",
    "plot_evaluation",
    "plot_cv_results",
    # Metrics
    "accuracy_score",
    "roc_auc_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "classification_report",
]
