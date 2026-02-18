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
from .cv import (
    LeaveOneOutValidator,
    StratifiedKFoldValidator,
    KFoldValidator,
    permutation_test,
    PermutationTestResult,
    plot_permutation_test,
)
from .data import Subject, Dataset, load_dataset_from_csv, NiftiLoader, load_nifti_BIDS
from .features import (
    AtlasFeatureExtractor,
    VoxelFeatureExtractor,
    MetadataFeatureExtractor,
    SelectedFeatureExtractor,
    CompositeFeatureExtractor,
    GlobalMeanExtractor,
)
from .selection import (
    TTestSelector,
    FRegressionSelector,
    AtlasSelector,
    VoxelSelectorFromImages,
    PermutationClusterSelector,
)
from .models import (
    LogisticRegressionModel,
    SVMModel,
)
from .pipeline import TESPipeline
from .train import train_model, cross_validate
from .predict import predict, predict_proba
from .explain import (
    explain_model,
    ModelExplainer,
    ROIRankingResult,
)
from .split import train_test_split
from .viz import create_stat_map, plot_glass_brain, plot_evaluation, plot_cv_results
from .plotting import (
    plot_feature_importance,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_prediction_distribution,
    plot_training_history,
    plot_intensity_response,
    prepare_intensity_response_data,
    plot_intensity_response_from_pipeline,
)
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
    "StratifiedKFoldValidator",
    "KFoldValidator",
    "permutation_test",
    "PermutationTestResult",
    "plot_permutation_test",
    "ROIRankingResult",
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
    "GlobalMeanExtractor",
    # Selection
    "TTestSelector",
    "FRegressionSelector",
    "AtlasSelector",
    "VoxelSelectorFromImages",
    "PermutationClusterSelector",
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
    "plot_feature_importance",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_confusion_matrix",
    "plot_prediction_distribution",
    "plot_training_history",
    "plot_intensity_response",
    "prepare_intensity_response_data",
    "plot_intensity_response_from_pipeline",
    # Metrics
    "accuracy_score",
    "roc_auc_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "classification_report",
]
