# TESLearn

**TESLearn** is a modular, extensible Python library for machine learning-based prediction of responders to transcranial electrical stimulation (TES). It provides tools for analyzing electric field intensity maps in MNI space to predict treatment response.

## Features

- **Modular Architecture**: Built on abstract base classes for easy extension
- **Multiple Feature Extraction Methods**:
  - Atlas-based ROI features (mean, max, top-percentile)
  - Voxel-based features with statistical selection
  - Whole-brain flattening
- **Feature Selection**:
  - Mass univariate t-test (default for classification)
  - F-regression (for continuous targets)
  - Atlas-based ROI selection
- **Machine Learning Models**:
  - Logistic Regression with L1/L2/ElasticNet (default)
  - Support Vector Machines
  - Random Forest
  - Elastic Net regression
- **Cross-Validation**: Nested CV with hyperparameter tuning
- **Model Interpretation**: Feature importance, weight maps, and visualizations
- **NIfTI Integration**: Built on nibabel for seamless neuroimaging workflows

## Installation

### Using UV (Recommended)

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable package management.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate  # On macOS/Linux
# OR: .venv\Scripts\activate  # On Windows

# Install with dependencies
uv pip install -e ".[plotting,jupyter]"  # For notebooks and visualization
uv pip install -e ".[all]"               # Install everything
```

See [UV_SETUP.md](UV_SETUP.md) for detailed instructions.

### From PyPI (when available)

```bash
pip install teslearn
```

### From Source (Traditional)

```bash
git clone https://github.com/yourusername/teslearn.git
cd teslearn
pip install -e .
```

### With Optional Dependencies

```bash
# Include plotting support
pip install "teslearn[plotting]"

# Jupyter notebook support
pip install "teslearn[jupyter]"

# Development dependencies
pip install "teslearn[dev]"

# All extras
pip install "teslearn[all]"
```

## Quick Start

### 1. Load Your Data

```python
import teslearn as tl

# Load dataset from CSV
dataset = tl.load_dataset_from_csv(
    csv_path='subjects.csv',
    efield_base_dir='./derivatives/efields/',
    target_col='response',
    task='classification'
)

# Load E-field images
from teslearn.data import NiftiLoader
loader = NiftiLoader()
images, indices = loader.load_dataset_images(dataset)
y = dataset.get_targets()
```

### 2. Train a Model (Default Configuration)

The default configuration uses:
- **Logistic Regression** with L2 regularization
- **T-test feature selection** (for classification)
- **Nested cross-validation** for unbiased performance estimation

```python
# Use default configuration
from teslearn.features import AtlasFeatureExtractor
from teslearn.models import LogisticRegressionModel
from teslearn.selection import TTestSelector

# Set up components
extractor = AtlasFeatureExtractor(
    atlas_path='path/to/atlas.nii.gz',
    statistics=['mean', 'top10mean']
)

selector = TTestSelector(p_threshold=0.001)

model = LogisticRegressionModel(
    C=1.0,
    penalty='l2',
    class_weight='balanced'
)

# Train with cross-validation
result = tl.train_model(
    images=images,
    y=y,
    feature_extractor=extractor,
    model=model,
    feature_selector=selector,
    outer_folds=5,
    inner_folds=3
)

print(result.get_summary())
```

### 3. Make Predictions

```python
# Load test data
test_images, test_indices = loader.load_dataset_images(test_dataset)

# Predict
test_result = tl.predict(
    pipeline=result.pipeline,
    images=test_images,
    subject_ids=[dataset.subjects[i].subject_id for i in test_indices]
)

# Save predictions
test_result.to_csv('predictions.csv')
```

### 4. Explain the Model

```python
from teslearn.explain import explain_model

explanation = explain_model(
    pipeline=result.pipeline,
    atlas_path='path/to/atlas.nii.gz',
    create_weight_maps=True,
    output_dir='./explanations'
)

print(explanation.get_summary())
```

## Advanced Usage

### Custom Feature Extraction

```python
from teslearn.features import VoxelFeatureExtractor

# Use voxel-based features with pre-selected coordinates
coords = [(45, 50, 30), (46, 50, 30), ...]  # From prior analysis
extractor = VoxelFeatureExtractor(voxel_coords=coords)
```

### Using SVM Instead of Logistic Regression

```python
from teslearn.models import SVMModel

model = SVMModel(
    kernel='rbf',
    C=1.0,
    class_weight='balanced'
)

result = tl.train_model(
    images=images,
    y=y,
    feature_extractor=extractor,
    model=model,
    feature_selector=selector
)
```

### Custom Cross-Validation Strategy

```python
from teslearn.cv import LeaveOneOutValidator, StratifiedKFoldValidator

# For very small datasets
outer_cv = LeaveOneOutValidator()
inner_cv = StratifiedKFoldValidator(n_splits=3)

from teslearn.train import cross_validate

result = cross_validate(
    pipeline=pipeline,
    images=images,
    y=y,
    outer_validator=outer_cv,
    inner_validator=inner_cv
)
```

### Working with Pipelines Directly

```python
from teslearn.pipeline import TESPipeline

# Build custom pipeline
pipeline = TESPipeline(
    feature_extractor=extractor,
    feature_selector=selector,
    model=model,
    use_scaling=True
)

# Fit
pipeline.fit(images, y)

# Predict
predictions = pipeline.predict(test_images)
probabilities = pipeline.predict_proba(test_images)

# Get feature importance
importance = pipeline.get_feature_importance()
```

## CSV Format

The expected CSV format for `load_dataset_from_csv()`:

```csv
subject_id,simulation_name,response,condition
sub-001,montage_A,1,active
sub-002,montage_A,0,active
sub-003,sham,0,sham
sub-004,montage_B,1,active
```

Required columns:
- `subject_id`: Unique subject identifier
- `simulation_name`: Stimulation configuration name
- `{target_col}`: Target variable (0/1 for classification, continuous for regression)

Optional columns:
- `condition`: Experimental condition (e.g., 'active', 'sham')

## Architecture

TESLearn is built on a modular architecture with abstract base classes:

- **`BaseFeatureExtractor`**: Interface for feature extraction from NIfTI images
- **`BaseFeatureSelector`**: Interface for dimensionality reduction
- **`BaseModel`**: Interface for ML models
- **`BaseValidator`**: Interface for CV strategies

This design allows you to:
1. Implement custom feature extractors
2. Add new ML models
3. Create domain-specific validators
4. Extend functionality without modifying core code

## Documentation

For full documentation, visit: https://teslearn.readthedocs.io


## License

TESLearn is licensed under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Acknowledgments

TESLearn draws inspiration from [nilearn](https://nilearn.github.io/) and is built on:
- [scikit-learn](https://scikit-learn.org/) for machine learning
- [nibabel](https://nipy.org/nibabel/) for neuroimaging I/O
- [scipy](https://scipy.org/) for statistical functions
