User Guide
==========

This guide provides detailed information about using TESLearn for TES responder prediction.

Feature Extraction
------------------

Atlas-Based Features
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.features import AtlasFeatureExtractor

   extractor = AtlasFeatureExtractor(
       atlas_path='path/to/atlas.nii.gz',
       statistics=['mean', 'max', 'top10mean'],
       top_percentile=90.0
   )

   X = extractor.fit_transform(images)

Available statistics:

- ``mean``: Mean intensity in ROI
- ``max``: Maximum intensity in ROI
- ``min``: Minimum intensity in ROI
- ``std``: Standard deviation in ROI
- ``median``: Median intensity in ROI
- ``sum``: Sum of intensities in ROI
- ``top10mean``: Mean of top 10% intensities

Voxel-Based Features
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.features import VoxelFeatureExtractor

   # Use pre-selected voxel coordinates
   coords = [(45, 50, 30), (46, 50, 30), ...]
   extractor = VoxelFeatureExtractor(voxel_coords=coords)

   X = extractor.fit_transform(images)

Feature Selection
-----------------

T-Test Selection (Classification)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.selection import TTestSelector

   selector = TTestSelector(
       p_threshold=0.001,
       correction=None  # or 'bonferroni', 'fdr'
   )

   X_selected = selector.fit_transform(X, y)

F-Regression Selection (Continuous)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.selection import FRegressionSelector

   selector = FRegressionSelector(p_threshold=0.001)
   X_selected = selector.fit_transform(X, y)

Voxel Selection from Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.selection import VoxelSelectorFromImages

   selector = VoxelSelectorFromImages(
       p_threshold=0.001,
       test_type='ttest'  # or 'fregression'
   )

   selector.fit(images, y)
   selected_coords = selector.get_voxel_coordinates()

   # Create extractor for selected voxels
   extractor = selector.create_voxel_extractor()

Machine Learning Models
-----------------------

Logistic Regression (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.models import LogisticRegressionModel

   model = LogisticRegressionModel(
       C=1.0,              # Inverse regularization strength
       penalty='l2',       # 'l1', 'l2', or 'elasticnet'
       solver='lbfgs',     # Optimization algorithm
       max_iter=1000,
       class_weight='balanced',
       random_state=42
   )

Support Vector Machine
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.models import SVMModel

   model = SVMModel(
       kernel='rbf',       # 'linear', 'rbf', 'poly', 'sigmoid'
       C=1.0,
       gamma='scale',
       class_weight='balanced',
       probability=True    # Enable probability estimates
   )

Elastic Net (Regression)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.models import ElasticNetModel

   model = ElasticNetModel(
       alpha=1.0,
       l1_ratio=0.5,       # 0=Ridge, 1=Lasso
       max_iter=1000
   )

Cross-Validation
----------------

Nested Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.cv import NestedCrossValidator
   from teslearn.cv import StratifiedKFoldValidator

   outer = StratifiedKFoldValidator(n_splits=5)
   inner = StratifiedKFoldValidator(n_splits=3)

   nested = NestedCrossValidator(outer, inner)

   for fold_idx, X_tr, X_te, y_tr, y_te in nested.split(X, y):
       # Train and evaluate
       pass

Leave-One-Out
~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.cv import LeaveOneOutValidator

   loocv = LeaveOneOutValidator()

   for train_idx, test_idx in loocv.split(X, y):
       # Train and evaluate
       pass

Hyperparameter Tuning
---------------------

.. code-block:: python

   from sklearn.model_selection import GridSearchCV

   param_grid = {
       'C': [0.1, 1.0, 10.0],
       'penalty': ['l1', 'l2'],
   }

   search = GridSearchCV(
       model,
       param_grid,
       cv=inner_validator,
       scoring='roc_auc'
   )

   search.fit(X_train, y_train)
   best_model = search.best_estimator_

Model Interpretation
--------------------

Feature Importance
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.explain import explain_model

   explanation = explain_model(
       pipeline=result.pipeline,
       atlas_path='atlas.nii.gz'
   )

   # Get top features
   top_positive = explanation.top_positive[:10]
   top_negative = explanation.top_negative[:10]

Weight Maps
~~~~~~~~~~~

.. code-block:: python

   from teslearn.explain import ModelExplainer

   explainer = ModelExplainer(pipeline, atlas_path='atlas.nii.gz')
   explainer.create_weight_map(output_path='weight_map.nii.gz')

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.plotting import (
       plot_feature_importance,
       plot_roc_curve,
       plot_confusion_matrix
   )

   # Feature importance
   plot_feature_importance(
       importance=explanation.feature_importance,
       output_path='feature_importance.png'
   )

   # ROC curve
   plot_roc_curve(y_true, y_score, output_path='roc.png')

   # Confusion matrix
   plot_confusion_matrix(y_true, y_pred, output_path='confusion_matrix.png')

Working with Pipelines
----------------------

Building Custom Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.pipeline import TESPipeline

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

Saving and Loading Models
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.io import ModelIO

   # Save
   ModelIO.save(pipeline, 'model.pkl', metadata={'version': '1.0'})

   # Load
   pipeline, metadata = ModelIO.load('model.pkl')