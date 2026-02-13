Examples
========

Complete Example: Atlas-Based Classification
--------------------------------------------

.. code-block:: python

   import teslearn as tl
   from teslearn.data import load_dataset_from_csv, NiftiLoader
   from teslearn.features import AtlasFeatureExtractor
   from teslearn.models import LogisticRegressionModel
   from teslearn.selection import TTestSelector
   from teslearn.explain import explain_model

   # 1. Load data
   dataset = load_dataset_from_csv(
       csv_path='subjects.csv',
       target_col='response',
       task='classification'
   )

   loader = NiftiLoader()
   images, indices = loader.load_dataset_images(dataset)
   y = dataset.get_targets()

   # 2. Set up components
   extractor = AtlasFeatureExtractor(
       atlas_path='atlas/aseg.nii.gz',
       statistics=['mean', 'top10mean']
   )

   selector = TTestSelector(p_threshold=0.001)

   model = LogisticRegressionModel(
       C=1.0,
       penalty='l2',
       class_weight='balanced'
   )

   # 3. Train with nested CV
   result = tl.train_model(
       images=images,
       y=y,
       feature_extractor=extractor,
       model=model,
       feature_selector=selector,
       outer_folds=5,
       inner_folds=3,
       use_scaling=True
   )

   print(result.get_summary())

   # 4. Explain model
   explanation = explain_model(
       pipeline=result.pipeline,
       atlas_path='atlas/aseg.nii.gz',
       create_weight_maps=True,
       output_dir='./explanations'
   )

   print(explanation.get_summary())

Voxel-Based Analysis
--------------------

.. code-block:: python

   from teslearn.selection import VoxelSelectorFromImages
   from teslearn.features import VoxelFeatureExtractor

   # Select voxels statistically
   voxel_selector = VoxelSelectorFromImages(
       p_threshold=0.001,
       test_type='ttest'
   )
   voxel_selector.fit(images, y)

   # Get selected coordinates
   coords = voxel_selector.get_voxel_coordinates()
   print(f"Selected {len(coords)} voxels")

   # Create extractor
   extractor = voxel_selector.create_voxel_extractor()

   # Train model
   result = tl.train_model(
       images=images,
       y=y,
       feature_extractor=extractor,
       model=tl.LogisticRegressionModel(),
       use_cross_validation=True
   )

Using SVM Instead of Logistic Regression
----------------------------------------

.. code-block:: python

   from teslearn.models import SVMModel

   model = SVMModel(
       kernel='rbf',
       C=1.0,
       gamma='scale',
       class_weight='balanced',
       probability=True
   )

   result = tl.train_model(
       images=images,
       y=y,
       feature_extractor=extractor,
       model=model,
       feature_selector=selector
   )

Regression Example
------------------

.. code-block:: python

   from teslearn.models import ElasticNetModel
   from teslearn.selection import FRegressionSelector

   # Load continuous targets
   dataset = load_dataset_from_csv(
       csv_path='subjects.csv',
       target_col='improvement_score',
       task='regression'
   )

   # Use regression-specific components
   selector = FRegressionSelector(p_threshold=0.001)
   model = ElasticNetModel(alpha=0.5, l1_ratio=0.3)

   result = tl.train_model(
       images=images,
       y=y,
       feature_extractor=extractor,
       model=model,
       feature_selector=selector
   )

   print(f"R²: {result.mean_r2:.4f}")

Small Sample Size: Leave-One-Out
--------------------------------

.. code-block:: python

   from teslearn.cv import LeaveOneOutValidator

   outer_cv = LeaveOneOutValidator()
   inner_cv = StratifiedKFoldValidator(n_splits=3)

   result = tl.cross_validate(
       pipeline=pipeline,
       images=images,
       y=y,
       outer_validator=outer_cv,
       inner_validator=inner_cv
   )

Working with Sham Subjects
--------------------------

.. code-block:: python

   dataset = load_dataset_from_csv(
       csv_path='subjects.csv',
       target_col='response',
       condition_col='condition',
       sham_value='sham'
   )

   # Sham subjects will have zero features
   # Active subjects use their E-field data
   print(f"Active: {dataset.n_active}, Sham: {dataset.n_sham}")

Batch Prediction
----------------

.. code-block:: python

   # Load trained model
   from teslearn.io import ModelIO
   pipeline, metadata = ModelIO.load('trained_model.pkl')

   # Load new data
   new_dataset = load_dataset_from_csv('new_subjects.csv')
   new_images, indices = loader.load_dataset_images(new_dataset)

   # Predict
   result = tl.predict(
       pipeline=pipeline,
       images=new_images,
       subject_ids=[new_dataset.subjects[i].subject_id for i in indices]
   )

   # Save results
   result.to_csv('predictions.csv')

   # Evaluate if targets available
   if new_dataset.has_targets:
       scores = tl.evaluate(
           pipeline=pipeline,
           images=new_images,
           y_true=new_dataset.get_targets()
       )
       print(f"ROC AUC: {scores['roc_auc']:.3f}")