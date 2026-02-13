Quickstart
==========

Installation
------------

Install TESLearn using pip:

.. code-block:: bash

   pip install teslearn

For additional features:

.. code-block:: bash

   pip install "teslearn[plotting]"  # Include matplotlib
   pip install "teslearn[all]"        # All extras

Basic Usage
-----------

1. Load Your Data
~~~~~~~~~~~~~~~~~

.. code-block:: python

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

2. Train a Model (Default Configuration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default configuration uses:

- **Logistic Regression** with L2 regularization
- **T-test feature selection** (for classification)  
- **Nested cross-validation** for unbiased performance estimation

.. code-block:: python

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

3. Make Predictions
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

4. Explain the Model
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from teslearn.explain import explain_model

   explanation = explain_model(
       pipeline=result.pipeline,
       atlas_path='path/to/atlas.nii.gz',
       create_weight_maps=True,
       output_dir='./explanations'
   )

   print(explanation.get_summary())

CSV Format
----------

The expected CSV format:

.. code-block:: text

   subject_id,simulation_name,response,condition
   sub-001,montage_A,1,active
   sub-002,montage_A,0,active
   sub-003,sham,0,sham
   sub-004,montage_B,1,active

Required columns:

- ``subject_id``: Unique subject identifier
- ``simulation_name``: Stimulation configuration name
- ``{target_col}``: Target variable (0/1 for classification)

Optional columns:

- ``condition``: Experimental condition (e.g., 'active', 'sham')