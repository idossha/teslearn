.. TESLearn documentation master file

TESLearn Documentation
======================

**TESLearn** is a modular, extensible Python library for machine learning-based prediction of responders to transcranial electrical stimulation (TES).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   user_guide
   api/index
   examples
   contributing

Features
--------

* **Modular Architecture**: Built on abstract base classes for easy extension
* **Multiple Feature Extraction Methods**: Atlas-based, voxel-based, whole-brain
* **Feature Selection**: T-test, F-regression, atlas-based ROI selection
* **Machine Learning Models**: Logistic Regression, SVM, Random Forest, Elastic Net
* **Cross-Validation**: Nested CV with hyperparameter tuning
* **Model Interpretation**: Feature importance, weight maps, visualizations
* **NIfTI Integration**: Built on nibabel for seamless neuroimaging workflows

Installation
------------

.. code-block:: bash

   pip install teslearn

For development:

.. code-block:: bash

   pip install "teslearn[dev]"

Quick Example
-------------

.. code-block:: python

   import teslearn as tl

   # Load data
   dataset = tl.load_dataset_from_csv('subjects.csv')
   
   # Train model (default: LR + T-test + nested CV)
   result = tl.train_model(
       images=images,
       y=targets,
       feature_extractor=tl.AtlasFeatureExtractor(atlas_path='atlas.nii.gz'),
       model=tl.LogisticRegressionModel(C=1.0),
       feature_selector=tl.TTestSelector(p_threshold=0.001)
   )
   
   # Predict
   proba = result.pipeline.predict_proba(test_images)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`