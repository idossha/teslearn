.. TESLearn documentation master file

TESLearn Documentation
======================

.. rst-class:: teslearn-hero

Machine Learning for Transcranial Electrical Stimulation Responder Prediction
-----------------------------------------------------------------------------

**TESLearn** is a modular, extensible Python library for predicting responders to 
transcranial electrical stimulation (TES) using electric field intensity maps in MNI space.
Built on scikit-learn and nibabel for seamless neuroimaging workflows.

.. rst-class:: badge-container

|Python| |License| |Docs| |GitHub|

.. |Python| image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. |License| image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/idossha/teslearn/blob/main/LICENSE
   :alt: MIT License

.. |Docs| image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: https://teslearn.readthedocs.io
   :alt: Documentation

.. |GitHub| image:: https://img.shields.io/badge/github-repo-black.svg
   :target: https://github.com/idossha/teslearn
   :alt: GitHub Repository

Quick Links
-----------

.. rst-class:: quick-links

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Installation
      :link: quickstart.html
      :link-type: url
      :class-card: quick-link-card

      Get started quickly with pip install and setup instructions for TESLearn.

   .. grid-item-card:: API Reference
      :link: api/index.html
      :link-type: url
      :class-card: quick-link-card

      Comprehensive documentation of all modules, classes, and functions.

   .. grid-item-card:: User Guide
      :link: user_guide.html
      :link-type: url
      :class-card: quick-link-card

      Detailed tutorials and guides for using TESLearn effectively.

   .. grid-item-card:: Examples
      :link: examples.html
      :link-type: url
      :class-card: quick-link-card

      Real-world examples and use cases with code snippets.

   .. grid-item-card:: GitHub
      :link: https://github.com/idossha/teslearn
      :link-type: url
      :class-card: quick-link-card

      Source code, issue tracker, and contribution guidelines.

   .. grid-item-card:: Contributing
      :link: contributing.html
      :link-type: url
      :class-card: quick-link-card

      Guidelines for contributing to TESLearn development.

Features
--------

.. rst-class:: feature-grid

Modular Architecture
~~~~~~~~~~~~~~~~~~~~
Built on abstract base classes for easy extension. Implement custom feature extractors, 
models, and validators without modifying core code.

Multiple Feature Extraction Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Atlas-based ROI features, voxel-based features, whole-brain flattening, and composite 
extractors for flexible analysis pipelines.

Machine Learning Models
~~~~~~~~~~~~~~~~~~~~~~~
Logistic Regression, SVM, Random Forest, Elastic Net with consistent interfaces and 
built-in cross-validation support.

Model Interpretation
~~~~~~~~~~~~~~~~~~~~
Feature importance analysis, weight maps, and visualization tools to understand 
what drives predictions.

NIfTI Integration
~~~~~~~~~~~~~~~~~
Built on nibabel for seamless neuroimaging workflows with automatic resampling and 
MNI space handling.

Installation
------------

Install TESLearn using pip:

.. code-block:: bash

   pip install teslearn

For development or all features:

.. code-block:: bash

   pip install "teslearn[all]"

Quick Example
-------------

.. code-block:: python

   import teslearn as tl

   # Load data
   dataset = tl.load_dataset_from_csv('subjects.csv')
   images, indices = tl.NiftiLoader().load_dataset_images(dataset)
   y = dataset.get_targets()

   # Train model with nested CV
   result = tl.train_model(
       images=images,
       y=y,
       feature_extractor=tl.AtlasFeatureExtractor(atlas_path='atlas.nii.gz'),
       model=tl.LogisticRegressionModel(C=1.0),
       feature_selector=tl.TTestSelector(p_threshold=0.001)
   )

   # Predict and explain
   proba = result.pipeline.predict_proba(test_images)
   explanation = tl.explain_model(result.pipeline, atlas_path='atlas.nii.gz')

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   user_guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   examples
   contributing

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Acknowledgments
---------------

TESLearn draws inspiration from `nilearn <https://nilearn.github.io/>`_ and is built on:

- `scikit-learn <https://scikit-learn.org/>`_ for machine learning
- `nibabel <https://nipy.org/nibabel/>`_ for neuroimaging I/O
- `scipy <https://scipy.org/>`_ for statistical functions

.. note::

   TESLearn is under active development. For bug reports and feature requests, 
   please visit our `GitHub Issues <https://github.com/idossha/teslearn/issues>`_.
