# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import teslearn

# -- Project information -----------------------------------------------------
project = "TESLearn"
copyright = "2025, TESLearn Contributors"
author = "TESLearn Contributors"
release = teslearn.__version__
version = teslearn.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"TESLearn {version}"
html_short_title = "TESLearn"

# Furo theme options for dark theme by default
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2b8cc4",
        "color-brand-content": "#2b8cc4",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5eb8e0",
        "color-brand-content": "#5eb8e0",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
}

# Default to dark mode
html_context = {
    "default_mode": "dark",
}

# Custom CSS
html_css_files = [
    "custom.css",
]

# -- Extension configuration -------------------------------------------------
# Mock imports for modules not needed during docs build
autodoc_mock_imports = [
    "matplotlib",
    "matplotlib.pyplot",
    "nilearn",
    "nilearn.plotting",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autosummary_generate = True
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
}

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Copybutton settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Autosummary settings
autosummary_imported_members = True
