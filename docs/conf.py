"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import pathlib
import sys

autodoc_mock_imports = [
    'grain.proto.execution_summary_pb2',
    'grain._src.python.experimental.index_shuffle.python.index_shuffle_module',
    'cloudpickle',
    'numpy',
    'orbax',
    'tree',
    'absl',
    'absl.logging',
    'array_record',
]

sys.path.insert(0, str(pathlib.Path('..', 'grain').resolve()))
sys.path.insert(0, os.path.abspath('..'))


# import grain  # pylint: disable=unused-import, g-import-not-at-top, g-bad-import-order

# raise ValueError(f'cwd: {os.getcwd()}\n\n path: {sys.path}')

# pylint: disable=unreachable
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Grain'
copyright = '2024, Grain team'  # pylint: disable=redefined-builtin
author = 'Grain team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb', '.md']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'tutorials/dataset_basic_tutorial.md',
]

# Suppress warning in exception basic_data_tutorial
suppress_warnings = [
    'misc.highlighting_failure',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = 'Grain'
html_static_path = ['_static']

# TODO: Add logo and favicon
# html_logo = '_static/'
# html_favicon = '_static/favicon.png'

# Theme-specific options
# https://sphinx-book-theme.readthedocs.io/en/stable/reference.html
html_theme_options = {
    'show_navbar_depth': 1,
    'show_toc_level': 3,
    'repository_url': 'https://github.com/google/grain',
    'use_issues_button': True,
    'use_repository_button': True,
    'path_to_docs': 'docs/',
    'navigation_with_keys': True,
}

# Autodoc settings
# Should be relative to the source of the documentation
autosummary_generate = True
autodoc_typehints = 'description'

# -- Myst configurations -------------------------------------------------
myst_enable_extensions = ['colon_fence']
nb_execution_mode = 'force'
nb_execution_allow_errors = False
nb_merge_streams = True
nb_execution_show_tb = True

# Notebook cell execution timeout; defaults to 30.
nb_execution_timeout = 100

# List of patterns, relative to source directory, that match notebook
# files that will not be executed.
nb_execution_excludepatterns = [
    'tutorials/dataset_advanced_tutorial.ipynb',
    'tutorials/dataset_basic_tutorial.ipynb',
    'tutorials/data_loader_tutorial.ipynb',
    'tutorials/dataset_debugging_tutorial.ipynb',
]
