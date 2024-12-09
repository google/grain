# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path

sys.path.insert(0, str(Path('..', 'grain').resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyGrain'
copyright = '2024, grain team'
author = 'PyGrain team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx_copybutton',
    'sphinx_design',
    # 'autodoc2',
    # 'sphinx.ext.autodoc',
    'autoapi.extension',
]
myst_enable_extensions = ["colon_fence"]

templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb', '.md']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = 'PyGrain'
html_static_path = ['_static']
# TODO: Add logo and favicon
# html_logo = '_static/'
# html_favicon = '_static/favicon.png'

# Theme-specific options
# https://sphinx-book-theme.readthedocs.io/en/stable/reference.html
html_theme_options = {
    'show_navbar_depth': 2,
    'show_toc_level': 3,
    'repository_url': 'https://github.com/google/grain',
    'use_issues_button': True,
    'use_repository_button': True,
    'path_to_docs': 'docs/',
    'navigation_with_keys': True,
}

# Autodoc settings
# Should be relative to the source of the documentation
autoapi_dirs = [
    '../grain/_src/core',              
                ]