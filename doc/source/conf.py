# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "funROI"
copyright = "2025, funROI developers"
author = "Ruimin Gao and Anna A. Ivanova"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # 'sphinx_gallery.gen_gallery',
    'myst_nb',
    'sphinx_design'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

nb_execution_mode = 'off'

templates_path = ["_templates"]
exclude_patterns = ["build", "funROI/tests"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# # Gallery configuration
# sphinx_gallery_conf = {
#     'examples_dirs': '../../examples',   # path to your example scripts
#     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
#     'copyfile_regex': r'.*\.png'
# }
