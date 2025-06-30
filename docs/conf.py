import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = "articuno"
author = "Odos Matthews"
release = "0.3.7"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]


autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': True,
    'no-index': True,  # This tells Sphinx not to index members separately
}

typehints_fully_qualified = True