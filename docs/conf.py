import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = "articuno"
author = "Odos Matthews"
release = "0.3.7"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]