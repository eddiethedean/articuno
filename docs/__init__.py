import os
import sys

import articuno
sys.path.insert(0, os.path.abspath(".."))  # ensures your package is importable

project = "articuno"
author = "Odos Matthews"
release = articuno.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",         # for Google/Numpy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",    # for nice type hints
]

html_theme = "furo"  # or "sphinx_rtd_theme"
