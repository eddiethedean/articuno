"""
❄️ Articuno ❄️

A blazing-fast tool to convert Polars DataFrames into Pydantic or Patito models,
with support for schema inference, validation, and clean code generation.
"""

from .models import (
    df_to_pydantic,
    infer_pydantic_model,
    df_to_patito,
    infer_patito_model,
)

from .codegen import (
    generate_pydantic_class_code,
    generate_patito_class_code,
)

from .bootstrap import infer_response_model
from .cli import app as cli_app

__all__ = [
    # Core conversion functions
    "df_to_pydantic",
    "infer_pydantic_model",
    "generate_pydantic_class_code",
    "df_to_patito",
    "infer_patito_model",
    "generate_patito_class_code",

    # FastAPI integration
    "infer_response_model",

    # CLI application (Typer app)
    "cli_app",
]



__version__ = "0.4.1"