from .models import df_to_pydantic, df_to_patito, infer_pydantic_model, infer_patito_model
from .codegen import generate_class_code
from .cli import app as cli_app
from .bootstrap import bootstrap_app

__all__ = [
    "df_to_pydantic",
    "df_to_patito",
    "infer_pydantic_model",
    "infer_patito_model",
    "generate_class_code",
    "cli_app",
    "bootstrap_app",
]

__version__ = "0.4.5"