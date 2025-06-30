from .convert import df_to_pydantic, infer_pydantic_model
from .codegen import generate_pydantic_class_code
from .cli import app as cli_app
from .bootstrap import get_inference_registry


__all__ = [
    "df_to_pydantic",
    "infer_pydantic_model",
    "generate_pydantic_class_code",
    "cli_app",
    "get_inference_registry"
]
__version__ = "0.3.1"
