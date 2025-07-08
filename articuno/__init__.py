# articuno/__init__.py

from .inference import df_to_pydantic, infer_pydantic_model
from .codegen import generate_class_code
from .iterable_infer import dicts_to_pydantic, infer_generic_model

__all__ = [
    "df_to_pydantic",
    "generate_class_code",
    "infer_pydantic_model",
    "dicts_to_pydantic",
    "infer_generic_model",
]

__version__ = "0.7.0"