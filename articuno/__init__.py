from .inference import df_to_pydantic
from .codegen import generate_class_code

__all__ = [
    "df_to_pydantic",
    "generate_class_code",
]

__version__ = "0.6.0"
