from .inference import df_to_pydantic, df_to_patito
from .codegen import generate_class_code

__all__ = [
    "df_to_pydantic",
    "df_to_patito",
    "generate_class_code",
]

__version__ = "0.5.0"
