# convert.py can be empty or provide high-level wrappers
# that just call models.py functions

from .models import df_to_pydantic, df_to_patito, infer_pydantic_model, infer_patito_model

# Optional: re-export for convenience
__all__ = [
    "df_to_pydantic",
    "df_to_patito",
    "infer_pydantic_model",
    "infer_patito_model",
]
