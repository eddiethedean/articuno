from typing import Type

import polars as pl
from pydantic import BaseModel
from poldantic import to_pydantic_model  # type: ignore[import-untyped]


def infer_pydantic_model(
    df: pl.DataFrame,
    model_name: str = "AutoPolarsModel",
    force_optional: bool = False,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a Polars DataFrame using Poldantic.

    Args:
        df: Polars DataFrame to infer schema from.
        model_name: Desired name for the resulting Pydantic model class.
        force_optional: If True, wrap all fields (including nested ones) in Optional[].

    Returns:
        A dynamically created Pydantic model class.
        
    Raises:
        ValueError: If `model_name` is not a valid Python identifier.
    """
    if not model_name.isidentifier():
        raise ValueError(
            f"Invalid model name '{model_name}'. Model names must be valid Python identifiers "
            "(start with letter/underscore, contain only letters/digits/underscores)."
        )
    
    schema = df.schema
    return to_pydantic_model(
        schema,
        model_name=model_name,
        force_optional=force_optional
    )
