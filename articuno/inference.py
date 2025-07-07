"""
Unified inference utilities for Articuno.

This module provides high-level functions to infer Pydantic models from either
pandas or polars DataFrames, with optional support for nested columns and force_optional.
"""

from typing import Any, List, Optional, Type, Union
from pydantic import BaseModel

try:
    import pandas as pd
    _has_pandas = True
except ImportError:
    _has_pandas = False

try:
    import polars as pl
    _has_polars = True
except ImportError:
    _has_polars = False

from articuno.pandas_infer import (
    _is_pandas_df,
    infer_pydantic_model as infer_pandas_model,
)
from articuno.polars_infer import (
    _is_polars_df,
    infer_pydantic_model as infer_polars_model,
)


def infer_pydantic_model(
    df: Union["pd.DataFrame", "pl.DataFrame"],
    model_name: str = "AutoModel",
    force_optional: bool = False,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from either a pandas or polars DataFrame.

    Args:
        df: Input DataFrame (pandas or polars).
        model_name: Optional name for the generated model class.
        force_optional: If True, force all fields (including nested) to be Optional.

    Returns:
        A dynamically created Pydantic model class.
    """
    if _has_pandas and _is_pandas_df(df):
        return infer_pandas_model(df, model_name=model_name, force_optional=force_optional)
    elif _has_polars and _is_polars_df(df):
        return infer_polars_model(df, model_name=model_name, force_optional=force_optional)
    else:
        raise TypeError("Expected a pandas or polars DataFrame.")


def df_to_pydantic(
    df: Union["pd.DataFrame", "pl.DataFrame"],
    model: Optional[Type[BaseModel]] = None,
    model_name: Optional[str] = None,
    force_optional: bool = False,
) -> List[BaseModel]:
    """
    Convert a DataFrame into a list of Pydantic model instances.

    Args:
        df: Input DataFrame (pandas or polars).
        model: Optional pre-defined Pydantic model class.
        model_name: Name for auto-inferred model if no model is provided.
        force_optional: If True, force all fields in the inferred model to be Optional.

    Returns:
        List of instantiated Pydantic models based on DataFrame rows.
    """
    if model is None:
        model = infer_pydantic_model(df, model_name or "AutoModel", force_optional=force_optional)

    dicts = df.to_dict(orient="records") if _has_pandas and _is_pandas_df(df) else df.to_dicts()
    return [model(**row) for row in dicts]
