"""
Main inference interface for converting Polars or Pandas DataFrames into Pydantic models.

This module provides the primary `df_to_pydantic` function which infers Pydantic models
from the given DataFrame and instantiates model instances for each row.
"""

from typing import List, Optional, Type, Union
from pydantic import BaseModel

try:
    import polars as pl
    _has_polars = True
except ImportError:
    _has_polars = False

try:
    import pandas as pd
    _has_pandas = True
except ImportError:
    _has_pandas = False

from .polars_infer import _is_polars_df, infer_pydantic_model as infer_polars_model
from .pandas_infer import _is_pandas_df, infer_pydantic_model as infer_pandas_model


def df_to_pydantic(
    df: Union["pl.DataFrame", "pd.DataFrame"],
    model: Optional[Type[BaseModel]] = None,
    model_name: Optional[str] = None,
) -> List[BaseModel]:
    """
    Infer a Pydantic model class from a Polars or Pandas DataFrame and return instances for each row.

    Args:
        df: A Polars or Pandas DataFrame to infer the schema from.
        model: Optional. A pre-defined Pydantic model class to use for instantiation.
        model_name: Optional. Name of the model class to generate if `model` is None.

    Returns:
        List of Pydantic model instances corresponding to DataFrame rows.

    Raises:
        TypeError: If the input is not a Polars or Pandas DataFrame.
    """
    if model is None:
        if _has_pandas and _is_pandas_df(df):
            model = infer_pandas_model(df, model_name or "AutoPandasModel")
        elif _has_polars and _is_polars_df(df):
            model = infer_polars_model(df, model_name or "AutoPolarsModel")
        else:
            raise TypeError("Expected a pandas or polars DataFrame.")

    dicts = df.to_dict(orient="records") if _has_pandas and _is_pandas_df(df) else df.to_dicts()
    return [model(**row) for row in dicts]
