"""
Model inference utilities for converting Polars or Pandas DataFrames into Pydantic or Patito models.

This module provides high-level helpers to dynamically infer validation models
from `polars.DataFrame` or `pandas.DataFrame`, returning either `pydantic.BaseModel`
or `patito.Model` instances.
"""

from typing import List, Optional, Type, Union
from pydantic import BaseModel

# Optional dependencies
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

try:
    import patito as pt
    _has_patito = True
except ImportError:
    _has_patito = False

from articuno.polars_infer import (
    _is_polars_df,
    infer_pydantic_model as infer_polars_model,
    infer_patito_model as infer_polars_patito_model,
)

from articuno.pandas_infer import (
    _is_pandas_df,
    infer_pydantic_model_from_pandas,
    infer_patito_model_from_pandas,
)


def df_to_pydantic(
    df: Union["pl.DataFrame", "pd.DataFrame"],
    model: Optional[Type[BaseModel]] = None,
    model_name: Optional[str] = None,
) -> List[BaseModel]:
    """
    Convert a Polars or Pandas DataFrame into a list of Pydantic model instances.

    Parameters
    ----------
    df : Union[pl.DataFrame, pd.DataFrame]
        Input DataFrame to convert.
    model : Optional[Type[BaseModel]]
        A custom Pydantic model class to use. If None, a model will be inferred.
    model_name : Optional[str]
        Optional name to assign to the inferred model class.

    Returns
    -------
    List[BaseModel]
        List of instantiated Pydantic models based on the DataFrame rows.
    """
    if model is None:
        if _has_pandas and _is_pandas_df(df):
            model = infer_pydantic_model_from_pandas(df, model_name or "AutoPandasModel")
        elif _has_polars and _is_polars_df(df):
            model = infer_polars_model(df, model_name or "AutoPolarsModel")
        else:
            raise TypeError("Expected a pandas or polars DataFrame.")

    dicts = df.to_dict(orient="records") if _is_pandas_df(df) else df.to_dicts()
    return [model(**row) for row in dicts]


def df_to_patito(
    df: Union["pl.DataFrame", "pd.DataFrame"],
    model: Optional[Type["pt.Model"]] = None,
    model_name: Optional[str] = None,
) -> List["pt.Model"]:
    """
    Convert a Polars or Pandas DataFrame into a list of Patito model instances.

    Parameters
    ----------
    df : Union[pl.DataFrame, pd.DataFrame]
        Input DataFrame to convert.
    model : Optional[Type[pt.Model]]
        A custom Patito model class to use. If None, a model will be inferred.
    model_name : Optional[str]
        Optional name to assign to the inferred model class.

    Returns
    -------
    List[pt.Model]
        List of instantiated Patito models based on the DataFrame rows.

    Raises
    ------
    ImportError
        If Patito is not installed or if the required DataFrame backend is missing.
    """
    if not _has_patito:
        raise ImportError("Patito is not installed. Try `pip install articuno[patito]`.")

    if _has_pandas and _is_pandas_df(df):
        if model is None:
            model = infer_patito_model_from_pandas(df, model_name or "AutoPandasPatitoModel")
        dicts = df.to_dict(orient="records")
    elif _has_polars and _is_polars_df(df):
        if model is None:
            model = infer_polars_patito_model(df, model_name or "AutoPolarsPatitoModel")
        dicts = df.to_dicts()
    else:
        raise TypeError("Expected a pandas or polars DataFrame.")

    return [model(**row) for row in dicts]
