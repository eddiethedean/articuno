"""
inference.py

Public APIs for inferring Pydantic or Patito models from Polars or Pandas DataFrames.
Handles optional backend dependencies and dispatches to format-specific logic.
"""

from typing import Optional, Type, List, Union, Any
from pydantic import BaseModel

# Dependency flags
try:
    import polars as pl  # type: ignore
    _has_polars = True
except ImportError:
    _has_polars = False

try:
    import pandas as pd  # type: ignore
    _has_pandas = True
except ImportError:
    _has_pandas = False

try:
    import patito as pt  # type: ignore
    _has_patito = True
except ImportError:
    _has_patito = False

# Import backend logic
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
    Convert a DataFrame (Polars or Pandas) into a list of Pydantic models.

    Args:
        df: Input Polars or Pandas DataFrame.
        model: Optional pre-inferred Pydantic model to instantiate.
        model_name: Optional name to use if generating the model dynamically.

    Returns:
        List of instantiated Pydantic models based on the DataFrame rows.
    """
    if model is None:
        if _has_pandas and _is_pandas_df(df):
            model = infer_pydantic_model_from_pandas(df, model_name or "AutoPandasModel")
        elif _has_polars and _is_polars_df(df):
            model = infer_polars_model(df, model_name or "AutoPolarsModel")
        else:
            raise TypeError("Expected a pandas or polars DataFrame.")

    dicts = df.to_dict(orient="records") if _has_pandas and _is_pandas_df(df) else df.to_dicts()
    return [model(**row) for row in dicts]


def df_to_patito(
    df: Union["pl.DataFrame", "pd.DataFrame"],
    model: Optional[Type["pt.Model"]] = None,
    model_name: Optional[str] = None,
    infer_constraints: bool = False,
) -> List["pt.Model"]:
    """
    Convert a DataFrame (Polars or Pandas) into a list of Patito models.

    Args:
        df: Input Polars or Pandas DataFrame.
        model: Optional pre-inferred Patito model to instantiate.
        model_name: Optional name to use if generating the model dynamically.
        infer_constraints: Whether to add min/max/length/unique constraints in the model.

    Returns:
        List of instantiated Patito models.
    """
    if not _has_patito:
        raise ImportError("Patito is not installed. Try `pip install patito`.")

    if model is None:
        if _has_pandas and _is_pandas_df(df):
            model = infer_patito_model_from_pandas(df, model_name or "AutoPatitoModel", infer_constraints=infer_constraints)
        elif _has_polars and _is_polars_df(df):
            model = infer_polars_patito_model(df, model_name or "AutoPatitoModel", infer_constraints=infer_constraints)
        else:
            raise TypeError("Expected a pandas or polars DataFrame.")

    dicts = df.to_dict(orient="records") if _has_pandas and _is_pandas_df(df) else df.to_dicts()
    return [model(**row) for row in dicts]


def infer_pydantic_model(
    df: Union["pl.DataFrame", "pd.DataFrame"],
    model_name: str = "AutoModel"
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a Polars or Pandas DataFrame.

    Args:
        df: Input DataFrame.
        model_name: Name of the model to generate.

    Returns:
        A dynamically created Pydantic model class.
    """
    if _has_pandas and _is_pandas_df(df):
        return infer_pydantic_model_from_pandas(df, model_name=model_name)
    elif _has_polars and _is_polars_df(df):
        return infer_polars_model(df, model_name=model_name)
    else:
        raise TypeError("Expected a pandas or polars DataFrame.")


def infer_patito_model(
    df: Union["pl.DataFrame", "pd.DataFrame"],
    model_name: str = "AutoPatitoModel",
    infer_constraints: bool = False,
) -> Type["pt.Model"]:
    """
    Infer a Patito model class from a Polars or Pandas DataFrame.

    Args:
        df: Input DataFrame.
        model_name: Name of the model to generate.
        infer_constraints: Whether to add min/max/length/unique constraints in the model.

    Returns:
        A dynamically created Patito model class.
    """
    if not _has_patito:
        raise ImportError("Patito is not installed. Try `pip install patito`.")

    if _has_pandas and _is_pandas_df(df):
        return infer_patito_model_from_pandas(df, model_name=model_name, infer_constraints=infer_constraints)
    elif _has_polars and _is_polars_df(df):
        return infer_polars_patito_model(df, model_name=model_name, infer_constraints=infer_constraints)
    else:
        raise TypeError("Expected a pandas or polars DataFrame.")
