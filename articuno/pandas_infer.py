"""
Pandas DataFrame model inference utilities for Articuno.

Provides functions to convert pandas DataFrames into Pydantic models,
with explicit support for PyArrow extension dtypes when available,
and nested dict columns using `dict_model._infer_dict_model`.
"""

from typing import Any, Dict, List, Tuple, Type
from pydantic import BaseModel, create_model
import datetime
import pandas as pd

# Nested dict model inference logic
from articuno.dict_model import _infer_dict_model


def is_pyarrow_available() -> bool:
    """
    Check if PyArrow is installed and importable.

    Returns
    -------
    bool
        True if PyArrow can be imported, False otherwise.
    """
    try:
        import pyarrow  # type: ignore
        return True
    except ImportError:
        return False


def _infer_type_from_series(
    series: pd.Series,
    col_name: str,
    force_optional: bool,
    sample_size: int = 100,
) -> Tuple[Any, Any]:
    """
    Infer Python and Pydantic type for a pandas Series, supporting PyArrow dtypes.

    Parameters
    ----------
    series : pd.Series
        Series to analyze.
    col_name : str
        Column name (used for nested model naming).
    force_optional : bool
        If True, all fields become Optional.
    sample_size : int
        Number of samples for object dtype inference.

    Returns
    -------
    typ : Any
        Inferred Python type for the field.
    default : Any
        Default value (None or ...) for the Pydantic field.
    """
    # Determine nullability
    nullable = series.isnull().any()
    non_null = series.dropna()
    samples = non_null.head(sample_size).tolist()
    sample_val = samples[0] if samples else None

    # PyArrow-backed checks
    if is_pyarrow_available() and hasattr(series.dtype, "arrow_dtype"):
        import pyarrow as pa  # type: ignore
        arrow_dtype = series.dtype.arrow_dtype
        if pa.types.is_integer(arrow_dtype):
            typ = int
        elif pa.types.is_floating(arrow_dtype):
            typ = float
        elif pa.types.is_string(arrow_dtype):
            typ = str
        elif pa.types.is_boolean(arrow_dtype):
            typ = bool
        else:
            typ = Any
    elif pd.api.types.is_integer_dtype(series.dtype):
        typ = int
    elif pd.api.types.is_float_dtype(series.dtype):
        typ = float
    elif pd.api.types.is_bool_dtype(series.dtype):
        typ = bool
    elif pd.api.types.is_datetime64_any_dtype(series.dtype):
        typ = datetime.datetime
    elif pd.api.types.is_object_dtype(series.dtype):
        # Nested dict
        if all(isinstance(x, dict) for x in samples if x is not None):
            typ = _infer_dict_model(samples, col_name, force_optional=force_optional)
        # List
        elif isinstance(sample_val, list):
            typ = List[Any]
        # String
        elif isinstance(sample_val, str):
            typ = str
        else:
            typ = Any
    else:
        typ = Any

    # Assign default and optional
    if force_optional or nullable:
        from typing import Optional as _Opt
        typ = _Opt[typ]
        default = None
    else:
        default = ...

    return typ, default


def infer_pydantic_model(
    df: pd.DataFrame,
    model_name: str = "AutoPandasModel",
    force_optional: bool = False,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a pandas DataFrame schema, supporting PyArrow dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to infer model from.
    model_name : str, optional
        Desired model class name.
    force_optional : bool, optional
        If True, force all fields Optional.

    Returns
    -------
    Type[BaseModel]
        Dynamically created Pydantic model class.
    """
    fields: Dict[str, tuple] = {}
    for col in df.columns:
        series = df[col]
        typ, default = _infer_type_from_series(
            series, col, force_optional=force_optional, sample_size=100
        )
        fields[col] = (typ, default)

    return create_model(model_name, **fields)
