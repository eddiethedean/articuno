"""
Pandas DataFrame model inference utilities for converting pandas DataFrames into Pydantic models,
with explicit support for PyArrow extension dtypes when available.
"""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, create_model
import datetime

try:
    import pandas as pd  # type: ignore
    _has_pandas = True
except ImportError:
    _has_pandas = False

try:
    import pyarrow as pa  # type: ignore
    _has_pyarrow = True
except ImportError:
    _has_pyarrow = False


def _is_pandas_df(df: Any) -> bool:
    return _has_pandas and isinstance(df, pd.DataFrame)


def _infer_dict_model(samples: List[dict], field_name: str, force_optional: bool) -> Any:
    """
    Merge keys from multiple sample dicts to create a nested Pydantic model,
    optionally forcing all fields to be optional.

    Args:
        samples: List of dict samples from the DataFrame column.
        field_name: Name of the parent column, used to name the nested model.
        force_optional: If True, force all nested fields to be Optional.

    Returns:
        A dynamically created nested Pydantic model.
    """
    merged_keys = set()
    for sample in samples:
        merged_keys.update(sample.keys())

    fields = {}
    for key in merged_keys:
        always_present = all(key in sample for sample in samples)
        has_null = any(key in sample and sample[key] is None for sample in samples)

        value_sample = next((sample[key] for sample in samples if key in sample and sample[key] is not None), None)

        if isinstance(value_sample, int):
            typ = int
        elif isinstance(value_sample, float):
            typ = float
        elif isinstance(value_sample, str):
            typ = str
        elif isinstance(value_sample, bool):
            typ = bool
        elif isinstance(value_sample, dict):
            typ = dict
        elif isinstance(value_sample, list):
            typ = List[Any]
        else:
            typ = Any

        if force_optional or not always_present or has_null:
            typ = Optional[typ]
            default = None
        else:
            default = ...

        fields[key] = (typ, default)

    return create_model(f"{field_name}_NestedModel", **fields)


def _infer_type_from_series(series: pd.Series, col_name: str, force_optional: bool, sample_size: int = 100) -> Any:
    """
    Infer Python/Pydantic type for a pandas Series, with PyArrow dtype support.

    Args:
        series: pandas Series to analyze.
        col_name: Column name (used for nested model naming).
        force_optional: If True, force all columns to be Optional.
        sample_size: Number of samples to inspect for object columns.

    Returns:
        Tuple of inferred type and default value.
    """
    nullable = series.isnull().any()
    non_nulls = series.dropna()
    samples = non_nulls.head(sample_size).tolist()
    sample_value = samples[0] if samples else None

    if _has_pyarrow and hasattr(series.dtype, "arrow_dtype"):
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
        if all(isinstance(x, dict) for x in samples if x is not None):
            typ = _infer_dict_model(samples, col_name, force_optional=force_optional)
        elif isinstance(sample_value, str):
            typ = str
        elif isinstance(sample_value, list):
            typ = List[Any]
        elif sample_value is not None:
            typ = type(sample_value)
        else:
            typ = Any
    else:
        typ = Any

    if force_optional or nullable:
        typ = Optional[typ]
        default = None
    else:
        default = ...

    return typ, default


def infer_pydantic_model(
    df: "pd.DataFrame",
    model_name: str = "AutoPandasModel",
    force_optional: bool = False,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a pandas DataFrame schema, supporting PyArrow dtypes.

    Args:
        df: pandas DataFrame to infer the model from.
        model_name: Optional model class name.
        force_optional: If True, force all fields in the model to be Optional.

    Returns:
        A dynamically created Pydantic model class.
    """
    if not _has_pandas:
        raise ImportError("Pandas is not installed. Try `pip install pandas`.")

    fields: Dict[str, tuple] = {}

    for col_name in df.columns:
        series = df[col_name]
        field_type, default = _infer_type_from_series(series, col_name, force_optional=force_optional, sample_size=100)
        fields[col_name] = (field_type, default)

    return create_model(model_name, **fields)
