"""
Pandas DataFrame model inference utilities for converting pandas DataFrames into Pydantic models.
"""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, create_model
import datetime

try:
    import pandas as pd  # type: ignore
    _has_pandas = True
except ImportError:
    _has_pandas = False


def _is_pandas_df(df: Any) -> bool:
    return _has_pandas and isinstance(df, pd.DataFrame)


def infer_pydantic_model(
    df: "pd.DataFrame",
    model_name: str = "AutoPandasModel",
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a pandas DataFrame schema.

    This function infers field types based on the DataFrame's dtypes and sample
    non-null values. Nested dict columns are typed as `dict` without further inference.

    Args:
        df: pandas DataFrame to infer the model from.
        model_name: Optional model class name.

    Returns:
        A dynamically created Pydantic model class.
    """
    if not _has_pandas:
        raise ImportError("Pandas is not installed. Try `pip install pandas`.")

    fields: Dict[str, tuple] = {}

    for name, dtype in df.dtypes.items():
        nullable = df[name].isnull().any()
        non_nulls = df[name].dropna()
        sample_value = non_nulls.iloc[0] if not non_nulls.empty else None

        if pd.api.types.is_integer_dtype(dtype):
            typ = int
        elif pd.api.types.is_float_dtype(dtype):
            typ = float
        elif pd.api.types.is_bool_dtype(dtype):
            typ = bool
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            typ = datetime.datetime
        elif pd.api.types.is_object_dtype(dtype):
            # Treat nested dicts and other objects as plain dict or type of sample
            if isinstance(sample_value, dict):
                typ = dict
            elif isinstance(sample_value, list):
                typ = List[Any]
            elif isinstance(sample_value, str):
                typ = str
            else:
                typ = type(sample_value) if sample_value is not None else Any
        else:
            typ = Any

        if nullable:
            typ = Optional[typ]

        fields[name] = (typ, sample_value if sample_value is not None else ...)

    return create_model(model_name, **fields)


def df_to_pydantic(
    df: "pd.DataFrame",
    model: Optional[Type[BaseModel]] = None,
    model_name: Optional[str] = None,
) -> List[BaseModel]:
    """
    Convert a pandas DataFrame to a list of Pydantic model instances.

    Args:
        df: pandas DataFrame to convert.
        model: Optional Pydantic model class to use.
        model_name: Optional name to generate the model if no model is passed.

    Returns:
        List of Pydantic model instances, one per row.
    """
    if model is None:
        model = infer_pydantic_model(df, model_name or "AutoPandasModel")

    dicts = df.to_dict(orient="records")
    return [model(**row) for row in dicts]
