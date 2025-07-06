"""
Pandas DataFrame model inference utilities for converting pandas DataFrames into Pydantic models,
with deep nested dict and list inference.
"""

from typing import Any, Dict, List, Optional, Type, Tuple
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
    Infer a Pydantic model class from a pandas DataFrame schema,
    supporting deep nested dict and list inference.

    Args:
        df: pandas DataFrame to infer the model from.
        model_name: Optional model class name.

    Returns:
        A dynamically created Pydantic model class.
    """
    if not _has_pandas:
        raise ImportError("Pandas is not installed. Try `pip install pandas`.")

    fields: Dict[str, Tuple[Any, Any]] = {}

    def _infer_value(val: Any, path: str) -> Tuple[Any, Any]:
        # Recursive inference for nested dicts and lists
        if isinstance(val, dict):
            nested_fields: Dict[str, Tuple[Any, Any]] = {}
            for k, v in val.items():
                nested_type, nested_default = _infer_value(v, f"{path}_{k}")
                nested_fields[k] = (
                    nested_type,
                    nested_default if nested_default is not None else ...
                )
            nested_model = create_model(f"{model_name}_{path}_Struct", **nested_fields)
            return nested_model, val

        if isinstance(val, list):
            # Infer element type from first non-null element
            non_null = [e for e in val if e is not None]
            if non_null:
                elem_type, _ = _infer_value(non_null[0], f"{path}_item")
            else:
                elem_type = Any
            return List[elem_type], val

        # Primitive types
        if isinstance(val, bool):
            return bool, val
        if isinstance(val, int):
            return int, val
        if isinstance(val, float):
            return float, val
        if isinstance(val, str):
            return str, val
        if isinstance(val, datetime.datetime):
            return datetime.datetime, val
        if isinstance(val, datetime.date):
            return datetime.date, val

        # Fallback
        return Any, val

    for col in df.columns:
        dtype = df[col].dtype
        nullable = df[col].isnull().any()
        non_nulls = df[col].dropna()
        sample_value = non_nulls.iloc[0] if not non_nulls.empty else None

        # Base dtype inference
        if pd.api.types.is_integer_dtype(dtype):
            typ, default = int, sample_value
        elif pd.api.types.is_float_dtype(dtype):
            typ, default = float, sample_value
        elif pd.api.types.is_bool_dtype(dtype):
            typ, default = bool, sample_value
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            typ, default = datetime.datetime, sample_value
        elif pd.api.types.is_object_dtype(dtype) and sample_value is not None:
            # Deep nested dict/list inference
            typ, default = _infer_value(sample_value, col)
        else:
            typ, default = Any, sample_value

        # Apply Optional if column has nulls
        if nullable:
            from typing import Optional as _Opt
            typ = _Opt[typ]

        fields[col] = (typ, default if default is not None else ...)

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
