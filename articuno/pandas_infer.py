"""
pandas_infer.py

Provides functions to infer Pydantic or Patito models from Pandas DataFrames.
Supports nullable fields and basic nested dict-to-struct conversion for Pydantic.
"""

from typing import Any, Dict, Optional, Type, List, Union
import datetime

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

from pydantic import BaseModel, create_model


def _is_pandas_df(df: Any) -> bool:
    return _has_pandas and isinstance(df, pd.DataFrame)


def infer_pydantic_model_from_pandas(
    df: "pd.DataFrame",
    model_name: str = "AutoPandasModel",
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a pandas DataFrame.

    This supports nullable fields and nested dictionaries as submodels.

    Args:
        df: A pandas DataFrame to infer the schema from.
        model_name: Name of the generated Pydantic model class.

    Returns:
        A Pydantic model class that matches the structure of the input DataFrame.
    """
    if not _has_pandas:
        raise ImportError("Pandas is not installed. Try `pip install pandas`.")

    fields: Dict[str, tuple] = {}

    for name, dtype in df.dtypes.items():
        nullable = df[name].isnull().any()
        non_nulls = df[name].dropna()
        sample_value = non_nulls.iloc[0] if not non_nulls.empty else None

        # Determine base type
        if pd.api.types.is_integer_dtype(dtype):
            typ = int
        elif pd.api.types.is_float_dtype(dtype):
            typ = float
        elif pd.api.types.is_bool_dtype(dtype):
            typ = bool
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            typ = datetime.datetime
        elif pd.api.types.is_object_dtype(dtype) and isinstance(sample_value, dict):
            nested_model = infer_pydantic_model_from_dict(sample_value, name.capitalize())
            typ = nested_model
        elif pd.api.types.is_object_dtype(dtype) and isinstance(sample_value, str):
            typ = str
        elif pd.api.types.is_object_dtype(dtype) and isinstance(sample_value, list):
            inner_type = type(sample_value[0]) if sample_value else Any
            typ = List[inner_type]
        elif pd.api.types.is_object_dtype(dtype):
            typ = type(sample_value) if sample_value is not None else Any
        else:
            typ = Any

        if nullable:
            typ = Optional[typ]

        fields[name] = (typ, sample_value if sample_value is not None else ...)

    return create_model(model_name, **fields)


def infer_pydantic_model_from_dict(
    sample: dict,
    model_name: str = "NestedModel"
) -> Type[BaseModel]:
    """
    Recursively build a nested Pydantic model from a dictionary.

    Args:
        sample: A sample dictionary representing a nested structure.
        model_name: Name to give the created submodel.

    Returns:
        A dynamically created Pydantic submodel class.
    """
    sub_fields: Dict[str, tuple] = {}
    for key, value in sample.items():
        if isinstance(value, dict):
            sub_model = infer_pydantic_model_from_dict(value, model_name=f"{model_name}_{key.capitalize()}")
            sub_fields[key] = (sub_model, ...)
        elif isinstance(value, list):
            inner_type = type(value[0]) if value else Any
            sub_fields[key] = (List[inner_type], ...)
        else:
            sub_fields[key] = (type(value), ...)
    return create_model(model_name, **sub_fields)


def infer_patito_model_from_pandas(
    df: "pd.DataFrame",
    model_name: str = "AutoPatitoModel",
    infer_constraints: bool = False,
) -> Type["pt.Model"]:
    """
    Infer a Patito model class from a pandas DataFrame.

    Supports basic constraints like bounds, length, and uniqueness if enabled.

    Args:
        df: A pandas DataFrame to infer the schema from.
        model_name: Name of the generated Patito model class.
        infer_constraints: Whether to add min/max/unique constraints.

    Returns:
        A Patito model class matching the structure of the DataFrame.
    """
    if not _has_pandas:
        raise ImportError("Pandas is not installed. Try `pip install pandas`.")
    if not _has_patito:
        raise ImportError("Patito is not installed. Try `pip install patito`.")

    fields = {}

    for name, dtype in df.dtypes.items():
        nullable = df[name].isnull().any()
        non_nulls = df[name].dropna()
        sample_value = non_nulls.iloc[0] if not non_nulls.empty else None
        kwargs = {}

        if pd.api.types.is_integer_dtype(dtype):
            typ = int
            if infer_constraints and not non_nulls.empty:
                kwargs["ge"] = int(non_nulls.min())
                kwargs["le"] = int(non_nulls.max())
        elif pd.api.types.is_float_dtype(dtype):
            typ = float
            if infer_constraints and not non_nulls.empty:
                kwargs["ge"] = float(non_nulls.min())
                kwargs["le"] = float(non_nulls.max())
        elif pd.api.types.is_bool_dtype(dtype):
            typ = bool
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            typ = datetime.datetime
        elif pd.api.types.is_object_dtype(dtype) and isinstance(sample_value, str):
            typ = str
            if infer_constraints and not non_nulls.empty:
                lengths = non_nulls.str.len()
                kwargs["min_length"] = int(lengths.min())
                kwargs["max_length"] = int(lengths.max())
        else:
            typ = type(sample_value) if sample_value is not None else Any

        if nullable:
            typ = Optional[typ]

        if infer_constraints and df[name].is_unique:
            kwargs["unique"] = True

        fields[name] = pt.Field(typ, **kwargs)

    return type(model_name, (pt.Model,), fields)
