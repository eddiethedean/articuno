"""
Pandas model inference utilities for generating Pydantic or Patito models.

This module provides functions to analyze pandas.DataFrames and construct equivalent
Pydantic or Patito models. Nested dictionaries are supported recursively as nested models.
"""

from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, create_model
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


def _is_pandas_df(df: Any) -> bool:
    return _has_pandas and isinstance(df, pd.DataFrame)


def infer_pydantic_model_from_pandas(
    df: "pd.DataFrame",
    model_name: str = "AutoPandasModel",
    _model_cache: Optional[Dict[str, Type[BaseModel]]] = None,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a pandas DataFrame.

    Recursively detects nested dictionaries and lists of dictionaries,
    and generates nested model classes.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to inspect.
    model_name : str
        The name of the root model class.
    _model_cache : dict, optional
        Internal cache to avoid regenerating identical nested models.

    Returns
    -------
    Type[BaseModel]
        A dynamically generated Pydantic model class.
    """
    if not _has_pandas:
        raise ImportError("Pandas is not installed. Try `pip install pandas`.")

    if _model_cache is None:
        _model_cache = {}

    def wrap_optional(tp: Any, nullable: bool) -> Any:
        return Optional[tp] if nullable else tp

    def resolve_dtype(
        series: pd.Series,
        prefix: str = ""
    ) -> Any:
        nullable = series.isnull().any()
        non_nulls = series.dropna()
        sample = non_nulls.iloc[0] if not non_nulls.empty else None

        # Handle nested dict
        if isinstance(sample, dict):
            struct_key = f"{prefix}{series.name}"
            if struct_key in _model_cache:
                struct_model = _model_cache[struct_key]
            else:
                nested_fields = {}
                for k, v in sample.items():
                    field_type, default = resolve_dtype(
                        pd.Series([d.get(k) for d in series.dropna() if isinstance(d, dict)]),
                        prefix=f"{prefix}{k}."
                    )
                    nested_fields[k] = (field_type, default if default is not None else ...)
                struct_model = create_model(f"{model_name}_{len(_model_cache)}_Nested", **nested_fields)
                _model_cache[struct_key] = struct_model
            return wrap_optional(struct_model, nullable), sample

        # Handle list of dicts
        if isinstance(sample, list) and sample and isinstance(sample[0], dict):
            # assume homogeneous list of dicts
            nested_series = pd.Series([sample[0] for sample in non_nulls if sample])
            nested_type, _ = resolve_dtype(nested_series, prefix=f"{prefix}item.")
            return wrap_optional(List[nested_type], nullable), sample

        # Primitive type mapping
        if pd.api.types.is_integer_dtype(series):
            typ = int
        elif pd.api.types.is_float_dtype(series):
            typ = float
        elif pd.api.types.is_bool_dtype(series):
            typ = bool
        elif pd.api.types.is_datetime64_any_dtype(series):
            typ = datetime.datetime
        elif isinstance(sample, str):
            typ = str
        elif isinstance(sample, list):
            inner_type = type(sample[0]) if sample else Any
            typ = List[inner_type]
        elif sample is not None:
            typ = type(sample)
        else:
            typ = Any

        return wrap_optional(typ, nullable), sample

    fields: Dict[str, tuple] = {}

    for name in df.columns:
        series = df[name]
        field_type, default = resolve_dtype(series)
        fields[name] = (field_type, default if default is not None else ...)

    return create_model(model_name, **fields)


def infer_patito_model_from_pandas(
    df: "pd.DataFrame",
    model_name: str = "AutoPatitoModel",
) -> Type["pt.Model"]:
    """
    Infer a Patito model class from a pandas DataFrame.

    Nested dictionaries are not supported in Patito inference.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to inspect.
    model_name : str
        Name of the generated Patito model.

    Returns
    -------
    Type[pt.Model]
        A dynamically generated Patito model class.
    """
    if not _has_pandas:
        raise ImportError("Pandas is not installed. Try `pip install pandas`.")
    if not _has_patito:
        raise ImportError("Patito is not installed. Try `pip install patito`.")

    fields = {}

    for name, series in df.items():
        nullable = series.isnull().any()
        sample = series.dropna().iloc[0] if not series.dropna().empty else None

        if pd.api.types.is_integer_dtype(series):
            typ = int
        elif pd.api.types.is_float_dtype(series):
            typ = float
        elif pd.api.types.is_bool_dtype(series):
            typ = bool
        elif pd.api.types.is_datetime64_any_dtype(series):
            typ = datetime.datetime
        elif isinstance(sample, str):
            typ = str
        elif isinstance(sample, list):
            inner = type(sample[0]) if sample else Any
            typ = List[inner]
        elif sample is not None:
            typ = type(sample)
        else:
            typ = Any

        if nullable:
            typ = Optional[typ]

        fields[name] = pt.Field(typ)

    return type(model_name, (pt.Model,), fields)
