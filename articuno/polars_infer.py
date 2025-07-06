"""
polars_infer.py

Provides functions to infer Pydantic or Patito models from Polars DataFrames.
Supports nested structs, nullable types, and optional constraint inference for Patito models.
"""

from typing import Any, Dict, Optional, Type, List, Union
import datetime

try:
    import polars as pl  # type: ignore
    import patito as pt  # type: ignore
    _has_polars = True
    _has_patito = True
except ImportError:
    _has_polars = False
    _has_patito = False

from pydantic import BaseModel, create_model


def _is_polars_df(df: Any) -> bool:
    return _has_polars and isinstance(df, pl.DataFrame)


def infer_pydantic_model(
    df: "pl.DataFrame",
    model_name: str = "AutoPolarsModel",
    _model_cache: Optional[Dict[str, Type[BaseModel]]] = None,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a Polars DataFrame.

    Args:
        df: A Polars DataFrame to infer the schema from.
        model_name: Name of the generated Pydantic model class.
        _model_cache: Internal use. Caches nested struct models to avoid duplication.

    Returns:
        A Pydantic model class that matches the structure of the input DataFrame.
    """
    if not _has_polars:
        raise ImportError("Polars is not installed. Try `pip install polars`.")

    if _model_cache is None:
        _model_cache = {}

    def wrap_optional(tp: Any, nullable: bool) -> Any:
        return Optional[tp] if nullable else tp

    def get_first_non_null(col: "pl.Series") -> Any:
        for val in col:
            if val is not None:
                return val
        return None

    def resolve_dtype(
        dtype: Any,
        column_data: Optional["pl.Series"] = None,
        prefix: str = "",
    ) -> Any:
        nullable = column_data is not None and column_data.is_null().any()
        default = get_first_non_null(column_data) if column_data is not None else None

        if dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
            return wrap_optional(int, nullable), default
        elif dtype in {pl.Float32, pl.Float64}:
            return wrap_optional(float, nullable), default
        elif dtype == pl.Boolean:
            return wrap_optional(bool, nullable), default
        elif dtype == pl.Utf8:
            return wrap_optional(str, nullable), default
        elif dtype == pl.Date:
            return wrap_optional(datetime.date, nullable), default
        elif dtype == pl.Datetime:
            return wrap_optional(datetime.datetime, nullable), default
        elif dtype == pl.Duration:
            return wrap_optional(datetime.timedelta, nullable), default
        elif dtype == pl.Null:
            return type(None), None
        elif dtype.__class__.__name__ == "Struct":
            struct_key = str(dtype)
            if struct_key in _model_cache:
                struct_model = _model_cache[struct_key]
            else:
                fields = {}
                for field in dtype.fields:
                    field_data = (
                        column_data.struct.field(field.name)
                        if column_data is not None else None
                    )
                    field_type, field_default = resolve_dtype(
                        field.dtype,
                        column_data=field_data,
                        prefix=f"{prefix}{field.name}."
                    )
                    is_nullable = field_data is not None and field_data.is_null().any()
                    fields[field.name] = (
                        wrap_optional(field_type, is_nullable),
                        field_default if is_nullable else ...
                    )
                struct_model = create_model(f"{model_name}_{len(_model_cache)}_Struct", **fields)
                _model_cache[struct_key] = struct_model
            return wrap_optional(struct_model, nullable), default
        elif dtype.__class__.__name__ == "List":
            inner_type, _ = resolve_dtype(dtype.inner)
            return wrap_optional(List[inner_type], nullable), default

        return wrap_optional(Any, nullable), default

    fields: Dict[str, tuple] = {}
    for name, dtype in df.schema.items():
        col = df.get_column(name)
        field_type, default = resolve_dtype(dtype, column_data=col)
        fields[name] = (field_type, default if default is not None else ...)

    return create_model(model_name, **fields)


def infer_patito_model(
    df: "pl.DataFrame",
    model_name: str = "AutoPatitoModel",
    infer_constraints: bool = False,
) -> Type["pt.Model"]:
    """
    Infer a Patito model class from a Polars DataFrame.

    Args:
        df: A Polars DataFrame to infer the schema from.
        model_name: Name of the generated Patito model class.
        infer_constraints: Whether to automatically generate bounds, lengths, and uniqueness constraints.

    Returns:
        A Patito model class that matches the DataFrame's structure.
    """
    if not _has_polars:
        raise ImportError("Polars is not installed. Try `pip install polars`.")
    if not _has_patito:
        raise ImportError("Patito is not installed. Try `pip install patito`.")

    fields = {}

    for name, dtype in df.schema.items():
        col = df.get_column(name)
        nullable = col.is_null().any()
        kwargs = {}
        non_nulls = col.drop_nulls()
        sample_value = non_nulls[0] if len(non_nulls) else None

        if dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
            typ = int
            if infer_constraints and len(non_nulls):
                kwargs["ge"] = int(non_nulls.min())
                kwargs["le"] = int(non_nulls.max())
        elif dtype in {pl.Float32, pl.Float64}:
            typ = float
            if infer_constraints and len(non_nulls):
                kwargs["ge"] = float(non_nulls.min())
                kwargs["le"] = float(non_nulls.max())
        elif dtype == pl.Boolean:
            typ = bool
        elif dtype == pl.Utf8:
            typ = str
            if infer_constraints and len(non_nulls):
                lengths = non_nulls.str.len_chars()
                kwargs["min_length"] = int(lengths.min())
                kwargs["max_length"] = int(lengths.max())
        elif dtype == pl.Date:
            typ = datetime.date
        elif dtype == pl.Datetime:
            typ = datetime.datetime
        elif dtype == pl.Duration:
            typ = datetime.timedelta
        elif dtype == pl.Null:
            typ = type(None)
        else:
            typ = Any

        if nullable:
            typ = Optional[typ]

        if infer_constraints and col.is_unique():
            kwargs["unique"] = True

        fields[name] = pt.Field(typ, **kwargs)

    return type(model_name, (pt.Model,), fields)
