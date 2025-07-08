"""
Polars DataFrame model inference utilities for converting Polars DataFrames into Pydantic models.
"""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, create_model
import datetime
import polars as pl


def _infer_struct_model(dtype: Any, column_data: Optional[pl.Series], model_name: str, model_cache: Dict[str, Type[BaseModel]], force_optional: bool) -> Any:
    """
    Recursively infer a nested Pydantic model from a Polars Struct dtype.

    Args:
        dtype: Polars Struct dtype.
        column_data: Corresponding column data for null checks.
        model_name: Base name for the nested model.
        model_cache: Cache to avoid duplicate models.
        force_optional: If True, force all struct fields to be optional.

    Returns:
        Tuple of nested model type and default value.
    """
    struct_key = str(dtype)
    if struct_key in model_cache:
        struct_model = model_cache[struct_key]
    else:
        fields = {}
        for field in dtype.fields:
            field_data = column_data.struct.field(field.name) if column_data is not None else None
            field_type, default = _resolve_dtype(
                field.dtype,
                field_data,
                model_name=model_name,
                model_cache=model_cache,
                force_optional=force_optional
            )

            nullable = force_optional or (field_data is not None and field_data.is_null().any())
            if nullable:
                field_type = Optional[field_type]
                default = None
            else:
                default = ...

            fields[field.name] = (field_type, default)

        struct_model = create_model(f"{model_name}_{len(model_cache)}_Struct", **fields)
        model_cache[struct_key] = struct_model

    return struct_model, None


def _resolve_dtype(dtype: Any, column_data: Optional["pl.Series"], model_name: str, model_cache: Dict[str, Type[BaseModel]], force_optional: bool) -> Any:
    """
    Resolve a Polars dtype to a Pydantic type.

    Args:
        dtype: Polars dtype.
        column_data: Series data to check nullability.
        model_name: Base model name.
        model_cache: Nested model cache.
        force_optional: If True, force fields to be optional.

    Returns:
        Tuple of inferred type and default value.
    """
    nullable = force_optional or (column_data is not None and column_data.is_null().any())
    default = None if nullable else ...

    if dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                 pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
        typ = int
    elif dtype in {pl.Float32, pl.Float64}:
        typ = float
    elif dtype == pl.Boolean:
        typ = bool
    elif dtype == pl.Utf8:
        typ = str
    elif dtype == pl.Date:
        typ = datetime.date
    elif dtype == pl.Datetime:
        typ = datetime.datetime
    elif dtype == pl.Duration:
        typ = datetime.timedelta
    elif dtype == pl.Null:
        typ = Any
    elif dtype.__class__.__name__ == "Struct":
        typ, _ = _infer_struct_model(dtype, column_data, model_name, model_cache, force_optional=force_optional)
    elif dtype.__class__.__name__ == "List":
        inner_type, _ = _resolve_dtype(dtype.inner, None, model_name, model_cache, force_optional=force_optional)
        typ = List[inner_type]
    else:
        typ = Any

    if nullable:
        typ = Optional[typ]

    return typ, default


def infer_pydantic_model(
    df: pl.DataFrame,
    model_name: str = "AutoPolarsModel",
    force_optional: bool = False,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a Polars DataFrame schema.

    Args:
        df: Polars DataFrame to infer from.
        model_name: Optional model class name.
        force_optional: If True, force all fields (including nested) to be Optional.

    Returns:
        A dynamically created Pydantic model class.
    """

    model_cache: Dict[str, Type[BaseModel]] = {}
    fields: Dict[str, tuple] = {}

    for name, dtype in df.schema.items():
        col = df.get_column(name)
        field_type, default = _resolve_dtype(dtype, col, model_name, model_cache, force_optional=force_optional)
        fields[name] = (field_type, default)

    return create_model(model_name, **fields)
