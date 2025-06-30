from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
)
from datetime import datetime, date, time, timedelta
from decimal import Decimal

import polars as pl
from pydantic import BaseModel, create_model


def df_to_pydantic(
    df: pl.DataFrame,
    model: Optional[Type[BaseModel]] = None,
    model_name: str = "AutoModel"
) -> List[BaseModel]:
    """
    Convert a Polars DataFrame to a list of Pydantic model instances.
    If no model is provided, infer one from the DataFrame schema.
    """
    if model is None:
        model = infer_pydantic_model(df, model_name=model_name)
    return [model(**row) for row in df.to_dicts()]


def infer_pydantic_model(
    df: pl.DataFrame,
    model_name: str = "AutoModel",
    _model_cache: Optional[Dict[str, Type[BaseModel]]] = None
) -> Type[BaseModel]:
    """
    Infer a Pydantic model from a Polars DataFrame schema.
    Handles nested Struct types recursively with caching.

    Args:
        df: The Polars DataFrame.
        model_name: The base name for the generated Pydantic model.
        _model_cache: Internal cache to avoid duplicate model generation.

    Returns:
        A dynamically generated Pydantic BaseModel subclass.
    """
    if _model_cache is None:
        _model_cache = {}

    pl_to_py = {
        pl.Int8: int,
        pl.Int16: int,
        pl.Int32: int,
        pl.Int64: int,
        pl.UInt8: int,
        pl.UInt16: int,
        pl.UInt32: int,
        pl.UInt64: int,
        pl.Float32: float,
        pl.Float64: float,
        pl.Utf8: str,
        pl.Boolean: bool,
        pl.Date: date,
        pl.Datetime: datetime,
        pl.Time: time,
        pl.Duration: timedelta,
        pl.Object: Any,
        pl.Null: type(None),
        pl.Categorical: str,
        pl.Enum: str,
        pl.Decimal: Decimal,
        pl.Binary: bytes,
    }

    def resolve_dtype(dtype: pl.DataType) -> Any:
        """
        Recursively map a Polars dtype to a Python/Pydantic type.
        Handles nested lists and structs.
        Wraps nullable types in Optional[].
        """
        nullable = getattr(dtype, "nullable", False)

        if isinstance(dtype, pl.List):
            inner_type = resolve_dtype(dtype.inner)
            py_type = List[inner_type]
        elif isinstance(dtype, pl.Struct):
            struct_key = str(dtype)
            if struct_key in _model_cache:
                py_type = _model_cache[struct_key]
            else:
                fields = {
                    field_name: (resolve_dtype(field_type), ...)
                    for field_name, field_type in dtype.fields.items()
                }
                model_cls = create_model(f"{model_name}_{len(_model_cache)}_Struct", **fields)
                _model_cache[struct_key] = model_cls
                py_type = model_cls
        else:
            py_type = pl_to_py.get(dtype, Any)

        if nullable:
            return Optional[py_type]
        return py_type

    fields: Dict[str, tuple] = {
        name: (resolve_dtype(dtype), ...)
        for name, dtype in df.schema.items()
    }

    return create_model(model_name, **fields)
