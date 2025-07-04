from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, create_model
import polars as pl
import datetime

_model_cache: Dict[str, Type[BaseModel]] = {}

def infer_pydantic_model(
    df: pl.DataFrame,
    model_name: str = "AutoModel",
    _model_cache: Optional[Dict[str, Type[BaseModel]]] = None,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a Polars DataFrame schema.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to infer from.
    model_name : str, optional
        Name of the root Pydantic model class.
    _model_cache : dict[str, Type[BaseModel]], optional
        Internal cache to reuse nested model classes (e.g., for struct fields).

    Returns
    -------
    Type[BaseModel]
        A Pydantic model class representing the inferred schema.
    """
    if _model_cache is None:
        _model_cache = {}

    def resolve_dtype(dtype: pl.DataType) -> Any:
        if dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
            return int
        if dtype in {pl.Float32, pl.Float64}:
            return float
        if dtype == pl.Boolean:
            return bool
        if dtype == pl.Utf8:
            return str
        if dtype == pl.Date:
            return datetime.date
        if dtype == pl.Datetime:
            return datetime.datetime
        if dtype == pl.Duration:
            return datetime.timedelta
        if dtype == pl.Null:
            return type(None)

        if dtype.__class__.__name__ == "List":
            inner_type = resolve_dtype(dtype.inner)
            return List[inner_type]

        if dtype.__class__.__name__ == "Struct":
            struct_key = str(dtype)
            if struct_key in _model_cache:
                return _model_cache[struct_key]
            else:
                fields = {
                    field.name: (resolve_dtype(field.dtype), ...)
                    for field in dtype.fields
                }
                model_cls = create_model(f"{model_name}_{len(_model_cache)}_Struct", **fields)
                _model_cache[struct_key] = model_cls
                return model_cls

        return Any

    def wrap_nullable(name: str, dtype: pl.DataType, typ: Any) -> Any:
        if dtype.__class__.__name__ in ("Struct", "List"):
            return typ
        is_nullable = df.select(pl.col(name).is_null().any()).item()
        return Optional[typ] if is_nullable else typ

    fields: Dict[str, tuple] = {
        name: (wrap_nullable(name, dtype, resolve_dtype(dtype)), ...)
        for name, dtype in df.schema.items()
    }

    return create_model(model_name, **fields)
