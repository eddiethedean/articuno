from typing import Any, Dict, List, Optional, Type
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
        Polars DataFrame to infer from.
    model_name : str, optional
        Name of the root Pydantic model class.
    _model_cache : dict, optional
        Cache for nested models.

    Returns
    -------
    Type[BaseModel]
        Generated Pydantic model class.
    """
    if _model_cache is None:
        _model_cache = {}

    def wrap_optional(base_type: Any, nullable: bool) -> Any:
        from typing import Optional
        return Optional[base_type] if nullable else base_type

    def get_default_value(col: pl.Series) -> Any:
        for val in col:
            if val is not None:
                return val
        return None

    def resolve_dtype(
        dtype: pl.DataType,
        column_data: Optional[pl.Series] = None,
        prefix: str = "",
    ) -> Any:
        nullable = column_data is not None and column_data.is_null().any()
        default = get_default_value(column_data) if column_data is not None else None

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
                        if column_data is not None
                        else None
                    )
                    field_type, field_default = resolve_dtype(
                        field.dtype,
                        column_data=field_data,
                        prefix=f"{prefix}{field.name}."
                    )
                    is_field_nullable = field_data is not None and field_data.is_null().any()
                    from typing import Optional
                    fields[field.name] = (
                        Optional[field_type] if is_field_nullable else field_type,
                        field_default if is_field_nullable else ...
                    )
                struct_model = create_model(f"{model_name}_{len(_model_cache)}_Struct", **fields)
                _model_cache[struct_key] = struct_model
            return wrap_optional(struct_model, nullable), default

        elif dtype.__class__.__name__ == "List":
            from typing import List
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
    df: pl.DataFrame,
    model_name: str = "AutoModel",
    _model_cache: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Infer a Patito model class from a Polars DataFrame schema.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame to infer from.
    model_name : str, optional
        Name of the root Patito model class.
    _model_cache : dict, optional
        Cache for nested models.

    Returns
    -------
    Any
        Generated Patito model class.
    """
    try:
        import patito
    except ImportError as e:
        raise ImportError(
            "The 'patito' package is required to use infer_patito_model. "
            "Please install it with `pip install patito`."
        ) from e

    if _model_cache is None:
        _model_cache = {}

    def wrap_optional(base_type: Any, nullable: bool) -> Any:
        from typing import Optional
        return Optional[base_type] if nullable else base_type

    def resolve_dtype(dtype: pl.DataType, column_data: Optional[pl.Series] = None) -> Any:
        nullable = column_data is not None and column_data.is_null().any()

        if dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
            return wrap_optional(int, nullable)
        elif dtype in {pl.Float32, pl.Float64}:
            return wrap_optional(float, nullable)
        elif dtype == pl.Boolean:
            return wrap_optional(bool, nullable)
        elif dtype == pl.Utf8:
            return wrap_optional(str, nullable)
        elif dtype == pl.Date:
            return wrap_optional(patito.Date, nullable)
        elif dtype == pl.Datetime:
            return wrap_optional(patito.Datetime, nullable)
        elif dtype == pl.Duration:
            return wrap_optional(patito.Duration, nullable)
        elif dtype == pl.Null:
            return type(None)

        elif dtype.__class__.__name__ == "Struct":
            struct_key = str(dtype)
            if struct_key in _model_cache:
                return _model_cache[struct_key]
            fields = {}
            for field in dtype.fields:
                field_data = (
                    column_data.struct.field(field.name)
                    if column_data is not None
                    else None
                )
                field_type = resolve_dtype(field.dtype, column_data=field_data)
                fields[field.name] = field_type
            model_cls = patito.model(model_name + "_Struct")(type(model_name + "_Struct", (object,), fields))
            _model_cache[struct_key] = model_cls
            return model_cls

        elif dtype.__class__.__name__ == "List":
            from typing import List
            inner_type = resolve_dtype(dtype.inner)
            return wrap_optional(List[inner_type], nullable)

        return wrap_optional(Any, nullable)

    fields = {}
    for name, dtype in df.schema.items():
        col = df.get_column(name)
        fields[name] = resolve_dtype(dtype, column_data=col)

    model_cls = patito.model(model_name)(type(model_name, (object,), fields))
    return model_cls


def df_to_pydantic(
    df: pl.DataFrame,
    model: Optional[Type[BaseModel]] = None,
    model_name: Optional[str] = None,
) -> List[BaseModel]:
    """
    Convert a Polars DataFrame to a list of Pydantic model instances.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to convert.
    model : Type[BaseModel], optional
        An existing Pydantic model class to use for conversion.
        If None, a model will be inferred from the DataFrame.
    model_name : str, optional
        The name to use if inferring the model.

    Returns
    -------
    List[BaseModel]
        A list of Pydantic model instances corresponding to DataFrame rows.
    """
    if model is None:
        model = infer_pydantic_model(df, model_name=model_name or "AutoModel")
    return [model(**row) for row in df.to_dicts()]


def df_to_patito(
    df: pl.DataFrame,
    model: Optional[Any] = None,
    model_name: Optional[str] = None,
) -> List[Any]:
    """
    Convert a Polars DataFrame to a list of Patito model instances.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to convert.
    model : Any, optional
        An existing Patito model class to use for conversion.
        If None, a model will be inferred from the DataFrame.
    model_name : str, optional
        The name to use if inferring the model.

    Returns
    -------
    List[Any]
        A list of Patito model instances corresponding to DataFrame rows.
    """
    if model is None:
        model = infer_patito_model(df, model_name=model_name or "AutoModel")
    return [model(**row) for row in df.to_dicts()]
