from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, create_model
import polars as pl
import datetime

try:
    import patito as pt
    _has_patito = True
except ImportError:
    _has_patito = False


def df_to_pydantic(
    df: pl.DataFrame,
    model: Optional[Type[BaseModel]] = None,
    model_name: Optional[str] = None,
) -> List[BaseModel]:
    """
    Convert a Polars DataFrame into a list of Pydantic model instances.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to convert.
    model : Type[BaseModel], optional
        A Pydantic model to use. If None, a model is inferred from the DataFrame.
    model_name : str, optional
        Name to use when inferring the model class.

    Returns
    -------
    List[BaseModel]
        A list of Pydantic model instances.
    """
    if model is None:
        model = infer_pydantic_model(df, model_name=model_name or "AutoModel")
    return [model(**row) for row in df.to_dicts()]


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
        The DataFrame to infer the schema from.
    model_name : str
        The name of the generated Pydantic model class.
    _model_cache : dict, optional
        Internal cache used to prevent regenerating nested models.

    Returns
    -------
    Type[BaseModel]
        The generated Pydantic model class.
    """
    if _model_cache is None:
        _model_cache = {}

    def wrap_optional(tp: Any, nullable: bool) -> Any:
        return Optional[tp] if nullable else tp

    def get_first_non_null(col: pl.Series) -> Any:
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


def df_to_patito(
    df: pl.DataFrame,
    model: Optional[Type["pt.Model"]] = None,
    model_name: Optional[str] = None,
) -> List["pt.Model"]:
    """
    Convert a Polars DataFrame into a list of Patito model instances.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to convert.
    model : Type[pt.Model], optional
        A Patito model to use. If None, a model is inferred from the DataFrame.
    model_name : str, optional
        Name to use when inferring the model class.

    Returns
    -------
    List[pt.Model]
        A list of Patito model instances.

    Raises
    ------
    ImportError
        If Patito is not installed.
    """
    if not _has_patito:
        raise ImportError("Patito is not installed. Try `pip install patito`.")

    if model is None:
        model = infer_patito_model(df, model_name=model_name or "AutoPatitoModel")

    return [model(**row) for row in df.to_dicts()]


def infer_patito_model(
    df: pl.DataFrame,
    model_name: str = "AutoPatitoModel",
) -> Type["pt.Model"]:
    """
    Infer a Patito model class from a Polars DataFrame schema.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to infer the schema from.
    model_name : str
        The name of the generated Patito model class.

    Returns
    -------
    Type[pt.Model]
        The generated Patito model class.

    Raises
    ------
    ImportError
        If Patito is not installed.
    """
    if not _has_patito:
        raise ImportError("Patito is not installed. Try `pip install patito`.")

    fields = {}

    for name, dtype in df.schema.items():
        col = df.get_column(name)
        nullable = col.is_null().any()

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
            typ = type(None)
        else:
            typ = Any

        if nullable:
            typ = Optional[typ]

        fields[name] = pt.Field(typ)

    return type(model_name, (pt.Model,), fields)
