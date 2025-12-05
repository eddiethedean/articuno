"""
SQLAlchemy model inference utilities for Articuno.

Provides functions to convert SQLAlchemy declarative model classes
to/from Pydantic models with proper type mapping.
"""

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from pydantic import BaseModel, create_model
import datetime

if TYPE_CHECKING:
    pass


def _sqlalchemy_type_to_pydantic(
    column_type: Any, nullable: bool, force_optional: bool
) -> tuple[Any, Any]:
    """
    Map SQLAlchemy column type to Pydantic type.

    Parameters
    ----------
    column_type : Any
        SQLAlchemy column type instance.
    nullable : bool
        Whether the column is nullable.
    force_optional : bool
        If True, force field to be Optional.

    Returns
    -------
    tuple[Any, Any]
        Tuple of (pydantic_type, default_value).
    """
    type_name = type(column_type).__name__
    typ: Any
    default: Any

    # Integer types
    if type_name in ("Integer", "BigInteger", "SmallInteger"):
        typ = int
    # Float types
    elif type_name in ("Float", "Numeric", "DECIMAL"):
        typ = float
    # String types
    elif type_name in ("String", "Text", "Unicode", "UnicodeText"):
        typ = str
    # Boolean
    elif type_name == "Boolean":
        typ = bool
    # DateTime
    elif type_name == "DateTime":
        typ = datetime.datetime
    # Date
    elif type_name == "Date":
        typ = datetime.date
    # Time
    elif type_name == "Time":
        typ = datetime.time
    # JSON (treat as dict/Any)
    elif type_name in ("JSON", "JSONB"):
        typ = Any
    else:
        # Default to Any for unknown types
        typ = Any

    # Apply optionality
    if force_optional or nullable:
        from typing import Optional as _Opt

        typ = _Opt[typ]
        default = None
    else:
        default = ...

    return typ, default


def infer_pydantic_model_from_sqlalchemy(
    sqlalchemy_model: Type[Any],
    model_name: Optional[str] = None,
    force_optional: bool = False,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a SQLAlchemy declarative model.

    Parameters
    ----------
    sqlalchemy_model : Type[Any]
        SQLAlchemy declarative model class.
    model_name : str, optional
        Name for the generated Pydantic model. If None, uses the SQLAlchemy model name.
    force_optional : bool, default False
        If True, all fields will be Optional.

    Returns
    -------
    Type[BaseModel]
        Dynamically created Pydantic model class.

    Raises
    ------
    ValueError
        If `model_name` is not a valid Python identifier.
    ImportError
        If SQLAlchemy is not installed.
    """
    try:
        from sqlalchemy import inspect
    except ImportError:
        raise ImportError(
            "SQLAlchemy is required for this functionality. "
            "Install it with: pip install articuno[sqlalchemy]"
        )

    # Validate model name
    if model_name is None:
        model_name = sqlalchemy_model.__name__

    if not model_name.isidentifier():
        raise ValueError(
            f"Invalid model name '{model_name}'. Model names must be valid Python identifiers "
            "(start with letter/underscore, contain only letters/digits/underscores)."
        )

    # Get SQLAlchemy mapper
    mapper = inspect(sqlalchemy_model)
    fields: Dict[str, tuple] = {}

    # Iterate through columns
    for column in mapper.columns:
        column_type = column.type
        nullable = column.nullable
        field_name = column.key

        typ, default = _sqlalchemy_type_to_pydantic(
            column_type, nullable, force_optional
        )
        fields[field_name] = (typ, default)

    return create_model(model_name, **fields)  # type: ignore[call-overload, no-any-return]


def pydantic_to_sqlalchemy(
    pydantic_model: Type[BaseModel],
    model_name: Optional[str] = None,
    table_name: Optional[str] = None,
) -> Type[Any]:
    """
    Generate a SQLAlchemy declarative model class from a Pydantic model.

    Parameters
    ----------
    pydantic_model : Type[BaseModel]
        Pydantic model class.
    model_name : str, optional
        Name for the generated SQLAlchemy model. If None, uses the Pydantic model name.
    table_name : str, optional
        Table name for SQLAlchemy. If None, uses lowercase model name.

    Returns
    -------
    Type[Any]
        Dynamically created SQLAlchemy declarative model class.

    Raises
    ------
    ImportError
        If SQLAlchemy is not installed.
    """
    try:
        from sqlalchemy import (
            Column,
            Integer,
            String,
            Float,
            Boolean,
            DateTime,
            Date,
            Time,
            JSON,
        )
        from sqlalchemy.orm import DeclarativeBase
    except ImportError:
        raise ImportError(
            "SQLAlchemy is required for this functionality. "
            "Install it with: pip install articuno[sqlalchemy]"
        )

    if model_name is None:
        model_name = pydantic_model.__name__

    if table_name is None:
        table_name = model_name.lower()

    # Get Pydantic model fields
    fields = pydantic_model.model_fields

    # Create base class if needed
    class Base(DeclarativeBase):
        pass

    # Build attributes dict for SQLAlchemy model
    attrs: Dict[str, Any] = {
        "__tablename__": table_name,
    }

    # Map Pydantic fields to SQLAlchemy columns
    # Track if we have a primary key
    has_primary_key = False
    field_names = list(fields.keys())

    for field_name, field_info in fields.items():
        field_type = field_info.annotation
        is_optional = False

        # Check if Optional
        origin = getattr(field_type, "__origin__", None)
        if origin is not None:
            if hasattr(origin, "__name__") and origin.__name__ == "Union":
                args = getattr(field_type, "__args__", ())
                if len(args) == 2 and type(None) in args:
                    is_optional = True
                    # Get the non-None type
                    field_type = next(arg for arg in args if arg is not type(None))

        # Map Python types to SQLAlchemy types
        col_type: Any
        if field_type is int:
            col_type = Integer
        elif field_type is float:
            col_type = Float
        elif field_type is str:
            col_type = String(255)  # Default length
        elif field_type is bool:
            col_type = Boolean
        elif field_type is datetime.datetime:
            col_type = DateTime
        elif field_type is datetime.date:
            col_type = Date
        elif field_type == datetime.time:
            col_type = Time
        else:
            # Default to JSON for complex types
            col_type = JSON

        # If field is named 'id' or it's the first field and no primary key yet, make it primary key
        is_primary_key = False
        if field_name == "id" or (not has_primary_key and field_name == field_names[0]):
            is_primary_key = True
            has_primary_key = True
            # Primary keys should not be nullable
            is_optional = False

        attrs[field_name] = Column(
            col_type, nullable=is_optional, primary_key=is_primary_key
        )

    # Create SQLAlchemy model class
    sqlalchemy_model = type(model_name, (Base,), attrs)
    return sqlalchemy_model
