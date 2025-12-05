"""
SQLModel inference utilities for Articuno.

Provides functions to convert SQLModel model classes to/from Pydantic models.
SQLModel already extends Pydantic, so conversion is relatively straightforward.
"""

from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    pass


def infer_pydantic_model_from_sqlmodel(
    sqlmodel_class: Type[Any],
    model_name: Optional[str] = None,
    force_optional: bool = False,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a SQLModel class.

    Since SQLModel extends Pydantic, we can extract the Pydantic schema directly.
    However, we create a new Pydantic model to avoid SQLModel-specific dependencies.

    Parameters
    ----------
    sqlmodel_class : Type[Any]
        SQLModel class.
    model_name : str, optional
        Name for the generated Pydantic model. If None, uses the SQLModel class name.
    force_optional : bool, default False
        If True, all fields will be Optional (note: this may override existing optionality).

    Returns
    -------
    Type[BaseModel]
        Dynamically created Pydantic model class.

    Raises
    ------
    ValueError
        If `model_name` is not a valid Python identifier.
    ImportError
        If SQLModel is not installed.
    """
    try:
        from pydantic import create_model
    except ImportError:
        raise ImportError(
            "SQLModel is required for this functionality. "
            "Install it with: pip install articuno[sqlmodel]"
        )

    if model_name is None:
        model_name = sqlmodel_class.__name__

    if not model_name.isidentifier():
        raise ValueError(
            f"Invalid model name '{model_name}'. Model names must be valid Python identifiers "
            "(start with letter/underscore, contain only letters/digits/underscores)."
        )

    # Get Pydantic model fields from SQLModel
    # SQLModel is a subclass of Pydantic BaseModel, so we can use model_fields
    fields: Dict[str, tuple] = {}

    for field_name, field_info in sqlmodel_class.model_fields.items():
        field_type = field_info.annotation
        default_value = field_info.default

        # Apply force_optional if requested
        if force_optional:
            from typing import Optional as _Opt

            # Check if already Optional
            origin = getattr(field_type, "__origin__", None)
            if origin is not None:
                if hasattr(origin, "__name__") and origin.__name__ == "Union":
                    args = getattr(field_type, "__args__", ())
                    if type(None) not in args:
                        field_type = _Opt[field_type]
            else:
                field_type = _Opt[field_type]
            default_value = None
        elif default_value is not ... and default_value is not None:
            # Keep existing default
            pass
        elif default_value is ...:
            # Required field
            pass
        else:
            # None default means Optional
            from typing import Optional as _Opt

            origin = getattr(field_type, "__origin__", None)
            if origin is None or (
                hasattr(origin, "__name__") and origin.__name__ != "Union"
            ):
                field_type = _Opt[field_type]

        fields[field_name] = (field_type, default_value)

    return create_model(model_name, **fields)  # type: ignore[call-overload, no-any-return]


def pydantic_to_sqlmodel(
    pydantic_model: Type[BaseModel],
    model_name: Optional[str] = None,
    table_name: Optional[str] = None,
) -> Type[Any]:
    """
    Generate a SQLModel class from a Pydantic model.

    Parameters
    ----------
    pydantic_model : Type[BaseModel]
        Pydantic model class.
    model_name : str, optional
        Name for the generated SQLModel. If None, uses the Pydantic model name.
    table_name : str, optional
        Table name for SQLModel. If None, uses lowercase model name.

    Returns
    -------
    Type[Any]
        Dynamically created SQLModel class.

    Raises
    ------
    ImportError
        If SQLModel is not installed.
    """
    try:
        from sqlmodel import SQLModel, Field
        from typing import get_origin, get_args
    except ImportError:
        raise ImportError(
            "SQLModel is required for this functionality. "
            "Install it with: pip install articuno[sqlmodel]"
        )

    if model_name is None:
        model_name = pydantic_model.__name__

    if table_name is None:
        table_name = model_name.lower()

    # Get Pydantic model fields
    fields = pydantic_model.model_fields

    # Build class definition string
    # SQLModel requires fields to be defined as: field_name: type = Field(...)
    field_defs = []
    for field_name, field_info in fields.items():
        field_type = field_info.annotation
        default_value = field_info.default

        # Format type annotation as string - need to handle Union types, etc.
        import typing

        if hasattr(typing, "get_origin") and get_origin(field_type):
            # For generic types like Union, Optional, etc., use string representation
            type_str = str(field_type).replace("typing.", "").replace("builtins.", "")
            # Clean up common patterns
            type_str = type_str.replace("Union[", "").replace("]", "")
            if "None" in type_str:
                # Handle Optional - convert to | None syntax or keep as is
                non_none_type = next(
                    (t for t in get_args(field_type) if t is not type(None)), None
                )
                if non_none_type:
                    type_str = f"{non_none_type.__name__} | None"
        elif field_type is not None and hasattr(field_type, "__name__"):
            type_str = field_type.__name__
        else:
            type_str = str(field_type)

        # Create Field() call
        # In Pydantic v2, PydanticUndefined means no default (not ...)
        # Check if default_value is PydanticUndefined by checking its string representation
        # or by checking if it's not None and not a regular value
        default_str = str(default_value)
        is_undefined = (
            default_str == "PydanticUndefined"
            or default_value is ...
            or (
                hasattr(default_value, "__class__")
                and "Undefined" in default_value.__class__.__name__
            )
        )

        # Check if this should be a primary key (id field or first int field)
        field_names_list = list(fields.keys())
        # Check if field_type is int (handle Optional[int] by checking origin)
        is_int_type = field_type is int or (
            field_type is not None
            and get_origin(field_type) is None
            and hasattr(field_type, "__name__")
            and field_type.__name__ == "int"
        )
        is_primary_key = field_name == "id" or (
            field_name == field_names_list[0] and is_int_type
        )
        primary_key_arg = ", primary_key=True" if is_primary_key else ""

        if is_undefined:
            # Required field - no default
            if is_primary_key:
                # Primary keys in SQLModel should be Optional[int] with default=None
                field_str = f"{field_name}: {type_str} | None = Field(default=None{primary_key_arg})"
            else:
                field_str = f"{field_name}: {type_str} = Field()"
        elif default_value is None:
            # Optional field with None default
            field_str = f"{field_name}: {type_str} | None = Field(default=None{primary_key_arg})"
        else:
            # Field with a default value
            default_repr = repr(default_value)
            field_str = f"{field_name}: {type_str} = Field(default={default_repr}{primary_key_arg})"

        field_defs.append(f"    {field_str}")

    # Build class definition
    class_def = f"""class {model_name}(SQLModel, table=True):
    __tablename__ = {repr(table_name)}
{chr(10).join(field_defs)}
"""

    # Execute in a namespace with SQLModel, Field, and common types available
    import datetime as dt

    namespace_exec = {
        "SQLModel": SQLModel,
        "Field": Field,
        "datetime": dt.datetime,
        "date": dt.date,
        "time": dt.time,
        "timedelta": dt.timedelta,
        "__name__": model_name,
    }

    exec(class_def, namespace_exec)

    sqlmodel_class = namespace_exec[model_name]
    return sqlmodel_class  # type: ignore[return-value]
