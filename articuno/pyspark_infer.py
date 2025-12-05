"""
PySpark DataFrame inference utilities for Articuno.

Provides functions to convert PySpark DataFrames to/from Pydantic models
with proper type mapping from Spark types.
"""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, create_model
import datetime


def _pyspark_type_to_pydantic(
    spark_type: Any,
    nullable: bool,
    force_optional: bool,
    field_name: str = "",
) -> tuple[Any, Any]:
    """
    Map PySpark data type to Pydantic type.

    Parameters
    ----------
    spark_type : Any
        PySpark DataType instance.
    nullable : bool
        Whether the field is nullable.
    force_optional : bool
        If True, force field to be Optional.
    field_name : str
        Field name (used for nested model naming).

    Returns
    -------
    tuple[Any, Any]
        Tuple of (pydantic_type, default_value).
    """
    type_name = type(spark_type).__name__
    typ: Any
    default: Any

    # Integer types
    if type_name in ("IntegerType", "LongType", "ShortType", "ByteType"):
        typ = int
    # Float types
    elif type_name in ("FloatType", "DoubleType", "DecimalType"):
        typ = float
    # String types
    elif type_name == "StringType":
        typ = str
    # Boolean
    elif type_name == "BooleanType":
        typ = bool
    # Timestamp
    elif type_name == "TimestampType":
        typ = datetime.datetime
    # Date
    elif type_name == "DateType":
        typ = datetime.date
    # Array
    elif type_name == "ArrayType":
        element_type = spark_type.elementType
        element_pydantic, _ = _pyspark_type_to_pydantic(
            element_type, False, force_optional, field_name
        )
        # element_pydantic is a type, use Any for List element type
        typ = List[Any]  # type: ignore[valid-type]
    # Struct (nested)
    elif type_name == "StructType":
        # Create nested model

        # We'll need to handle this differently - create fields dict
        nested_fields: Dict[str, tuple] = {}
        for field in spark_type.fields:
            field_pydantic, field_default = _pyspark_type_to_pydantic(
                field.dataType, field.nullable, force_optional, field.name
            )
            nested_fields[field.name] = (field_pydantic, field_default)
        nested_model_name = f"{field_name}_NestedModel" if field_name else "NestedModel"
        typ = create_model(nested_model_name, **nested_fields)  # type: ignore[call-overload]
    # Map (treat as dict/Any)
    elif type_name == "MapType":
        typ = Dict[str, Any]
    # Binary
    elif type_name == "BinaryType":
        typ = bytes
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


def infer_pydantic_model(
    df: Any,
    model_name: str = "AutoPySparkModel",
    force_optional: bool = False,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a PySpark DataFrame schema.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        PySpark DataFrame to infer model from.
    model_name : str, optional
        Desired model class name.
    force_optional : bool, optional
        If True, force all fields Optional.

    Returns
    -------
    Type[BaseModel]
        Dynamically created Pydantic model class.

    Raises
    ------
    ValueError
        If `model_name` is not a valid Python identifier.
    ImportError
        If PySpark is not installed.
    """
    try:
        pass  # PySpark import check
    except ImportError:
        raise ImportError(
            "PySpark is required for this functionality. "
            "Install it with: pip install articuno[pyspark]"
        )

    if not model_name.isidentifier():
        raise ValueError(
            f"Invalid model name '{model_name}'. Model names must be valid Python identifiers "
            "(start with letter/underscore, contain only letters/digits/underscores)."
        )

    # Get schema
    schema = df.schema
    fields: Dict[str, tuple] = {}

    # Iterate through schema fields
    for field in schema.fields:
        field_name = field.name
        field_type = field.dataType
        nullable = field.nullable

        typ, default = _pyspark_type_to_pydantic(
            field_type, nullable, force_optional, field_name
        )
        fields[field_name] = (typ, default)

    return create_model(model_name, **fields)  # type: ignore[call-overload, no-any-return]


def pydantic_to_pyspark(
    pydantic_instances: List[BaseModel],
    model: Optional[Type[BaseModel]] = None,
) -> Any:
    """
    Convert a list of Pydantic model instances to a PySpark DataFrame.

    Parameters
    ----------
    pydantic_instances : List[BaseModel]
        List of Pydantic model instances to convert.
    model : Type[BaseModel], optional
        Pydantic model class. If None, inferred from first instance.

    Returns
    -------
    pyspark.sql.DataFrame
        PySpark DataFrame containing the data.

    Raises
    ------
    ImportError
        If PySpark is not installed.
    ValueError
        If instances list is empty and model is not provided.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise ImportError(
            "PySpark is required for this functionality. "
            "Install it with: pip install articuno[pyspark]"
        )

    if not pydantic_instances:
        if model is None:
            raise ValueError(
                "Cannot create DataFrame: instances list is empty and model is not provided."
            )
        # Create empty DataFrame with schema
        spark = SparkSession.getActiveSession()
        if spark is None:
            spark = (
                SparkSession.builder.appName("articuno")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .getOrCreate()
            )

        schema = _pydantic_to_spark_schema(model)
        return spark.createDataFrame([], schema)

    # Use first instance to infer model if not provided
    if model is None:
        model = type(pydantic_instances[0])

    # Convert instances to dicts
    rows = [instance.model_dump() for instance in pydantic_instances]

    # Create Spark session
    spark = SparkSession.getActiveSession()
    if spark is None:
        spark = (
            SparkSession.builder.appName("articuno")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate()
        )

    # Create DataFrame
    return spark.createDataFrame(rows, schema=_pydantic_to_spark_schema(model))  # type: ignore[type-var]


def _pydantic_to_spark_schema(pydantic_model: Type[BaseModel]) -> Any:
    """
    Convert a Pydantic model to a PySpark StructType schema.

    Parameters
    ----------
    pydantic_model : Type[BaseModel]
        Pydantic model class.

    Returns
    -------
    pyspark.sql.types.StructType
        PySpark schema.
    """
    try:
        from pyspark.sql.types import (
            StructType,
            StructField,
            LongType,
            DoubleType,
            StringType,
            BooleanType,
            TimestampType,
            DateType,
            ArrayType,
            MapType,
            BinaryType,
        )
        from typing import get_origin, get_args
    except ImportError:
        raise ImportError(
            "PySpark is required for this functionality. "
            "Install it with: pip install articuno[pyspark]"
        )

    fields = pydantic_model.model_fields
    struct_fields = []

    for field_name, field_info in fields.items():
        field_type = field_info.annotation
        is_optional = False

        # Check if Optional
        origin = get_origin(field_type)
        if origin is not None:
            args = get_args(field_type)
            if type(None) in args:
                is_optional = True
                # Get the non-None type
                field_type = next(arg for arg in args if arg is not type(None))

        # Map Python types to PySpark types
        from pyspark.sql.types import DataType

        spark_type: DataType
        if field_type is int:
            spark_type = LongType()
        elif field_type is float:
            spark_type = DoubleType()
        elif field_type is str:
            spark_type = StringType()
        elif field_type is bool:
            spark_type = BooleanType()
        elif field_type is datetime.datetime:
            spark_type = TimestampType()
        elif field_type is datetime.date:
            spark_type = DateType()
        elif field_type is bytes:
            spark_type = BinaryType()
        elif (
            hasattr(field_type, "__origin__")
            and field_type is not None
            and field_type.__origin__ is list
        ):
            # List type
            args = get_args(field_type)
            element_type = args[0] if args else StringType()
            element_spark_type = _python_type_to_spark_type(element_type)
            spark_type = ArrayType(element_spark_type)
        elif (
            hasattr(field_type, "__origin__")
            and field_type is not None
            and field_type.__origin__ is dict
        ):
            # Map type
            spark_type = MapType(StringType(), StringType())  # Simplified
        else:
            # Default to String for unknown types
            spark_type = StringType()

        struct_fields.append(StructField(field_name, spark_type, nullable=is_optional))

    return StructType(struct_fields)


def _python_type_to_spark_type(python_type: Any) -> Any:
    """
    Convert a Python type to a PySpark DataType.

    Parameters
    ----------
    python_type : Any
        Python type.

    Returns
    -------
    pyspark.sql.types.DataType
        PySpark data type.
    """
    try:
        from pyspark.sql.types import (
            LongType,
            DoubleType,
            StringType,
            BooleanType,
            TimestampType,
            DateType,
        )
    except ImportError:
        raise ImportError("PySpark is required for this functionality.")

    if python_type is int:
        return LongType()
    elif python_type is float:
        return DoubleType()
    elif python_type is str:
        return StringType()
    elif python_type is bool:
        return BooleanType()
    elif python_type is datetime.datetime:
        return TimestampType()
    elif python_type is datetime.date:
        return DateType()
    else:
        return StringType()  # Default
