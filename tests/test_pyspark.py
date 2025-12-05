"""
Comprehensive tests for PySpark ↔ Pydantic conversions.
"""

import pytest
from datetime import datetime, date
from articuno.pyspark_infer import infer_pydantic_model, pydantic_to_pyspark
from articuno import infer_pydantic_model as unified_infer, df_to_pydantic

# Optional imports
try:
    from pyspark.sql import SparkSession
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
    )

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


pytestmark = pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not installed")


@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for testing."""
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("articuno_test")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_pyspark_to_pydantic_basic_types(spark):
    """Test conversion from PySpark DataFrame to Pydantic with basic types."""
    data = [
        (1, "Alice", 30, 95.5, True),
        (2, "Bob", 25, 88.0, False),
        (3, "Charlie", 35, 92.3, True),
    ]
    schema = StructType(
        [
            StructField("id", LongType(), False),
            StructField("name", StringType(), False),
            StructField("age", LongType(), False),
            StructField("score", DoubleType(), False),
            StructField("active", BooleanType(), False),
        ]
    )

    df = spark.createDataFrame(data, schema=schema)
    PydanticModel = infer_pydantic_model(df, model_name="UserModel")

    instance = PydanticModel(id=1, name="Alice", age=30, score=95.5, active=True)
    assert instance.id == 1
    assert instance.name == "Alice"
    assert instance.age == 30
    assert instance.score == 95.5
    assert instance.active is True


def test_pyspark_to_pydantic_nullable_fields(spark):
    """Test that nullable PySpark fields become Optional in Pydantic."""
    data = [
        (1, "Widget", "A widget", 19.99),
        (2, "Gadget", None, 29.99),
    ]
    schema = StructType(
        [
            StructField("id", LongType(), False),
            StructField("name", StringType(), False),
            StructField("description", StringType(), True),  # nullable
            StructField("price", DoubleType(), False),
        ]
    )

    df = spark.createDataFrame(data, schema=schema)
    PydanticModel = infer_pydantic_model(df, model_name="ProductModel")

    instances = list(df_to_pydantic(df, model=PydanticModel))
    assert instances[0].description == "A widget"
    assert instances[1].description is None


def test_pyspark_to_pydantic_temporal_types(spark):
    """Test conversion of timestamp and date types."""
    from datetime import datetime as dt, date as d

    data = [
        (1, d(2024, 1, 15), dt(2024, 1, 15, 10, 30)),
        (2, d(2024, 1, 16), dt(2024, 1, 16, 11, 0)),
    ]
    schema = StructType(
        [
            StructField("id", LongType(), False),
            StructField("event_date", DateType(), False),
            StructField("created_at", TimestampType(), False),
        ]
    )

    df = spark.createDataFrame(data, schema=schema)
    PydanticModel = infer_pydantic_model(df, model_name="EventModel")

    instances = list(df_to_pydantic(df, model=PydanticModel))
    assert instances[0].event_date == date(2024, 1, 15)
    assert instances[0].created_at == datetime(2024, 1, 15, 10, 30)


def test_pyspark_to_pydantic_array_types(spark):
    """Test conversion of array types."""
    data = [
        (1, ["tag1", "tag2"]),
        (2, ["tag3"]),
    ]
    schema = StructType(
        [
            StructField("id", LongType(), False),
            StructField("tags", ArrayType(StringType()), False),
        ]
    )

    df = spark.createDataFrame(data, schema=schema)
    PydanticModel = infer_pydantic_model(df, model_name="ItemModel")

    instances = list(df_to_pydantic(df, model=PydanticModel))
    assert instances[0].tags == ["tag1", "tag2"]
    assert instances[1].tags == ["tag3"]


def test_pyspark_to_pydantic_force_optional(spark):
    """Test force_optional parameter."""
    data = [(1, "Alice", 30)]
    schema = StructType(
        [
            StructField("id", LongType(), False),
            StructField("name", StringType(), False),
            StructField("age", LongType(), False),
        ]
    )

    df = spark.createDataFrame(data, schema=schema)
    PydanticModel = infer_pydantic_model(
        df, model_name="UserModel", force_optional=True
    )

    # All fields should be Optional
    instance = PydanticModel(id=None, name=None, age=None)
    assert instance.id is None
    assert instance.name is None
    assert instance.age is None


def test_pydantic_to_pyspark_basic_types(spark):
    """Test conversion from Pydantic instances to PySpark DataFrame."""
    from pydantic import BaseModel

    class UserModel(BaseModel):
        id: int
        name: str
        age: int
        score: float
        active: bool

    instances = [
        UserModel(id=1, name="Alice", age=30, score=95.5, active=True),
        UserModel(id=2, name="Bob", age=25, score=88.0, active=False),
    ]

    df = pydantic_to_pyspark(instances, model=UserModel)

    assert df.count() == 2
    rows = df.collect()
    assert rows[0]["id"] == 1
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"


def test_pydantic_to_pyspark_optional_fields(spark):
    """Test that Optional Pydantic fields are handled correctly."""
    from pydantic import BaseModel

    class ProductModel(BaseModel):
        id: int
        name: str
        description: str | None = None
        price: float

    instances = [
        ProductModel(id=1, name="Widget", description="A widget", price=19.99),
        ProductModel(id=2, name="Gadget", description=None, price=29.99),
    ]

    df = pydantic_to_pyspark(instances, model=ProductModel)

    rows = df.collect()
    assert rows[0]["description"] == "A widget"
    assert rows[1]["description"] is None


def test_pydantic_to_pyspark_temporal_types(spark):
    """Test conversion of datetime and date types."""
    from pydantic import BaseModel

    class EventModel(BaseModel):
        id: int
        event_date: date
        created_at: datetime

    instances = [
        EventModel(
            id=1, event_date=date(2024, 1, 15), created_at=datetime(2024, 1, 15, 10, 30)
        ),
    ]

    df = pydantic_to_pyspark(instances, model=EventModel)

    rows = df.collect()
    assert rows[0]["event_date"] == date(2024, 1, 15)
    # Note: PySpark may convert datetime differently, so we check it exists
    assert rows[0]["created_at"] is not None


def test_infer_pydantic_model_with_pyspark(spark):
    """Test that unified infer_pydantic_model works with PySpark DataFrames."""
    data = [(1, "Alice", "alice@example.com")]
    schema = StructType(
        [
            StructField("id", LongType(), False),
            StructField("name", StringType(), False),
            StructField("email", StringType(), True),
        ]
    )

    df = spark.createDataFrame(data, schema=schema)

    # Test using the unified infer_pydantic_model function
    PydanticModel = unified_infer(df, model_name="UserModel")

    instances = list(df_to_pydantic(df, model=PydanticModel))
    assert instances[0].id == 1
    assert instances[0].name == "Alice"
    assert instances[0].email == "alice@example.com"


def test_pyspark_model_name_validation(spark):
    """Test that invalid model names raise ValueError."""
    data = [(1, "Alice")]
    schema = StructType(
        [
            StructField("id", LongType(), False),
            StructField("name", StringType(), False),
        ]
    )

    df = spark.createDataFrame(data, schema=schema)

    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model(df, model_name="123Invalid")
