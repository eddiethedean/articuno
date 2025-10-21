"""
Comprehensive tests for Polars DataFrame inference.
"""

import pytest
from datetime import datetime, date, timedelta
from articuno import infer_pydantic_model, df_to_pydantic

# Optional imports
try:
    import polars as pl
except ImportError:
    pl = None


pytestmark = pytest.mark.skipif(pl is None, reason="polars not installed")


def test_basic_types():
    """Test inference of basic data types."""
    df = pl.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.1, 2.2, 3.3],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True]
    })
    
    Model = infer_pydantic_model(df, model_name="BasicModel")
    instance = Model(int_col=1, float_col=1.1, str_col="a", bool_col=True)
    
    assert instance.int_col == 1
    assert instance.float_col == 1.1
    assert instance.str_col == "a"
    assert instance.bool_col is True


def test_numeric_types():
    """Test various Polars numeric types."""
    df = pl.DataFrame({
        "int8": pl.Series([1, 2, 3], dtype=pl.Int8),
        "int16": pl.Series([100, 200, 300], dtype=pl.Int16),
        "int32": pl.Series([1000, 2000, 3000], dtype=pl.Int32),
        "int64": pl.Series([10000, 20000, 30000], dtype=pl.Int64),
        "uint8": pl.Series([1, 2, 3], dtype=pl.UInt8),
        "float32": pl.Series([1.1, 2.2, 3.3], dtype=pl.Float32),
        "float64": pl.Series([10.1, 20.2, 30.3], dtype=pl.Float64),
    })
    
    Model = infer_pydantic_model(df, model_name="NumericModel")
    instance = Model(
        int8=1, int16=100, int32=1000, int64=10000,
        uint8=1, float32=1.1, float64=10.1
    )
    
    assert instance.int8 == 1
    assert instance.float64 == 10.1


def test_date_datetime_types():
    """Test date and datetime type inference."""
    df = pl.DataFrame({
        "date_col": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
        "datetime_col": [
            datetime(2023, 1, 1, 12, 0),
            datetime(2023, 1, 2, 13, 30),
            datetime(2023, 1, 3, 14, 45)
        ]
    })
    
    Model = infer_pydantic_model(df, model_name="DateModel")
    instance = Model(
        date_col=date(2023, 1, 1),
        datetime_col=datetime(2023, 1, 1, 12, 0)
    )
    
    assert instance.date_col == date(2023, 1, 1)
    assert instance.datetime_col == datetime(2023, 1, 1, 12, 0)


def test_duration_type():
    """Test duration/timedelta type inference."""
    df = pl.DataFrame({
        "duration_col": [
            timedelta(days=1),
            timedelta(hours=2),
            timedelta(minutes=30)
        ]
    })
    
    Model = infer_pydantic_model(df, model_name="DurationModel")
    instance = Model(duration_col=timedelta(days=1))
    
    assert instance.duration_col == timedelta(days=1)


def test_nested_struct():
    """Test nested struct column inference."""
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "user": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35}
        ]
    })
    
    Model = infer_pydantic_model(df, model_name="NestedModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert instances[0].id == 1
    # Polars with poldantic creates nested models, not dicts
    if hasattr(instances[0].user, 'name'):
        assert instances[0].user.name == "Alice"
        assert instances[1].user.age == 25
    else:
        # Dict fallback
        assert instances[0].user["name"] == "Alice"
        assert instances[1].user["age"] == 25


def test_list_columns():
    """Test list column inference."""
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "tags": [["a", "b"], ["c"], ["d", "e", "f"]]
    })
    
    Model = infer_pydantic_model(df, model_name="ListModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert instances[0].tags == ["a", "b"]
    assert instances[1].tags == ["c"]
    assert len(instances[2].tags) == 3


def test_nullable_fields():
    """Test detection of nullable/optional fields."""
    df = pl.DataFrame({
        "required": [1, 2, 3],
        "optional": [1, None, 3]
    })
    
    # Use force_optional to ensure None values are handled
    Model = infer_pydantic_model(df, model_name="NullableModel", force_optional=True)
    
    # Convert the DataFrame to instances to verify nulls are handled
    instances = list(df_to_pydantic(df, model=Model))
    # Check that we can process rows with None values
    assert instances[1].required == 2
    # The optional field with None should be present
    assert instances[1].optional is None


def test_force_optional_flag():
    """Test force_optional=True makes all fields optional."""
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })
    
    Model = infer_pydantic_model(df, model_name="ForceOptionalModel", force_optional=True)
    
    # Should be able to create instance with None values
    instance = Model(id=None, name=None)
    assert instance.id is None
    assert instance.name is None


def test_df_to_pydantic_conversion():
    """Test converting DataFrame rows to Pydantic instances."""
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [95.5, 88.0, 92.3]
    })
    
    instances = list(df_to_pydantic(df, model_name="StudentModel"))
    
    assert len(instances) == 3
    assert instances[0].id == 1
    assert instances[0].name == "Alice"
    assert instances[1].score == 88.0
    assert instances[2].name == "Charlie"


def test_model_name_validation():
    """Test that invalid model names raise ValueError."""
    df = pl.DataFrame({"id": [1, 2, 3]})
    
    # Invalid: starts with number
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model(df, model_name="123Model")
    
    # Invalid: contains spaces
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model(df, model_name="My Model")
    
    # Invalid: contains dashes
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model(df, model_name="My-Model")


def test_multiple_rows_varying_nullability():
    """Test handling of fields with mixed null/non-null values."""
    df = pl.DataFrame({
        "always_present": [1, 2, 3, 4],
        "sometimes_null": [1, None, 3, None],
        "mostly_null": [None, None, 1, None]
    })
    
    Model = infer_pydantic_model(df, model_name="MixedNullModel", force_optional=True)
    instances = list(df_to_pydantic(df, model=Model))
    
    assert instances[0].always_present == 1
    # With force_optional=True, None values should be handled
    assert instances[1].sometimes_null is None
    assert instances[2].mostly_null == 1


def test_pre_provided_model():
    """Test using a pre-provided model class instead of inferring."""
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })
    
    # First infer a model
    PreModel = infer_pydantic_model(df, model_name="PreModel")
    
    # Use it to convert another DataFrame
    df2 = pl.DataFrame({
        "id": [4, 5],
        "name": ["David", "Eve"]
    })
    
    instances = list(df_to_pydantic(df2, model=PreModel))
    
    assert len(instances) == 2
    assert instances[0].id == 4
    assert instances[1].name == "Eve"

