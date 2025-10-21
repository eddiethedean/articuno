"""
Comprehensive tests for Pandas DataFrame inference with PyArrow support.
"""

import pytest
from datetime import datetime, date, timedelta
from articuno import infer_pydantic_model, df_to_pydantic

# Optional imports
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


pytestmark = pytest.mark.skipif(pd is None, reason="pandas not installed")


def test_basic_numpy_dtypes():
    """Test inference of basic numpy dtypes."""
    df = pd.DataFrame({
        "int_col": pd.Series([1, 2, 3], dtype="int64"),
        "float_col": pd.Series([1.1, 2.2, 3.3], dtype="float64"),
        "str_col": pd.Series(["a", "b", "c"], dtype="object"),
        "bool_col": pd.Series([True, False, True], dtype="bool")
    })
    
    Model = infer_pydantic_model(df, model_name="BasicNumpyModel")
    instance = Model(int_col=1, float_col=1.1, str_col="a", bool_col=True)
    
    assert instance.int_col == 1
    assert instance.float_col == 1.1
    assert instance.str_col == "a"
    assert instance.bool_col is True


def test_nullable_integer_types():
    """Test pandas nullable integer types (Int64, Int32, etc.)."""
    df = pd.DataFrame({
        "int64_col": pd.Series([1, 2, None], dtype="Int64"),
        "int32_col": pd.Series([10, None, 30], dtype="Int32")
    })
    
    Model = infer_pydantic_model(df, model_name="NullableIntModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert instances[0].int64_col == 1
    assert instances[1].int32_col is None
    assert instances[2].int64_col is None


def test_nested_dict_columns():
    """Test nested dict/struct-like columns."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "user": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35}
        ]
    })
    
    Model = infer_pydantic_model(df, model_name="NestedDictModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert instances[0].id == 1
    assert instances[0].user.name == "Alice"
    assert instances[0].user.age == 30
    assert instances[1].user.name == "Bob"


def test_list_columns():
    """Test list column inference."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "tags": [["a", "b"], ["c"], ["d", "e", "f"]]
    })
    
    Model = infer_pydantic_model(df, model_name="ListColumnModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert instances[0].tags == ["a", "b"]
    assert instances[1].tags == ["c"]
    assert len(instances[2].tags) == 3


def test_datetime64_columns():
    """Test datetime64 column inference."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    })
    
    Model = infer_pydantic_model(df, model_name="DatetimeModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert isinstance(instances[0].timestamp, datetime)


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
def test_pyarrow_int64():
    """Test PyArrow-backed int64 columns."""
    df = pd.DataFrame({
        "id": pd.Series([1, 2, 3], dtype="int64[pyarrow]")
    })
    
    Model = infer_pydantic_model(df, model_name="PyArrowIntModel")
    instance = Model(id=1)
    
    assert instance.id == 1


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
def test_pyarrow_string():
    """Test PyArrow-backed string columns."""
    df = pd.DataFrame({
        "name": pd.Series(["Alice", "Bob", "Charlie"], dtype="string[pyarrow]")
    })
    
    Model = infer_pydantic_model(df, model_name="PyArrowStringModel")
    instance = Model(name="Alice")
    
    assert instance.name == "Alice"


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
def test_pyarrow_bool():
    """Test PyArrow-backed boolean columns."""
    df = pd.DataFrame({
        "active": pd.Series([True, False, True], dtype="bool[pyarrow]")
    })
    
    Model = infer_pydantic_model(df, model_name="PyArrowBoolModel")
    instance = Model(active=True)
    
    assert instance.active is True


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
def test_pyarrow_timestamp():
    """Test PyArrow timestamp types."""
    df = pd.DataFrame({
        "created": pd.Series(
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            dtype=pd.ArrowDtype(pa.timestamp("ms"))
        )
    })
    
    Model = infer_pydantic_model(df, model_name="PyArrowTimestampModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert isinstance(instances[0].created, datetime)


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
def test_pyarrow_date():
    """Test PyArrow date types."""
    df = pd.DataFrame({
        "birth_date": pd.Series(
            [date(2000, 1, 1), date(2000, 1, 2)],
            dtype=pd.ArrowDtype(pa.date32())
        )
    })
    
    Model = infer_pydantic_model(df, model_name="PyArrowDateModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert isinstance(instances[0].birth_date, date)


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
def test_pyarrow_duration():
    """Test PyArrow duration types."""
    df = pd.DataFrame({
        "duration": pd.Series(
            [timedelta(days=1), timedelta(hours=2)],
            dtype=pd.ArrowDtype(pa.duration("ms"))
        )
    })
    
    Model = infer_pydantic_model(df, model_name="PyArrowDurationModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert isinstance(instances[0].duration, timedelta)


def test_mixed_null_non_null():
    """Test columns with mixed null and non-null values."""
    df = pd.DataFrame({
        "required": [1, 2, 3, 4],
        "optional": [1.0, None, 3.0, None]
    })
    
    Model = infer_pydantic_model(df, model_name="MixedNullModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert instances[0].optional == 1.0
    # Pandas may return nan instead of None
    import math
    assert instances[1].optional is None or (isinstance(instances[1].optional, float) and math.isnan(instances[1].optional))


def test_force_optional_flag():
    """Test force_optional=True makes all fields optional."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })
    
    Model = infer_pydantic_model(df, model_name="ForceOptionalModel", force_optional=True)
    
    # Should be able to create instance with None values
    instance = Model(id=None, name=None)
    assert instance.id is None
    assert instance.name is None


def test_empty_samples_defensive_check():
    """Test that empty dict columns are handled gracefully."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "data": [None, None, None]  # All None values
    })
    
    Model = infer_pydantic_model(df, model_name="EmptySamplesModel")
    # Should not crash, field should be Any or optional
    instances = list(df_to_pydantic(df, model=Model))
    assert len(instances) == 3


def test_model_name_validation():
    """Test that invalid model names raise ValueError."""
    df = pd.DataFrame({"id": [1, 2, 3]})
    
    # Invalid: starts with number
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model(df, model_name="123Model")
    
    # Invalid: contains spaces
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model(df, model_name="My Model")
    
    # Invalid: contains dashes
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model(df, model_name="My-Model")


def test_df_to_pydantic_generator():
    """Test that df_to_pydantic returns a generator."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })
    
    result = df_to_pydantic(df, model_name="GeneratorTest")
    
    # Should be a generator
    instances = list(result)
    assert len(instances) == 3
    assert instances[0].id == 1


def test_pre_provided_model():
    """Test using a pre-provided model class."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })
    
    # First infer a model
    PreModel = infer_pydantic_model(df, model_name="PreModel")
    
    # Use it to convert another DataFrame
    df2 = pd.DataFrame({
        "id": [4, 5],
        "name": ["David", "Eve"]
    })
    
    instances = list(df_to_pydantic(df2, model=PreModel))
    
    assert len(instances) == 2
    assert instances[0].id == 4
    assert instances[1].name == "Eve"


def test_nested_dict_with_sparse_keys():
    """Test nested dicts where not all dicts have the same keys."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "metadata": [
            {"name": "Alice", "age": 30},
            {"name": "Bob"},  # Missing 'age'
            {"name": "Charlie", "age": 35, "city": "NYC"}  # Extra 'city'
        ]
    })
    
    Model = infer_pydantic_model(df, model_name="SparseKeysModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert instances[0].metadata.name == "Alice"
    assert instances[1].metadata.name == "Bob"

