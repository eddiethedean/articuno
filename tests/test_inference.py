"""
Tests for the unified inference interface and type detection.
"""

import pytest
from articuno import df_to_pydantic, infer_pydantic_model

# Optional imports
try:
    import polars as pl
except ImportError:
    pl = None

try:
    import pandas as pd
except ImportError:
    pd = None


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_polars_inference_simple():
    """Test basic Polars DataFrame inference."""
    df = pl.DataFrame({"id": [1, 2], "name": ["A", "B"]})
    model = infer_pydantic_model(df, model_name="PolarsModel")
    instance = model(id=1, name="A")
    assert instance.id == 1
    assert instance.name == "A"


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_pandas_inference_nested():
    """Test Pandas DataFrame with nested dict columns."""
    df = pd.DataFrame({
        "user": [{"name": "Alice"}, {"name": "Bob"}],
        "active": [True, False]
    })
    models = list(df_to_pydantic(df, model_name="UserModel"))
    assert models[0].user.name == "Alice"
    assert models[1].active is False


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_auto_detect_polars():
    """Test automatic detection of Polars DataFrames."""
    df = pl.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    
    # Should automatically detect it's a Polars DataFrame
    Model = infer_pydantic_model(df, model_name="AutoPolarsModel")
    instance = Model(id=1, value="a")
    
    assert instance.id == 1
    assert instance.value == "a"


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_auto_detect_pandas():
    """Test automatic detection of Pandas DataFrames."""
    df = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    
    # Should automatically detect it's a Pandas DataFrame
    Model = infer_pydantic_model(df, model_name="AutoPandasModel")
    instance = Model(id=1, value="a")
    
    assert instance.id == 1
    assert instance.value == "a"


def test_auto_detect_dict_iterable():
    """Test automatic detection of dict iterables."""
    dicts = [
        {"id": 1, "value": "a"},
        {"id": 2, "value": "b"},
        {"id": 3, "value": "c"},
    ]
    
    # Should automatically detect it's a dict iterable
    Model = infer_pydantic_model(dicts, model_name="AutoDictModel")
    instance = Model(id=1, value="a")
    
    assert instance.id == 1
    assert instance.value == "a"


def test_check_order_dataframe_before_iterable():
    """Test that DataFrames are checked before generic Iterable."""
    # This tests the bug fix where DataFrames were incorrectly
    # treated as dict iterables due to being iterable
    
    if pd is not None:
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        
        # Should use pandas path, not iterable path
        instances = list(df_to_pydantic(df, model_name="CheckOrderModel"))
        
        assert len(instances) == 2
        assert instances[0].id == 1
        assert instances[1].name == "B"


def test_unsupported_type_error():
    """Test that unsupported input types raise TypeError or ValueError."""
    # String is not a valid input (string is iterable so it goes through, but fails as not dict iterable)
    with pytest.raises((TypeError, ValueError, AttributeError)):
        infer_pydantic_model("not a dataframe", model_name="BadModel")
    
    # Number is not a valid input
    with pytest.raises(TypeError, match="Expected a pandas.DataFrame, polars.DataFrame, or iterable of dicts"):
        infer_pydantic_model(123, model_name="BadModel2")


def test_model_name_validation_unified():
    """Test model name validation works across all input types."""
    # Test with dict iterable
    dicts = [{"id": 1}]
    
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model(dicts, model_name="Invalid-Name")
    
    if pd is not None:
        df = pd.DataFrame({"id": [1]})
        with pytest.raises(ValueError, match="Invalid model name"):
            infer_pydantic_model(df, model_name="123Invalid")
    
    if pl is not None:
        df = pl.DataFrame({"id": [1]})
        with pytest.raises(ValueError, match="Invalid model name"):
            infer_pydantic_model(df, model_name="Invalid Name")


def test_df_to_pydantic_with_dicts():
    """Test df_to_pydantic works with dict iterables."""
    dicts = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    
    instances = list(df_to_pydantic(dicts, model_name="DictConversionModel"))
    
    assert len(instances) == 2
    assert instances[0].id == 1
    assert instances[1].name == "Bob"


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_df_to_pydantic_with_pandas():
    """Test df_to_pydantic works with Pandas DataFrames."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })
    
    instances = list(df_to_pydantic(df, model_name="PandasConversionModel"))
    
    assert len(instances) == 3
    assert instances[0].id == 1
    assert instances[2].name == "Charlie"


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_df_to_pydantic_with_polars():
    """Test df_to_pydantic works with Polars DataFrames."""
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })
    
    instances = list(df_to_pydantic(df, model_name="PolarsConversionModel"))
    
    assert len(instances) == 3
    assert instances[0].id == 1
    assert instances[2].name == "Charlie"


def test_max_scan_parameter_with_dicts():
    """Test max_scan parameter works with dict iterables."""
    # Create many dicts, but first few have different structure
    dicts = [{"id": i, "name": f"User{i}"} for i in range(5)]
    dicts.extend([{"id": i, "name": f"User{i}", "extra": "data"} for i in range(5, 100)])
    
    # Only scan first 3
    Model = infer_pydantic_model(dicts, model_name="MaxScanModel", max_scan=3)
    
    # Model should be based on first 3 records only
    instance = Model(id=1, name="User1")
    assert instance.id == 1


def test_force_optional_unified():
    """Test force_optional works across all input types."""
    dicts = [{"id": 1, "name": "Alice"}]
    
    Model = infer_pydantic_model(dicts, model_name="OptModel", force_optional=True)
    
    # Should allow None values
    instance = Model(id=None, name=None)
    assert instance.id is None
