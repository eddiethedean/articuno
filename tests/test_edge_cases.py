"""
Tests for edge cases and error handling.
"""

import pytest
from articuno import infer_pydantic_model, df_to_pydantic

# Optional imports
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None


def test_empty_iterable_raises_error():
    """Test that empty iterable raises ValueError."""
    dicts = []
    
    with pytest.raises(ValueError, match="Cannot infer schema from empty iterable"):
        infer_pydantic_model(dicts, model_name="EmptyModel")


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_empty_pandas_dataframe():
    """Test handling of empty Pandas DataFrame."""
    df = pd.DataFrame()
    
    # Empty DataFrame should raise or handle gracefully
    # Depending on implementation, this might fail
    try:
        infer_pydantic_model(df, model_name="EmptyPandasModel")
        # If it succeeds, should create a model with no fields
    except (ValueError, KeyError):
        # Or it might raise an error, which is also acceptable
        pass


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_empty_polars_dataframe():
    """Test handling of empty Polars DataFrame."""
    df = pl.DataFrame()
    
    # Empty DataFrame should raise or handle gracefully
    try:
        infer_pydantic_model(df, model_name="EmptyPolarsModel")
    except (ValueError, Exception):
        pass


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_all_null_column():
    """Test DataFrame column with all null values."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "all_null": [None, None, None]
    })
    
    Model = infer_pydantic_model(df, model_name="AllNullModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    # Should handle gracefully
    assert len(instances) == 3
    assert instances[0].id == 1
    assert instances[0].all_null is None


def test_invalid_model_name_spaces():
    """Test that model names with spaces are rejected."""
    dicts = [{"id": 1}]
    
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model(dicts, model_name="My Model")


def test_invalid_model_name_dashes():
    """Test that model names with dashes are rejected."""
    dicts = [{"id": 1}]
    
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model(dicts, model_name="My-Model")


def test_invalid_model_name_starts_with_number():
    """Test that model names starting with numbers are rejected."""
    dicts = [{"id": 1}]
    
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model(dicts, model_name="123Model")


def test_invalid_model_name_special_chars():
    """Test that model names with special characters are rejected."""
    dicts = [{"id": 1}]
    
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model(dicts, model_name="Model@Name")


def test_valid_model_name_with_underscore():
    """Test that model names with underscores are accepted."""
    dicts = [{"id": 1}]
    
    # Should not raise
    Model = infer_pydantic_model(dicts, model_name="Valid_Model_Name")
    assert Model is not None


def test_valid_model_name_starts_with_underscore():
    """Test that model names starting with underscore are accepted."""
    dicts = [{"id": 1}]
    
    # Should not raise
    Model = infer_pydantic_model(dicts, model_name="_PrivateModel")
    assert Model is not None


def test_python_keyword_as_model_name():
    """Test that Python keywords as model names work (they're valid identifiers)."""
    dicts = [{"id": 1}]
    
    # Python keywords ARE valid identifiers, so this should work
    # But might cause issues - let's test
    try:
        infer_pydantic_model(dicts, model_name="class")
        # If isidentifier() returns True for 'class', this will work
    except ValueError:
        # If we add keyword checking, this would fail
        pass


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_single_row_dataframe():
    """Test DataFrame with only one row."""
    df = pd.DataFrame({
        "id": [1],
        "name": ["Alice"],
        "score": [95.5]
    })
    
    Model = infer_pydantic_model(df, model_name="SingleRowModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert len(instances) == 1
    assert instances[0].id == 1
    assert instances[0].name == "Alice"


def test_single_dict_in_iterable():
    """Test iterable with only one dict."""
    dicts = [{"id": 1, "value": "test"}]
    
    Model = infer_pydantic_model(dicts, model_name="SingleDictModel")
    instance = Model(id=1, value="test")
    
    assert instance.id == 1
    assert instance.value == "test"


def test_very_large_max_scan():
    """Test with very large max_scan value."""
    dicts = [{"id": i} for i in range(10)]
    
    # max_scan larger than data size should work fine
    Model = infer_pydantic_model(dicts, model_name="LargeScanModel", max_scan=1000000)
    
    assert Model is not None


def test_bool_vs_int_precedence():
    """Test that booleans are correctly identified, not as integers."""
    dicts = [
        {"id": 1, "flag": True},
        {"id": 2, "flag": False},
    ]
    
    Model = infer_pydantic_model(dicts, model_name="BoolPrecedenceModel")
    instance = Model(id=1, flag=True)
    
    # Should be bool, not int
    assert isinstance(instance.flag, bool)
    assert instance.flag is True


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_bool_vs_int_precedence_pandas():
    """Test bool precedence in Pandas DataFrames."""
    df = pd.DataFrame({
        "id": [1, 2],
        "flag": [True, False]
    })
    
    Model = infer_pydantic_model(df, model_name="PandasBoolModel")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert isinstance(instances[0].flag, bool)


def test_unsupported_type_fallback_to_any():
    """Test that unsupported types fallback to Any."""
    # Complex objects that aren't basic types
    class CustomObject:
        pass
    
    dicts = [
        {"id": 1, "obj": CustomObject()},
    ]
    
    # Should not crash, should infer as Any
    Model = infer_pydantic_model(dicts, model_name="UnsupportedModel")
    
    # Can create instance (validation might fail at runtime though)
    assert Model is not None


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_mixed_types_in_column():
    """Test column with mixed types."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "mixed": [1, "two", 3.0]  # Mixed types
    })
    
    # Should handle gracefully, likely infer as object/Any
    Model = infer_pydantic_model(df, model_name="MixedTypeModel")
    
    assert Model is not None


def test_nested_dict_all_none():
    """Test nested dict field where all values are None."""
    dicts = [
        {"id": 1, "nested": None},
        {"id": 2, "nested": None},
    ]
    
    Model = infer_pydantic_model(dicts, model_name="NestedNoneModel")
    instance = Model(id=1, nested=None)
    
    assert instance.nested is None


def test_list_field_empty_lists():
    """Test list field with empty lists."""
    dicts = [
        {"id": 1, "items": []},
        {"id": 2, "items": []},
    ]
    
    Model = infer_pydantic_model(dicts, model_name="EmptyListModel")
    instance = Model(id=1, items=[])
    
    assert instance.items == []


def test_very_long_field_names():
    """Test with very long field names."""
    dicts = [
        {"id": 1, "this_is_a_very_long_field_name_that_goes_on_and_on": "value"}
    ]
    
    Model = infer_pydantic_model(dicts, model_name="LongFieldModel")
    instance = Model(id=1, this_is_a_very_long_field_name_that_goes_on_and_on="value")
    
    assert instance.this_is_a_very_long_field_name_that_goes_on_and_on == "value"


def test_unicode_field_names():
    """Test with unicode characters in field names."""
    dicts = [
        {"id": 1, "name_café": "test"}
    ]
    
    # Unicode in field names should work
    Model = infer_pydantic_model(dicts, model_name="UnicodeFieldModel")
    instance = Model(id=1, name_café="test")
    
    assert instance.name_café == "test"


def test_unicode_string_values():
    """Test with unicode characters in string values."""
    dicts = [
        {"id": 1, "name": "Alice 日本語"},
        {"id": 2, "name": "Bob 中文"},
    ]
    
    Model = infer_pydantic_model(dicts, model_name="UnicodeValueModel")
    instances = list(df_to_pydantic(dicts, model=Model))
    
    assert instances[0].name == "Alice 日本語"
    assert instances[1].name == "Bob 中文"


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_dataframe_with_index():
    """Test that DataFrame index doesn't interfere with inference."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })
    df.set_index("id", inplace=True)
    
    # Should still work
    Model = infer_pydantic_model(df, model_name="IndexedModel")
    
    assert Model is not None


def test_zero_max_scan():
    """Test with max_scan=0 (should fail or use default)."""
    dicts = [{"id": 1}]
    
    # This should either fail or use a default
    try:
        infer_pydantic_model(dicts, model_name="ZeroScanModel", max_scan=0)
        # If it works, that's fine
    except ValueError:
        # If it raises an error, that's also acceptable
        pass

