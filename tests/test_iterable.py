"""
Comprehensive tests for iterable/dict inference functionality.
"""

import pytest
from datetime import datetime, date, timedelta
from articuno import infer_pydantic_model, df_to_pydantic
from articuno.iterable_infer import dicts_to_pydantic, infer_generic_model


def test_basic_dict_inference():
    """Test basic type inference from dicts."""
    dicts = [
        {"id": 1, "name": "Alice", "score": 95.5, "active": True},
        {"id": 2, "name": "Bob", "score": 88.0, "active": False},
        {"id": 3, "name": "Charlie", "score": 92.3, "active": True},
    ]
    
    Model = infer_generic_model(dicts, model_name="BasicDictModel")
    instance = Model(id=1, name="Alice", score=95.5, active=True)
    
    assert instance.id == 1
    assert instance.name == "Alice"
    assert instance.score == 95.5
    assert instance.active is True


def test_datetime_support():
    """Test datetime type inference from dicts."""
    dicts = [
        {"id": 1, "created": datetime(2023, 1, 1, 12, 0)},
        {"id": 2, "created": datetime(2023, 1, 2, 13, 30)},
    ]
    
    Model = infer_generic_model(dicts, model_name="DatetimeModel")
    instance = Model(id=1, created=datetime(2023, 1, 1, 12, 0))
    
    assert isinstance(instance.created, datetime)
    assert instance.created == datetime(2023, 1, 1, 12, 0)


def test_date_support():
    """Test date type inference from dicts."""
    dicts = [
        {"id": 1, "birth_date": date(2000, 1, 1)},
        {"id": 2, "birth_date": date(2000, 1, 2)},
    ]
    
    Model = infer_generic_model(dicts, model_name="DateModel")
    instance = Model(id=1, birth_date=date(2000, 1, 1))
    
    assert isinstance(instance.birth_date, date)
    assert instance.birth_date == date(2000, 1, 1)


def test_timedelta_support():
    """Test timedelta type inference from dicts."""
    dicts = [
        {"id": 1, "duration": timedelta(days=1)},
        {"id": 2, "duration": timedelta(hours=2)},
    ]
    
    Model = infer_generic_model(dicts, model_name="TimedeltaModel")
    instance = Model(id=1, duration=timedelta(days=1))
    
    assert isinstance(instance.duration, timedelta)
    assert instance.duration == timedelta(days=1)


def test_nested_dict_inference():
    """Test nested dict inference."""
    dicts = [
        {"id": 1, "user": {"name": "Alice", "age": 30}},
        {"id": 2, "user": {"name": "Bob", "age": 25}},
    ]
    
    Model = infer_generic_model(dicts, model_name="NestedDictModel")
    instance = Model(id=1, user={"name": "Alice", "age": 30})
    
    assert instance.id == 1
    assert instance.user.name == "Alice"
    assert instance.user.age == 30


def test_list_fields():
    """Test list field inference."""
    dicts = [
        {"id": 1, "tags": ["a", "b", "c"]},
        {"id": 2, "tags": ["d", "e"]},
    ]
    
    Model = infer_generic_model(dicts, model_name="ListFieldModel")
    instance = Model(id=1, tags=["a", "b", "c"])
    
    assert instance.tags == ["a", "b", "c"]


def test_max_scan_parameter():
    """Test that max_scan limits schema inference."""
    # Create 100 dicts, but first 5 have different structure
    dicts = [{"id": i, "name": f"User{i}"} for i in range(5)]
    dicts.extend([{"id": i, "name": f"User{i}", "extra": "field"} for i in range(5, 100)])
    
    # Scan only first 3
    Model = infer_generic_model(dicts, model_name="ScanLimitModel", scan_limit=3)
    schema = Model.model_json_schema() if hasattr(Model, 'model_json_schema') else Model.schema()
    
    # Should not have 'extra' field since we only scanned first 3
    properties = schema.get('properties', {})
    assert 'id' in properties
    assert 'name' in properties


def test_sparse_missing_keys():
    """Test handling of missing keys across records."""
    dicts = [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob"},  # Missing email
        {"id": 3, "email": "charlie@example.com"},  # Missing name
    ]
    
    Model = infer_generic_model(dicts, model_name="SparseKeyModel")
    
    # All fields should be optional due to missing keys
    instance1 = Model(id=1, name="Alice", email="alice@example.com")
    instance2 = Model(id=2, name="Bob", email=None)
    instance3 = Model(id=3, name=None, email="charlie@example.com")
    
    assert instance1.email == "alice@example.com"
    assert instance2.email is None
    assert instance3.name is None


def test_all_null_field():
    """Test field that is always None."""
    dicts = [
        {"id": 1, "data": None},
        {"id": 2, "data": None},
        {"id": 3, "data": None},
    ]
    
    Model = infer_generic_model(dicts, model_name="AllNullModel")
    instance = Model(id=1, data=None)
    
    assert instance.data is None


def test_force_optional_flag():
    """Test force_optional=True makes all fields optional."""
    dicts = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    
    Model = infer_generic_model(dicts, model_name="ForceOptModel", force_optional=True)
    
    # Should be able to create with None values
    instance = Model(id=None, name=None)
    assert instance.id is None
    assert instance.name is None


def test_dicts_to_pydantic_generator():
    """Test that dicts_to_pydantic returns a generator."""
    dicts = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"},
    ]
    
    result = dicts_to_pydantic(dicts, model_name="GenModel")
    
    # Should be a generator
    instances = list(result)
    assert len(instances) == 3
    assert instances[0].id == 1
    assert instances[1].name == "Bob"


def test_pre_provided_model():
    """Test using a pre-provided model class."""
    dicts1 = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    
    # Infer model from first set
    PreModel = infer_generic_model(dicts1, model_name="PreModel")
    
    # Use it with another set
    dicts2 = [
        {"id": 3, "name": "Charlie"},
        {"id": 4, "name": "David"},
    ]
    
    instances = list(dicts_to_pydantic(dicts2, model=PreModel))
    
    assert len(instances) == 2
    assert instances[0].id == 3
    assert instances[1].name == "David"


def test_model_name_validation():
    """Test that invalid model names raise ValueError."""
    dicts = [{"id": 1}]
    
    # Invalid: starts with number
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_generic_model(dicts, model_name="123Model")
    
    # Invalid: contains spaces
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_generic_model(dicts, model_name="My Model")
    
    # Invalid: contains dashes
    with pytest.raises(ValueError, match="Invalid model name"):
        infer_generic_model(dicts, model_name="My-Model")


def test_empty_iterable_error():
    """Test that empty iterable raises ValueError."""
    dicts = []
    
    with pytest.raises(ValueError, match="Cannot infer schema from empty iterable"):
        infer_generic_model(dicts, model_name="EmptyModel")


def test_bool_vs_int_precedence():
    """Test that bool is correctly identified before int."""
    dicts = [
        {"id": 1, "flag": True},
        {"id": 2, "flag": False},
    ]
    
    Model = infer_generic_model(dicts, model_name="BoolModel")
    instance = Model(id=1, flag=True)
    
    # Should be bool, not int
    assert instance.flag is True
    assert isinstance(instance.flag, bool)


def test_unified_interface_with_dicts():
    """Test using df_to_pydantic with dict iterable."""
    dicts = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    
    instances = list(df_to_pydantic(dicts, model_name="UnifiedModel"))
    
    assert len(instances) == 2
    assert instances[0].id == 1
    assert instances[1].name == "Bob"


def test_infer_pydantic_model_with_dicts():
    """Test using infer_pydantic_model with dict iterable."""
    dicts = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    
    Model = infer_pydantic_model(dicts, model_name="InferredModel")
    instance = Model(id=1, name="Alice")
    
    assert instance.id == 1
    assert instance.name == "Alice"


def test_generator_input():
    """Test that generator input works correctly."""
    def dict_generator():
        for i in range(3):
            yield {"id": i, "value": f"item_{i}"}
    
    instances = list(dicts_to_pydantic(dict_generator(), model_name="GenInputModel"))
    
    assert len(instances) == 3
    assert instances[0].id == 0
    assert instances[2].value == "item_2"


def test_scan_limit_with_generator():
    """Test scan_limit parameter with generator input."""
    def large_generator():
        for i in range(1000):
            if i < 5:
                yield {"id": i, "type": "A"}
            else:
                yield {"id": i, "type": "B", "extra": "field"}
    
    # Only scan first 3 records
    Model = infer_generic_model(large_generator(), model_name="LimitModel", scan_limit=3)
    schema = Model.model_json_schema() if hasattr(Model, 'model_json_schema') else Model.schema()
    
    properties = schema.get('properties', {})
    assert 'id' in properties
    assert 'type' in properties

