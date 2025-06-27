import polars as pl
from pydantic import ValidationError
from articuno import df_to_pydantic, infer_pydantic_model
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from typing import List


def test_basic_types():
    df = pl.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [30, 25],
        "active": [True, False],
        "height": [5.5, 6.1],
        "birthdate": [date(1990, 1, 1), date(1985, 7, 4)],
        "last_seen": [datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 2, 13, 0)],
        "wake_up": [time(7, 30), time(6, 45)],
        "duration": [timedelta(minutes=5), timedelta(minutes=10)],
        "score": [Decimal("12.5"), Decimal("7.3")],
        "binary_data": [b"abc", b"def"],
        "category": ["A", "B"]
    })
    model = infer_pydantic_model(df)
    instances = df_to_pydantic(df, model)
    assert instances[0].name == "Alice"
    assert isinstance(instances[0].birthdate, date)
    assert isinstance(instances[1].last_seen, datetime)
    assert isinstance(instances[0].wake_up, time)
    assert isinstance(instances[1].duration, timedelta)
    assert isinstance(instances[0].score, Decimal)
    assert instances[1].binary_data == b"def"

def test_list_type():
    df = pl.DataFrame({
        "values": [[1, 2, 3], [4, 5]]
    })
    model = infer_pydantic_model(df)
    instances = df_to_pydantic(df, model)
    assert instances[0].values == [1, 2, 3]
    assert instances[1].values == [4, 5]

def test_struct_type():
    df = pl.DataFrame({
        "info": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    })
    model = infer_pydantic_model(df)
    instances = df_to_pydantic(df, model)
    assert instances[0].info == {"a": 1, "b": 2}

def test_nested_struct_type():
    df = pl.DataFrame({
        "user": [
            {"name": "Alice", "address": {"city": "NY", "zip": 10001}},
            {"name": "Bob", "address": {"city": "LA", "zip": 90001}},
        ]
    })
    model = infer_pydantic_model(df)
    instances = df_to_pydantic(df, model)
    assert instances[0].user["name"] == "Alice"
    assert instances[1].user["address"]["zip"] == 90001

def test_validation_error():
    df = pl.DataFrame({
        "age": [30, "not an int"]
    })
    model = infer_pydantic_model(df)
    try:
        df_to_pydantic(df, model)
    except ValidationError as e:
        assert "value is not a valid integer" in str(e)
    else:
        assert False, "ValidationError was not raised"
