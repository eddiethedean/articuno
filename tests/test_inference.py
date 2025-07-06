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
    df = pl.DataFrame({"id": [1, 2], "name": ["A", "B"]})
    model = infer_pydantic_model(df, model_name="PolarsModel")
    instance = model(id=1, name="A")
    assert instance.id == 1
    assert instance.name == "A"


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_pandas_inference_nested():
    df = pd.DataFrame({
        "user": [{"name": "Alice"}, {"name": "Bob"}],
        "active": [True, False]
    })
    models = df_to_pydantic(df, model_name="UserModel")
    assert models[0].user["name"] == "Alice"
    assert models[1].active is False
