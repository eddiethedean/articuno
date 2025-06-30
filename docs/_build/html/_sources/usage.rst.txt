Usage Guide
===========

Articuno converts Polars DataFrames into Pydantic models. Here's a basic example:

.. code-block:: python

    import polars as pl
    from articuno import df_to_pydantic

    df = pl.DataFrame({"name": ["Alice"], "age": [30]})
    model = df_to_pydantic(df)
    print(model.model_json_schema())