Usage Guide
===========

This section provides examples on how to use Articuno's main functionality.

---

Inferring a Pydantic model from a Polars DataFrame
--------------------------------------------------

.. code-block:: python

   import polars as pl
   from articuno.convert import infer_pydantic_model, df_to_pydantic

   # Create a Polars DataFrame
   df = pl.DataFrame([
       {"id": 1, "name": "Alice", "score": 95.5},
       {"id": 2, "name": "Bob", "score": 87.0},
   ])

   # Infer a Pydantic model from the DataFrame schema
   Model = infer_pydantic_model(df, model_name="StudentScore")

   # Convert DataFrame rows to Pydantic model instances
   instances = df_to_pydantic(df, model=Model)

   print(instances[0])

Expected output:

.. code-block:: none

   id=1 name='Alice' score=95.5

---

Generating Python class code from a Pydantic model
--------------------------------------------------

.. code-block:: python

   from articuno.codegen import generate_pydantic_class_code

   # Generate Python source code for the inferred model
   code = generate_pydantic_class_code(Model)

   print(code)

Expected output snippet:

.. code-block:: python

   from pydantic import BaseModel

   class StudentScore(BaseModel):
       id: int
       name: str
       score: float

---

Bootstrapping FastAPI response models
-------------------------------------

Use the `@infer_response_model` decorator to mark FastAPI endpoints for automatic Pydantic model generation:

.. code-block:: python

   from fastapi import FastAPI
   from articuno.bootstrap import infer_response_model
   import polars as pl

   app = FastAPI()

   @infer_response_model(
       name="UserDataModel",
       example_input={"user_id": 123}
   )
   @app.get("/userdata")
   def get_user_data(user_id: int):
       # Return a Polars DataFrame simulating data retrieval
       return pl.DataFrame([
           {"user_id": user_id, "username": "tester", "active": True}
       ])

The Articuno CLI can then generate matching Pydantic models for your endpoints.

---

CLI Usage
---------

Run the CLI bootstrap command to generate models:

.. code-block:: console

   articuno bootstrap path/to/your/app.py

This will generate Pydantic model classes and update your source code with `response_model` annotations.
