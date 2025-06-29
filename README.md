# ❄️ Articuno ❄️

Convert Polars DataFrames to Pydantic models — and optionally generate clean Python code from them.

> A blazing-fast tool for schema inference, data validation, and model generation powered by [Polars](https://pola.rs/) and [Pydantic](https://docs.pydantic.dev/).

---

## 🚀 Features

- 🔍 **Infer Pydantic models** directly from `polars.DataFrame` schemas
- 🧪 **Validate data** by converting DataFrame rows to Pydantic instances
- 🧱 **Supports nested Structs**, Lists, Nullable fields, and advanced types
- 🧬 **Generate Python model code** from dynamic models using [datamodel-code-generator](https://pypi.org/project/datamodel-code-generator/)

---

## 📦 Installation

```bash
pip install articuno
```

---

## 🛠 Usage

### 1. Convert a DataFrame to Pydantic Models

```python
import polars as pl
from articuno import df_to_pydantic

df = pl.DataFrame({
    "name": ["Alice", "Bob"],
    "age": [30, 25],
    "is_active": [True, False],
})

models = df_to_pydantic(df)

print(models[0])
print(models[0].dict())
```

**Output:**
```
name='Alice' age=30 is_active=True
{'name': 'Alice', 'age': 30, 'is_active': True}
```

---

### 2. Infer a Model Only

```python
from articuno import infer_pydantic_model

model = infer_pydantic_model(df, model_name="UserModel")
print(model.schema_json(indent=2))
```

**Output (snippet):**
```json
{
  "title": "UserModel",
  "type": "object",
  "properties": {
    "name": { "title": "Name", "type": "string" },
    "age": { "title": "Age", "type": "integer" },
    "is_active": { "title": "Is Active", "type": "boolean" }
  },
  "required": ["name", "age", "is_active"]
}
```

---

### 3. Generate Python Source Code from a Model

```python
from articuno import generate_pydantic_class_code

code = generate_pydantic_class_code(model, model_name="UserModel")
print(code)
```

**Output:**
```python
from pydantic import BaseModel

class UserModel(BaseModel):
    name: str
    age: int
    is_active: bool
```

Or write it to a file:

```python
generate_pydantic_class_code(model, output_path="user_model.py")
```

---

## 🧬 Example: Nested Structs

```python
nested_df = pl.DataFrame({
    "user": pl.Series([
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ], dtype=pl.Struct([
        ("name", pl.Utf8),
        ("age", pl.Int64),
    ]))
})

models = df_to_pydantic(nested_df)
print(models[0])
print(models[0].user.name)
```

**Output:**
```
AutoModel_user_Struct(name='Alice', age=30)
Alice
```

---

## 𞧯 When to Use Articuno

- ✅ You use **Polars** and want **type-safe modeling**
- ✅ You dynamically load or transform tabular data
- ✅ You want to **generate sharable Python classes**
- ✅ You want to **validate Polars DataFrames** using Pydantic rules

---

## ⚙️ Supported Type Mappings

Polars Type | Pydantic Type
------------|---------------
`pl.Int*`, `pl.UInt*` | `int`
`pl.Float*`           | `float`
`pl.Utf8`             | `str`
`pl.Boolean`          | `bool`
`pl.Date`             | `datetime.date`
`pl.Datetime`         | `datetime.datetime`
`pl.Duration`         | `datetime.timedelta`
`pl.List`             | `List[...]`
`pl.Struct`           | Nested Pydantic model
`pl.Null`             | `Optional[...]`

---

## 🧩 Integration Ideas

- 🔐 Use for **FastAPI** or **Litestar** API schemas
- 🧼 Use in **ETL pipelines** to enforce schema contracts
- 📄 Use to **generate Pydantic models** from data exports
- 🔀 Use with `polars.read_json` / `read_parquet` to auto-model nested data

---

## 🧪 Development & Testing

```bash
git clone https://github.com/your-username/articuno
cd articuno
pip install -e ".[dev]"
pytest
```

---

## 🧙‍♂️ FastAPI Integration (Decorator + CLI Bootstrap)
Articuno makes it easy to generate response_models for your FastAPI endpoints that return polars.DataFrames — no need to manually define Pydantic models.

### 🧩 Step 1: Add the Decorator
Use the @infer_response_model decorator on your FastAPI endpoint. Provide:
- a name for the generated Pydantic model,
- an example input dict to simulate a call to your endpoint,
- an optional path to your models.py file (defaults to models.py next to the FastAPI app file).

```python
from fastapi import FastAPI
from articuno.decorator import infer_response_model
import polars as pl

app = FastAPI()

@infer_response_model(
    name="UserModel",
    example_input={"limit": 2},
    models_path="models.py"  # Optional, relative to this file by default
)
@app.get("/users")
def get_users(limit: int):
    return pl.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [30, 25],
    }).head(limit)
```

📝 The decorator doesn't change behavior at runtime — it simply registers this endpoint for the CLI to analyze later.

### ⚙️ Step 2: Run the CLI Bootstrap
After writing or modifying your endpoints, run the Articuno CLI:

```bash
articuno bootstrap app/main.py
```

This will:

1. Import and call all decorated endpoints with the given example_input
2. Infer a Pydantic model from the returned polars.DataFrame
3. Write the model to the specified models.py file
4. Update your FastAPI app:
    - Add response_model=YourModel to the route decorator
    - Import the model at the top
    - Remove the @infer_response_model(...) decorator

### 🎯 Example Result (After Bootstrapping)
Before CLI:

```python
@infer_response_model(name="UserModel", example_input={"limit": 2})
@app.get("/users")
def get_users(limit: int):
    ...
```

After CLI:
```python
from models import UserModel

@app.get("/users", response_model=UserModel)
def get_users(limit: int):
    ...
```

models.py will contain:
```python
from pydantic import BaseModel

class UserModel(BaseModel):
    name: str
    age: int
```

---

## 🛠 CLI Options

```bash
Usage: cli.py bootstrap [OPTIONS] APP_PATH

Arguments:
  APP_PATH                Path to your FastAPI app file (e.g., app/main.py)

Options:
  --models-path PATH      Optional output path for models.py (defaults to same folder as app)
  --dry-run               Preview changes without writing files
  --help                  Show this message and exit
```




---

## 📜 Patito vs Articuno

| Feature                    | **Patito**             | **Articuno**               |
|----------------------------|------------------------|----------------------------|
| Polars–Pydantic bridge     | ✅ Declarative schema  | ✅ Dynamic inference       |
| Validation constraints     | ✅ Unique, bounds       | ⚠️ Basic types, nullables |
| Nested Structs            | ❌ Not supported       | ✅ Fully recursive         |
| Code generation           | ❌                     | ✅ via datamodel-code-gen  |
| Example/mock data         | ✅ `.examples`         | ❌                        |

**[Patito](https://pypi.org/project/patito/)** is ideal for static schema validation with custom constraints and ETL pipelines.

**Articuno** excels at dynamic schema inference, nested model generation, and code export for API use cases.

---

## 🎜️ License

MIT © 2025 Odos Matthews
