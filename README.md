# 🧊 Articuno

Convert Polars DataFrames to Pydantic models — and optionally generate clean Python code from them.

> A blazing-fast tool for schema inference, data validation, and model generation powered by [Polars](https://pola.rs/), [Pydantic](https://docs.pydantic.dev/), and [datamodel-code-generator](https://github.com/koxudaxi/datamodel-code-generator).

---

## 🚀 Features

- 🔍 **Infer Pydantic models** directly from `polars.DataFrame` schemas
- 🧪 **Validate data** by converting DataFrame rows to Pydantic instances
- 🧱 **Supports nested Structs**, Lists, Nullable fields, and advanced types
- 🧬 **Generate Pydandic model code** from dynamic models using `datamodel-code-generator`

---

## 📦 Installation

```bash
pip install articuno
```

## 🛠 Usage

### 1. Convert a DataFrame to Pydantic Models

```python
import polars as pl
from articuno.convert import df_to_pydantic

df = pl.DataFrame({
    "name": ["Alice", "Bob"],
    "age": [30, 25],
    "is_active": [True, False],
})

models = df_to_pydantic(df)

# models[0] is a Pydantic model instance
print(models[0].dict())
```

### Output:
```python
name='Alice' age=30 is_active=True
{'name': 'Alice', 'age': 30, 'is_active': True}
```

### 2. Infer a Model Only

```python
from articuno.convert import infer_pydantic_model

model = infer_pydantic_model(df, model_name="UserModel")
print(model.schema_json(indent=2))
```

### Output (snippet):

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

### 3. Generate Python Source Code from a Model

```python
from articuno.codegen import generate_pydantic_class_code

code = generate_pydantic_class_code(model, model_name="UserModel")
print(code)
```

### Output:

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
print(models[0].user.name)  # "Alice"
```

### Output:

```python
AutoModel_0_Struct(name='Alice', age=30)
Alice
```

## ⚙️ Supported Type Mappings
| Polars Type           | Pydantic Type         |
| --------------------- | --------------------- |
| `pl.Int*`, `pl.UInt*` | `int`                 |
| `pl.Float*`           | `float`               |
| `pl.Utf8`             | `str`                 |
| `pl.Boolean`          | `bool`                |
| `pl.Date`             | `datetime.date`       |
| `pl.Datetime`         | `datetime.datetime`   |
| `pl.Duration`         | `datetime.timedelta`  |
| `pl.List`             | `List[...]`           |
| `pl.Struct`           | Nested Pydantic model |
| `pl.Null`             | `Optional[...]`       |

## 🧩 Integration Ideas
🔐 Use for FastAPI or Litestar API schemas

🧼 Use in ETL pipelines to enforce schema contracts

📄 Use to generate typed Python models from data exports

🔁 Use with polars.read_json / read_parquet to auto-model nested data

## 🧪 Development & Testing

```bash
git clone https://github.com/your-username/articuno
cd articuno
pip install -e ".[dev]"
pytest
```

## 🧊 Why the name "Articuno"?
Polars is named after polar bears — animals adapted to cold environments. Articuno, a legendary ice-type bird Pokémon, fits the same cold-weather theme while symbolizing elegance, power, and structure. It’s the perfect metaphor for bringing clarity and form to complex data.


## 📜 License
MIT © 2025 Odos Matthews