# ❄️ Articuno ❄️

Convert Polars or Pandas DataFrames to Pydantic models with schema inference — and generate clean Python class code.

---

## ✨ Features

- Infer Pydantic models dynamically from Polars or Pandas DataFrames
- Supports nested structs, optional fields, and common data types
- Supports **PyArrow‑backed Pandas columns** (e.g., `int64[pyarrow]`, `string[pyarrow]`)
- Optional **force_optional** flag to make all fields optional regardless of data
- Generate clean Python model code using [datamodel‑code‑generator](https://github.com/koxudaxi/datamodel-code-generator)
- Lightweight, dependency‑flexible design

---

## 📦 Installation

Install the core package:

```bash
pip install articuno
```

Add optional dependencies as needed:

- Polars support:

  ```bash
  pip install articuno[polars]
  ```

- Pandas support (with optional PyArrow support):

  ```bash
  pip install articuno[pandas]
  ```

Or install all extras:

```bash
pip install articuno[polars,pandas]
```

---

## 🚀 Usage

### 🔍 Infer a Pydantic model from a DataFrame

```python
from articuno import df_to_pydantic
import polars as pl

df = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "score": [95.5, 88.0, 92.3]
})

models = df_to_pydantic(df)
print(models[0])
```

**Output:**

```
id=1 name='Alice' score=95.5
```

---

### 🌟 Using PyArrow‑backed Pandas columns

```python
import pandas as pd

df = pd.DataFrame({
    "id": pd.Series([1, 2, 3], dtype="int64[pyarrow]"),
    "name": pd.Series(["Alice", "Bob", "Charlie"], dtype="string[pyarrow]")
})

from articuno import infer_pydantic_model

Model = infer_pydantic_model(df, model_name="ArrowUser")
print(Model.schema_json(indent=2))
```

**Output (abbreviated):**

```json
{
  "title": "ArrowUser",
  "type": "object",
  "properties": {
    "id": { "title": "Id", "type": "integer" },
    "name": { "title": "Name", "type": "string" }
  },
  "required": ["id", "name"]
}
```

---

### 🔥 Force all fields to be optional

```python
Model = infer_pydantic_model(df, model_name="MyOptionalModel", force_optional=True)
```

---

### 🧾 Generate Python class code from a Pydantic model

```python
from articuno.codegen import generate_class_code

code = generate_class_code(Model)
print(code)
```

**Output (example):**

```python
from pydantic import BaseModel

class ArrowUser(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
```

---

## 📜 Patito vs Articuno

| Feature                       | 🦜 Patito                   | ❄️ Articuno                      |
|-------------------------------|-----------------------------|---------------------------------|
| Polars–Pydantic bridge        | ✅ Declarative schema        | ✅ Dynamic inference             |
| Validation constraints        | ✅ Unique, bounds            | ⚠️ Basic types, nullables        |
| Nested Structs                | ❌ Not supported            | ✅ Fully recursive              |
| Code generation               | ❌                          | ✅ via datamodel‑code‑gen        |
| Example/mock data             | ✅ `.examples`               | ❌                              |
| Direct Pandas/Polars support  | ❌ Indirect via dicts        | ✅ Native support with inference |

Patito is ideal for static schema validation with custom constraints and ETL pipelines.

Articuno excels at dynamic schema inference, nested model generation, and code export for API use cases.

---

## ⚙️ Supported Type Mappings

| Polars Type          | Pandas Type (incl. PyArrow)                            | Pydantic Type           |
|----------------------|--------------------------------------------------------|------------------------|
| `pl.Int*`, `pl.UInt*` | `int64`, `int32`, `Int64` (nullable int), `int64[pyarrow]`, `int32[pyarrow]` | `int` |
| `pl.Float*`           | `float64`, `float32`, `float64[pyarrow]`, `float32[pyarrow]` | `float` |
| `pl.Utf8`             | `object` (string), `string[pyarrow]`                  | `str`                  |
| `pl.Boolean`          | `bool`, `boolean`, `bool[pyarrow]`                   | `bool`                 |
| `pl.Date`             | `datetime64[ns]` (date only)                         | `datetime.date`        |
| `pl.Datetime`         | `datetime64[ns]` (timestamp)                         | `datetime.datetime`    |
| `pl.Duration`         | `timedelta64[ns]`                                    | `datetime.timedelta`   |
| `pl.List`             | `list`, `object` with lists                          | `List[...]`            |
| `pl.Struct`           | `dict`, `object` with nested dicts                   | Nested Pydantic model  |
| `pl.Null`             | `NaN`, `None` (nullable fields)                      | `Optional[...]`        |

---

## ⚡ Force Optional Mode

If you want to enforce that **all fields (top-level and nested) are optional**, use:

```python
Model = infer_pydantic_model(df, force_optional=True)
```

Or:

```python
models = df_to_pydantic(df, force_optional=True)
```

---

## 🛠️ Development

To install development dependencies:

```bash
pip install articuno[dev]
```

Run tests with:

```bash
pytest
```

---

## 🔗 Links

- [GitHub Repository](https://github.com/eddiethedean/articuno)  
- [Datamodel Code Generator](https://github.com/koxudaxi/datamodel-code-generator)  
- [Polars](https://www.pola.rs/)  
- [Pandas](https://pandas.pydata.org/)  
- [PyArrow](https://arrow.apache.org/docs/python/pandas.html)  

---

## 📄 License

MIT © Odos Matthews