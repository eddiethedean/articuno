# ❄️ Articuno ❄️

Convert Polars or Pandas DataFrames to Pydantic models with schema inference — and generate clean Python class code.

---

## ✨ Features

- Infer Pydantic models dynamically from Polars or Pandas DataFrames  
- Infer Pydantic models and instances directly from **iterables of dictionaries**  
- Supports nested structs, optional fields, and common data types  
- Supports **PyArrow-backed Pandas columns** (e.g., `int64[pyarrow]`, `string[pyarrow]`)  
- Optional **force_optional** flag to make all fields optional regardless of data  
- Configurable **max_scan** parameter to limit schema inference to the first _N_ records of an iterable  
- Generate clean Python model code using [datamodel-code-generator](https://github.com/koxudaxi/datamodel-code-generator)  
- Lightweight, dependency-flexible design

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
- Full install:
  ```bash
  pip install articuno[polars,pandas]
  ```

---

## 🚀 Usage

### 🔍 DataFrame-based Inference

Infer models from Polars or Pandas DataFrames:

```python
from articuno import df_to_pydantic
import polars as pl

df = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "score": [95.5, 88.0, 92.3]
})

instances = df_to_pydantic(df, model_name="UserModel")
print(instances[0])  # id=1 name='Alice' score=95.5
```

Or just get the model class:

```python
from articuno import infer_pydantic_model
Model = infer_pydantic_model(df, model_name="UserModel")
print(Model.schema_json(indent=2))
```

---

### 🧰 Iterable-of-Dicts Inference

Infer schemas and instantiate models directly from iterables of `dict` (e.g., SQL query results, JSON records):

```python
from articuno import (
    df_to_pydantic,
    infer_pydantic_model,
    dicts_to_pydantic,
    infer_generic_model,
)

# Sample records
dicts = [
    {"id": 1, "value": "foo"},
    {"id": 2, "value": "bar"},
    # ...
]

# Convert to Pydantic instances (scans first 1000 by default)
instances = df_to_pydantic(dicts)
for obj in instances:
    print(obj)

# Get model class only with custom name/scan limit
ModelClass = infer_pydantic_model(
    dicts,
    model_name="RecModel",
    max_scan=500
)
print(ModelClass.schema_json(indent=2))

# Lazy generator of instances
for obj in dicts_to_pydantic(dicts, max_scan=200):
    print(obj)

# Generic model inference
GenericModel = infer_generic_model(dicts, model_name="GenModel")
```

---

### 🌟 PyArrow-backed Pandas Columns

```python
import pandas as pd
from articuno import infer_pydantic_model

df = pd.DataFrame({
    "id": pd.Series([1,2,3], dtype="int64[pyarrow]"),
    "name": pd.Series(["A","B","C"], dtype="string[pyarrow]")
})
Model = infer_pydantic_model(df, model_name="ArrowUser")
print(Model.schema_json(indent=2))
```

---

### 🔥 Force Optional Fields

```python
from articuno import infer_pydantic_model, df_to_pydantic

Model = infer_pydantic_model(df, force_optional=True)
models = df_to_pydantic(df, force_optional=True)
```

---

### 🧾 Generate Code

```python
from articuno.codegen import generate_class_code
code = generate_class_code(Model)
print(code)
```

---

## ⚙️ Supported Type Mappings

| Polars Type          | Pandas Type (incl. PyArrow)               | Pydantic Type        |
|----------------------|-------------------------------------------|----------------------|
| `pl.Int*`, `pl.UInt*`| `int64`, `Int64`, `int64[pyarrow]`       | `int`                |
| `pl.Float*`          | `float64`, `float64[pyarrow]`            | `float`              |
| `pl.Utf8`            | `object`, `string[pyarrow]`              | `str`                |
| `pl.Boolean`         | `bool`, `bool[pyarrow]`                  | `bool`               |
| `pl.Date`            | `datetime64[ns]`                         | `datetime.date`      |
| `pl.Datetime`        | `datetime64[ns]`                         | `datetime.datetime`  |
| `pl.Duration`        | `timedelta64[ns]`                        | `datetime.timedelta` |
| `pl.List`            | `list`                                   | `List[...]`          |
| `pl.Struct`          | `dict`                                   | Nested model         |
| `pl.Null`            | `None`, `NaN`                            | `Optional[...]`      |

---

## 🛠️ Development

```bash
pip install articuno[dev]
pytest
```

---

## 🔗 Links

- [GitHub](https://github.com/eddiethedean/articuno)  
- [Datamodel Code Generator](https://github.com/koxudaxi/datamodel-code-generator)  
- [Polars](https://pola.rs/)  
- [Pandas](https://pandas.pydata.org/)  

---

## 📄 License

MIT © Odos Matthews
