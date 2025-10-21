# ❄️ Articuno ❄️

Convert Polars or Pandas DataFrames to Pydantic models with schema inference — and generate clean Python class code.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Type Checked](https://img.shields.io/badge/type--checked-mypy-blue.svg)](http://mypy-lang.org/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-blue.svg)](https://github.com/astral-sh/ruff)
[![Test Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen.svg)](https://github.com/eddiethedean/articuno)

---

## ✨ Features

**Core Functionality:**
- 🔍 Infer Pydantic models dynamically from **Polars** or **Pandas** DataFrames
- 📋 Infer models directly from **iterables of dictionaries** (SQL results, JSON records, etc.)
- 🎯 **Automatic type detection** for basic types, nested structures, and temporal data
- 🔄 **Generator-based** for memory-efficient processing of large datasets
- 🎨 Generate clean Python model code using [datamodel-code-generator](https://github.com/koxudaxi/datamodel-code-generator)

**Advanced Features:**
- ⚡ **PyArrow support** for high-performance Pandas columns (`int64[pyarrow]`, `string[pyarrow]`, `timestamp[pyarrow]`, etc.)
- 📅 **Full temporal type support**: `datetime`, `date`, `timedelta` across all backends
- 🗂️ **Nested structures**: Supports nested dicts, lists, and complex hierarchies
- 🔧 **Optional field detection**: Automatically identifies nullable fields
- 🎛️ **Configurable scanning**: `max_scan` parameter to limit schema inference
- 🔒 **Force optional mode**: Make all fields optional regardless of data
- ✅ **Model name validation**: Ensures valid Python identifiers
- 🧪 **Comprehensively tested**: 112 tests, 87% code coverage

**Design:**
- 🪶 Lightweight, dependency-flexible architecture
- 🔌 Optional dependencies for Polars, Pandas, and PyArrow
- 🎯 Type-checked with mypy
- 📏 Linted with ruff

---

## 📦 Installation

Install the core package:

```bash
pip install articuno
```

Add optional dependencies as needed:

```bash
# For Polars support
pip install articuno[polars]

# For Pandas support (with PyArrow)
pip install articuno[pandas]

# Full install with all backends
pip install articuno[polars,pandas]

# Development dependencies (includes pytest, mypy, ruff)
pip install articuno[dev]
```

---

## 🚀 Quick Start

### DataFrame to Pydantic Models

```python
from articuno import df_to_pydantic, infer_pydantic_model
import polars as pl

# Create a DataFrame
df = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "score": [95.5, 88.0, 92.3],
    "active": [True, False, True]
})

# Convert to Pydantic instances (returns a generator)
instances = list(df_to_pydantic(df, model_name="UserModel"))
print(instances[0])
# Output: id=1 name='Alice' score=95.5 active=True

# Or just get the model class
Model = infer_pydantic_model(df, model_name="UserModel")
print(Model.model_json_schema())
```

### Dict Iterables to Pydantic

Perfect for SQL query results, API responses, or JSON data:

```python
from articuno import df_to_pydantic, infer_pydantic_model

# From database results, API responses, etc.
records = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
]

# Automatically infer and create instances
instances = list(df_to_pydantic(records, model_name="User"))

# Or infer just the model
Model = infer_pydantic_model(records, model_name="User")
```

---

## 📓 Example Notebooks

Comprehensive Jupyter notebooks demonstrating all features:

### Core Examples
- **[01_quick_start.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/01_quick_start.ipynb)** - Basic usage with Polars, Pandas, and dict iterables
- **[02_temporal_types.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/02_temporal_types.ipynb)** - Working with datetime, date, and timedelta
- **[03_pyarrow_support.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/03_pyarrow_support.ipynb)** - PyArrow-backed Pandas columns
- **[04_nested_structures.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/04_nested_structures.ipynb)** - Complex nested dictionaries and lists

### Advanced Examples
- **[05_force_optional_and_max_scan.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/05_force_optional_and_max_scan.ipynb)** - Control optional fields and scanning
- **[06_code_generation.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/06_code_generation.ipynb)** - Generate Python code from models
- **[07_api_responses.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/07_api_responses.ipynb)** - Process REST API responses
- **[08_database_results.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/08_database_results.ipynb)** - SQL query results to Pydantic
- **[09_advanced_features.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/09_advanced_features.ipynb)** - Generators, type precedence, unicode

### Legacy Examples
- **[polars_inference.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/polars_inference.ipynb)** - Polars-specific inference
- **[articuno_pandas_pyarrow_example.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/articuno_pandas_pyarrow_example.ipynb)** - Pandas with PyArrow
- **[pandas_nested.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/pandas_nested.ipynb)** - Nested Pandas structures
- **[optional_nested_example.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/optional_nested_example.ipynb)** - Optional nested fields
- **[articuno_inference_demo.ipynb](https://github.com/eddiethedean/articuno/blob/main/examples/articuno_inference_demo.ipynb)** - General inference demo

> **Note:** All example notebooks have been executed and saved with outputs, so you can view the results directly on GitHub without running them.

---

## 📚 Advanced Usage

### Temporal Types Support

Articuno fully supports datetime, date, and timedelta types:

```python
from datetime import datetime, date, timedelta
from articuno import infer_pydantic_model

data = [
    {
        "event_id": 1,
        "event_date": date(2024, 1, 15),
        "timestamp": datetime(2024, 1, 15, 10, 30),
        "duration": timedelta(hours=2, minutes=30)
    }
]

Model = infer_pydantic_model(data, model_name="Event")
# Fields will have correct datetime.date, datetime.datetime, datetime.timedelta types
```

### PyArrow-Backed Pandas Columns

Full support for high-performance PyArrow dtypes:

```python
import pandas as pd
from articuno import infer_pydantic_model

df = pd.DataFrame({
    "id": pd.Series([1, 2, 3], dtype="int64[pyarrow]"),
    "name": pd.Series(["Alice", "Bob", "Charlie"], dtype="string[pyarrow]"),
    "created": pd.Series([
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 3)
    ], dtype=pd.ArrowDtype(pa.timestamp("ms"))),
    "active": pd.Series([True, False, True], dtype="bool[pyarrow]")
})

Model = infer_pydantic_model(df, model_name="ArrowModel")
```

Supported PyArrow types:
- `int64[pyarrow]`, `int32[pyarrow]`, etc.
- `string[pyarrow]`
- `bool[pyarrow]`
- `timestamp[pyarrow]` → `datetime.datetime`
- `date32[pyarrow]`, `date64[pyarrow]` → `datetime.date`
- `duration[pyarrow]` → `datetime.timedelta`

### Nested Structures

Handle complex nested data with ease:

```python
data = [
    {
        "user_id": 1,
        "profile": {
            "name": "Alice",
            "age": 30,
            "preferences": {
                "theme": "dark",
                "notifications": True
            }
        },
        "tags": ["python", "data-science"]
    }
]

Model = infer_pydantic_model(data, model_name="UserProfile")
# Nested dicts become nested Pydantic models
# Lists are preserved with List[...] typing
```

### Force Optional Fields

Make all fields optional regardless of the data:

```python
from articuno import infer_pydantic_model

df = pl.DataFrame({
    "required": [1, 2, 3],
    "also_required": ["a", "b", "c"]
})

# Force all fields to be Optional
Model = infer_pydantic_model(df, force_optional=True)

# Now you can create instances with None values
instance = Model(required=None, also_required=None)
```

### Limit Schema Scanning

For large datasets, limit how many records are scanned:

```python
# Only scan first 100 records for schema inference
Model = infer_pydantic_model(
    large_dataset,
    model_name="LargeModel",
    max_scan=100
)
```

### Memory-Efficient Processing

`df_to_pydantic` returns a generator for memory efficiency:

```python
# Generator - memory efficient for large datasets
instances_gen = df_to_pydantic(large_df, model_name="Record")

# Process one at a time
for instance in instances_gen:
    process(instance)

# Or collect all at once if needed
instances_list = list(df_to_pydantic(df, model_name="Record"))
```

### Code Generation

Generate clean Python code for your models:

```python
from articuno import infer_pydantic_model
from articuno.codegen import generate_class_code

# Infer model from data
Model = infer_pydantic_model(data, model_name="User")

# Generate Python code
code = generate_class_code(Model)
print(code)

# Or save to file
code = generate_class_code(Model, output_path="models.py")
```

### Pre-defined Models

Use a pre-defined model instead of inferring:

```python
from pydantic import BaseModel
from articuno import df_to_pydantic

class UserModel(BaseModel):
    id: int
    name: str
    email: str

# Use your existing model
instances = list(df_to_pydantic(df, model=UserModel))
```

---

## ⚙️ Supported Type Mappings

| Polars Type          | Pandas Type (incl. PyArrow)                      | Dict/Iterable     | Pydantic Type        |
|----------------------|--------------------------------------------------|-------------------|----------------------|
| `pl.Int*`, `pl.UInt*`| `int64`, `Int64`, `int64[pyarrow]`              | `int`             | `int`                |
| `pl.Float*`          | `float64`, `float64[pyarrow]`                   | `float`           | `float`              |
| `pl.Utf8`, `pl.String`| `object`, `string[pyarrow]`                     | `str`             | `str`                |
| `pl.Boolean`         | `bool`, `bool[pyarrow]`                         | `bool`            | `bool`               |
| `pl.Date`            | `datetime64[ns]`, `date[pyarrow]`               | `date`            | `datetime.date`      |
| `pl.Datetime`        | `datetime64[ns]`, `timestamp[pyarrow]`          | `datetime`        | `datetime.datetime`  |
| `pl.Duration`        | `timedelta64[ns]`, `duration[pyarrow]`          | `timedelta`       | `datetime.timedelta` |
| `pl.List`            | `list`                                           | `list`            | `List[...]`          |
| `pl.Struct`          | `dict` (nested)                                  | `dict` (nested)   | Nested `BaseModel`   |
| `pl.Null`            | `None`, `NaN`                                    | `None`            | `Optional[...]`      |

---

## 🎯 Real-World Examples

### API Response Processing

```python
# Process API responses
api_data = [
    {
        "status": "success",
        "data": {
            "user_id": 123,
            "username": "alice",
            "created_at": datetime(2024, 1, 15, 10, 30)
        }
    }
]

APIResponse = infer_pydantic_model(api_data, model_name="APIResponse")
instances = list(df_to_pydantic(api_data, model=APIResponse))
```

### SQL Query Results

```python
import sqlite3
from articuno import infer_pydantic_model, df_to_pydantic

# Get results from database
conn = sqlite3.connect("database.db")
conn.row_factory = sqlite3.Row
cursor = conn.execute("SELECT * FROM users")
rows = [dict(row) for row in cursor.fetchall()]

# Convert to Pydantic
UserModel = infer_pydantic_model(rows, model_name="User")
users = list(df_to_pydantic(rows, model=UserModel))
```

### E-commerce Order Processing

```python
orders = [
    {
        "order_id": 1001,
        "customer": {"id": 501, "name": "Alice", "email": "alice@example.com"},
        "items": [
            {"product": "Laptop", "quantity": 1, "price": 999.99},
            {"product": "Mouse", "quantity": 2, "price": 29.99}
        ],
        "total": 1059.97,
        "created_at": datetime(2024, 1, 15, 10, 30)
    }
]

Order = infer_pydantic_model(orders, model_name="Order")
```

---

## 🧪 Testing & Quality

Articuno is thoroughly tested and type-checked:

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=articuno --cov-report=term-missing

# Type checking
mypy articuno

# Linting
ruff check .
```

**Test Statistics:**
- 112 comprehensive tests
- 87% code coverage
- All tests passing ✅
- Type-checked with mypy ✅
- Linted with ruff ✅

---

## 🔧 Development

```bash
# Clone the repository
git clone https://github.com/eddiethedean/articuno.git
cd articuno

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy articuno

# Run linting
ruff check .
```

---

## 💡 Tips & Best Practices

1. **Use generators for large datasets**: `df_to_pydantic` returns a generator by default for memory efficiency
2. **Limit scanning for performance**: Use `max_scan` parameter when dealing with huge datasets
3. **Validate model names**: Articuno automatically validates that model names are valid Python identifiers
4. **Handle optional fields**: Use `force_optional=True` when working with sparse data
5. **Type precedence**: Articuno correctly handles bool vs int (bool is checked first)

---

## 🐛 Troubleshooting

### Import Errors

If you get import errors for `polars` or `pandas`:
```bash
pip install articuno[polars]  # or [pandas]
```

### PyArrow Issues

For PyArrow support:
```bash
pip install pyarrow
```

### Generator Indexing

`df_to_pydantic` returns a generator. Convert to list if you need indexing:
```python
instances = list(df_to_pydantic(df, model_name="Model"))
print(instances[0])  # Now you can index
```

---

## 📖 API Reference

### Main Functions

#### `infer_pydantic_model(source, model_name="AutoModel", force_optional=False, max_scan=1000)`

Infer a Pydantic model class from a DataFrame or dict iterable.

**Parameters:**
- `source`: Pandas DataFrame, Polars DataFrame, or iterable of dicts
- `model_name`: Name for the generated model (must be valid Python identifier)
- `force_optional`: Make all fields optional
- `max_scan`: Max records to scan for dict iterables

**Returns:** `Type[BaseModel]`

#### `df_to_pydantic(source, model=None, model_name=None, force_optional=False, max_scan=1000)`

Convert DataFrame or dict iterable to Pydantic instances.

**Parameters:**
- `source`: Pandas DataFrame, Polars DataFrame, or iterable of dicts
- `model`: Optional pre-defined model to use
- `model_name`: Name for inferred model if `model` is None
- `force_optional`: Make all fields optional
- `max_scan`: Max records to scan for dict iterables

**Returns:** `Generator[BaseModel, None, None]`

#### `generate_class_code(model, output_path=None, model_name=None)`

Generate Python code from a Pydantic model.

**Parameters:**
- `model`: Pydantic model class
- `output_path`: Optional file path to write code to
- `model_name`: Optional name override

**Returns:** `str` (the generated code)

---

## 🔗 Links

- [GitHub Repository](https://github.com/eddiethedean/articuno)
- [Datamodel Code Generator](https://github.com/koxudaxi/datamodel-code-generator)
- [Poldantic](https://github.com/eddiethedean/poldantic) (Polars integration)
- [Polars](https://pola.rs/)
- [Pandas](https://pandas.pydata.org/)
- [PyArrow](https://arrow.apache.org/docs/python/)

---

## 📄 License

MIT © Odos Matthews

---

## 🙏 Acknowledgments

- Built with [Pydantic](https://pydantic-docs.helpmanual.io/)
- Code generation powered by [datamodel-code-generator](https://github.com/koxudaxi/datamodel-code-generator)
- Polars support via [poldantic](https://github.com/eddiethedean/poldantic)

---

## 📝 Changelog

### v0.9.0
- ✨ Added full datetime/date/timedelta support across all backends
- ✨ Added PyArrow temporal type support (timestamp, date, duration)
- ✨ Added model name validation (ensures valid Python identifiers)
- 🐛 Fixed bool vs int type precedence
- 🐛 Fixed DataFrame vs iterable detection order
- 🐛 Fixed temporary directory cleanup in code generation
- 🐛 Added defensive checks for empty samples
- 📝 Improved documentation with generator behavior notes
- 🧪 Added comprehensive test suite (112 tests, 87% coverage)
- 🔍 Full mypy type checking
- 📏 Ruff linting compliance
- 📓 Added 9 comprehensive example notebooks with outputs
- 📚 Enhanced README with complete guide and examples
