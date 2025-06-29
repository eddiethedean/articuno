# Articuno

Convert Polars DataFrames into Pydantic models easily, with automatic schema inference including nested structs, nullable fields, and Pydantic class code generation.

---

## Installation

```bash
pip install articuno
```

## Basic Usage

```python
import articuno
import polars as pl

df = pl.DataFrame({
    "name": ["Alice", "Bob"],
    "age": [30, 25],
    "active": [True, False]
})

AutoModel = articuno.infer_pydantic_model(df)
models = articuno.df_to_pydantic(df, AutoModel)

for model in models:
    print(model)
```

## Handling Nested Structs

```python
df = pl.DataFrame({
    "user": [
        {"name": "Alice", "address": {"city": "NY", "zip": 10001}},
        {"name": "Bob", "address": {"city": "LA", "zip": 90001}},
    ]
})

Model = articuno.infer_pydantic_model(df)
instances = articuno.df_to_pydantic(df, Model)

print(instances[0].user.name)          # Alice
print(instances[1].user.address.zip)  # 90001
```

## Nullable Fields

```python
df = pl.DataFrame({
    "name": ["Alice", None],
    "age": [30, None]
}).with_columns([
    pl.col("name").cast(pl.Utf8).set_nullable(True),
    pl.col("age").cast(pl.Int32).set_nullable(True)
])

Model = articuno.infer_pydantic_model(df)
instances = articuno.df_to_pydantic(df, Model)

print(instances[0].name)  # Alice
print(instances[1].name)  # None
print(instances[1].age)   # None
```

## Using a Custom Pydantic Model

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

df = pl.DataFrame({
    "name": ["Alice", "Bob"],
    "age": [30, 25],
})

people = df_to_pydantic(df, Person)
```

## Supported Polars Types

- Numeric types: `Int8`–`Int64`, `UInt8–UInt64`, `Float32`, `Float64`
- String: `Utf8`
- Boolean
- Date and time: `Date`, `Datetime`, `Time`, `Duration`
- Complex types: `List`, `Struct`
- Other: `Decimal`, `Binary`, `Categorical`, `Enum`, `Null`


## Generate Pydantic Class Code

### Example Usage

```python
from pydantic import create_model
import articuno

# Create a dynamic model
DynamicUser = create_model('DynamicUser', name=(str, ...), age=(int, 0))

# Generate the class code
code = articuno.generate_pydantic_class_code(DynamicUser)

print(code)
```

### Output

```python
from __future__ import annotations
from pydantic import BaseModel

class DynamicUser(BaseModel):
    name: str
    age: int = 0
```

### Saving to a File
To write the generated code to a Python file:
```python
articuno.generate_pydantic_class_code(DynamicUser, output_path="user_model.py")
```

### Custom Class Name
To override the class name in the output (useful for renaming dynamic models):
```python
articuno.generate_pydantic_class_code(DynamicUser, model_name="User")
```


