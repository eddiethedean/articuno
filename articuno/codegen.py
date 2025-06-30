from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, create_model
import polars as pl
import json
import tempfile
from pathlib import Path
from datamodel_code_generator import InputFileType, generate

_model_cache: Dict[str, Type[BaseModel]] = {}

def infer_pydantic_model(
    df: pl.DataFrame,
    model_name: str = "AutoModel",
    _model_cache: Optional[Dict[str, Type[BaseModel]]] = None,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from a Polars DataFrame schema.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to infer from.
    model_name : str, optional
        Name of the root Pydantic model class.
    _model_cache : dict[str, Type[BaseModel]], optional
        Internal cache to reuse nested model classes (e.g., for struct fields).

    Returns
    -------
    Type[BaseModel]
        A Pydantic model class representing the inferred schema.
    """
    if _model_cache is None:
        _model_cache = {}

    def resolve_dtype(dtype: pl.DataType) -> Any:
        # Map Polars primitive types to Python types
        if dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
            return int
        if dtype in {pl.Float32, pl.Float64}:
            return float
        if dtype == pl.Boolean:
            return bool
        if dtype == pl.Utf8:
            return str
        if dtype == pl.Date:
            import datetime
            return datetime.date
        if dtype == pl.Datetime:
            import datetime
            return datetime.datetime
        if dtype == pl.Duration:
            import datetime
            return datetime.timedelta
        if dtype == pl.Null:
            return type(None)

        # Handle List types
        if dtype.__class__.__name__ == "List":
            inner_type = resolve_dtype(dtype.inner)
            from typing import List
            return List[inner_type]

        # Handle Struct types
        if dtype.__class__.__name__ == "Struct":
            struct_key = str(dtype)
            if struct_key in _model_cache:
                return _model_cache[struct_key]
            else:
                fields = {
                    field.name: (resolve_dtype(field.dtype), ...)
                    for field in dtype.fields
                }
                model_cls = create_model(f"{model_name}_{len(_model_cache)}_Struct", **fields)
                _model_cache[struct_key] = model_cls
                return model_cls

        # Fallback to Any for unknown types
        return Any

    fields: Dict[str, tuple] = {
        name: (resolve_dtype(dtype), ...)
        for name, dtype in df.schema.items()
    }

    return create_model(model_name, **fields)

def df_to_pydantic(
    df: pl.DataFrame,
    model: Optional[Type[BaseModel]] = None,
    model_name: Optional[str] = None,
) -> List[BaseModel]:
    """
    Convert a Polars DataFrame to a list of Pydantic model instances.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to convert.
    model : Type[BaseModel], optional
        An existing Pydantic model class to use for conversion.
        If None, a model will be inferred from the DataFrame.
    model_name : str, optional
        The name to use if inferring the model.

    Returns
    -------
    List[BaseModel]
        A list of Pydantic model instances corresponding to DataFrame rows.
    """
    if model is None:
        model = infer_pydantic_model(df, model_name=model_name or "AutoModel")
    return [model(**row) for row in df.to_dicts()]

def generate_pydantic_class_code(
    model: Type[BaseModel],
    output_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate Python class source code from a Pydantic model using datamodel-code-generator.

    This function converts a Pydantic model (including dynamically created ones)
    to JSON Schema, feeds it to datamodel-code-generator in-memory, and returns the
    equivalent Python class definition as a string.

    Parameters
    ----------
    model : Type[BaseModel]
        A Pydantic model class (can be dynamic).
    output_path : str or Path, optional
        If given, saves the generated class code to this file path.
    model_name : str, optional
        Optionally override the class name in the schema title before generation.

    Returns
    -------
    str
        The generated Python source code as a string.
    """
    if hasattr(model, "model_json_schema"):
        schema = model.model_json_schema()
    else:
        schema = model.schema()

    if model_name:
        schema["title"] = model_name

    schema_str = json.dumps(schema, indent=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "schema.json"
        output_file = Path(tmpdir) / "model.py"

        input_file.write_text(schema_str, encoding="utf-8")

        generate(
            input_file,
            input_file_type=InputFileType.JsonSchema,
            output=output_file,
        )

        code = output_file.read_text(encoding="utf-8")

        if output_path:
            Path(output_path).write_text(code, encoding="utf-8")

        return code
