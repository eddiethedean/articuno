from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel
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
    Infer a Pydantic model class from the schema of a Polars DataFrame.

    This function inspects the schema of the input Polars DataFrame and
    recursively maps Polars data types (including nested structs and lists)
    to Python types compatible with Pydantic models. It dynamically generates
    a Pydantic model class representing the DataFrame schema.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame whose schema is used to infer the model.
    model_name : str, optional
        The desired name for the generated Pydantic model class. Default is "AutoModel".
    _model_cache : dict[str, Type[BaseModel]], optional
        Internal cache dictionary to avoid regenerating nested models for
        identical struct schemas. Should not be set by external callers.

    Returns
    -------
    Type[BaseModel]
        A dynamically created Pydantic model class reflecting the DataFrame schema.

    Notes
    -----
    - Nested Polars Struct columns are converted into nested Pydantic models.
    - Polars List types are mapped to typing.List of the inner type.
    - Nullable columns are not explicitly marked here; use `convert.py` for nullable detection.
    """
    if _model_cache is None:
        _model_cache = {}

    def resolve_dtype(dtype: pl.DataType) -> Any:
        """
        Recursively resolve a Polars dtype to a Python type for Pydantic.

        Parameters
        ----------
        dtype : pl.DataType
            The Polars data type to resolve.

        Returns
        -------
        Any
            Corresponding Python or Pydantic-compatible type.
        """
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

        # Handle Struct types recursively
        if dtype.__class__.__name__ == "Struct":
            struct_key = str(dtype)
            if struct_key in _model_cache:
                return _model_cache[struct_key]
            else:
                fields = {
                    field.name: (resolve_dtype(field.dtype), ...)
                    for field in dtype.fields
                }
                from pydantic import create_model
                model_cls = create_model(f"{model_name}_{len(_model_cache)}_Struct", **fields)
                _model_cache[struct_key] = model_cls
                return model_cls

        # Fallback to Any for unknown types
        return Any

    fields: Dict[str, tuple] = {
        name: (resolve_dtype(dtype), ...)
        for name, dtype in df.schema.items()
    }

    from pydantic import create_model
    return create_model(model_name, **fields)

def df_to_pydantic(
    df: pl.DataFrame,
    model: Optional[Type[BaseModel]] = None,
    model_name: Optional[str] = None,
) -> List[BaseModel]:
    """
    Convert a Polars DataFrame into a list of Pydantic model instances.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame to convert.
    model : Type[BaseModel], optional
        A pre-existing Pydantic model class to instantiate. If None,
        a model will be inferred from the DataFrame schema.
    model_name : str, optional
        If inferring a model, use this as the generated model class name.

    Returns
    -------
    List[BaseModel]
        List of Pydantic model instances representing each row of the DataFrame.

    Raises
    ------
    ValidationError
        If the DataFrame data does not conform to the inferred or provided model schema.
    """
    if model is None:
        model = infer_pydantic_model(df, model_name=model_name or "AutoModel")
    return [model(**row) for row in df.to_dicts()]

def _write_json_schema_to_tempfile(schema: dict):
    """
    Write a JSON Schema dictionary to a temporary file.

    Parameters
    ----------
    schema : dict
        The JSON Schema to write.

    Returns
    -------
    Tuple[tempfile.TemporaryDirectory, pathlib.Path]
        The TemporaryDirectory context and path to the written schema file.
    """
    temp_dir = tempfile.TemporaryDirectory()
    json_path = Path(temp_dir.name) / "schema.json"
    json_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return temp_dir, json_path

def _run_datamodel_codegen(schema_path: Path, output_path: Optional[Path]) -> str:
    """
    Run datamodel-code-generator to generate Python code from a JSON schema file.

    Parameters
    ----------
    schema_path : pathlib.Path
        Path to the input JSON schema file.
    output_path : pathlib.Path or None
        Optional path to write the generated Python code file.

    Returns
    -------
    str
        The generated Python source code as a string if output_path is None,
        otherwise the content of the written output file.
    """
    generate(
        schema_path,
        input_file_type=InputFileType.JsonSchema,
        output=str(output_path) if output_path else None,
    )
    if output_path:
        code = output_path.read_text(encoding="utf-8")
        return code
    else:
        # If no output path specified, codegen writes to current directory (default)
        # No reliable way to read code back, return empty string
        return ""

def generate_pydantic_class_code(
    model: Type[BaseModel],
    output_path: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate Python class source code from a Pydantic model using datamodel-code-generator.

    This method exports the Pydantic model schema as JSON Schema,
    saves it temporarily, and uses `datamodel-code-generator` to produce
    clean, annotated Python model class code.

    Parameters
    ----------
    model : Type[BaseModel]
        The Pydantic model class to convert to source code.
    output_path : str or pathlib.Path, optional
        If provided, save the generated Python code to this file.
    model_name : str, optional
        Optionally override the JSON Schema `title` before generation,
        which affects the class name in the generated code.

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

    temp_dir, schema_path = _write_json_schema_to_tempfile(schema)
    code = _run_datamodel_codegen(schema_path, Path(output_path) if output_path else None)
    temp_dir.cleanup()
    return code
