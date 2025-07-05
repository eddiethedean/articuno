import json
import tempfile
from pathlib import Path
from typing import Optional, Type

from datamodel_code_generator import InputFileType, generate
from pydantic import BaseModel

from .models import BaseModel  # for type hints, optional if in same package


def _write_json_schema_to_tempfile(schema: dict) -> Path:
    """
    Write JSON schema to a temporary file and return the Path.
    The temp directory is kept alive by attaching to the Path object.
    """
    temp_dir = tempfile.TemporaryDirectory()
    json_path = Path(temp_dir.name) / "schema.json"
    json_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    json_path.temp_dir = temp_dir  # keep reference to avoid deletion
    return json_path


def _run_datamodel_codegen(schema_path: Path, output_file: Optional[Path]) -> str:
    generate(
        input_=schema_path,
        input_file_type=InputFileType.JsonSchema,
        output=output_file,
    )
    code = output_file.read_text(encoding="utf-8") if output_file else ""
    return code


def generate_class_code(
    model: Type[BaseModel],
    output_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate Python class source code from a Pydantic or Patito model
    using datamodel-code-generator.

    Parameters
    ----------
    model : Type[BaseModel]
        The Pydantic or Patito model class.
    output_path : str, optional
        Path to save the generated code file.
    model_name : str, optional
        Override the model name in the generated schema.

    Returns
    -------
    str
        The generated Python code as a string.
    """
    if hasattr(model, "model_json_schema"):
        schema = model.model_json_schema()
    else:
        schema = model.schema()

    if model_name:
        schema["title"] = model_name

    schema_path = _write_json_schema_to_tempfile(schema)
    output_file = Path(output_path) if output_path else None

    code = _run_datamodel_codegen(schema_path, output_file)
    if output_file is None:
        # Clean up temp directory if no output path was specified
        schema_path.temp_dir.cleanup()

    return code
