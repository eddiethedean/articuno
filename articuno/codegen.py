import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union, Type
from io import StringIO

from datamodel_code_generator import InputFileType, generate
from pydantic import BaseModel


def _write_json_schema_to_tempfile(schema: dict) -> Tuple[Path, tempfile.TemporaryDirectory]:
    """
    Write the JSON schema to a temporary file and return the file path and its temp directory.
    """
    temp_dir = tempfile.TemporaryDirectory()
    json_path = Path(temp_dir.name) / "schema.json"
    json_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return json_path, temp_dir


def _run_datamodel_codegen(input_path: Path, output_path: Optional[Path] = None) -> str:
    """
    Run datamodel-code-generator on the given schema file path and capture the generated code.
    """
    # Use in-memory string buffer if no output path is provided
    if output_path is None:
        output_buffer = StringIO()
        generate(
            input_=input_path,
            input_file_type=InputFileType.JsonSchema,
            output=output_buffer,
        )
        return output_buffer.getvalue()
    else:
        generate(
            input_=input_path,
            input_file_type=InputFileType.JsonSchema,
            output=output_path,
        )
        return output_path.read_text(encoding="utf-8")


def generate_class_code(
    model: Type[BaseModel],
    output_path: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate Python class code from a Pydantic model using datamodel-code-generator.

    Parameters
    ----------
    model : Type[BaseModel]
        The Pydantic model to generate code for.
    output_path : str or Path, optional
        If given, writes code to this path. Otherwise returns code as string.
    model_name : str, optional
        Optional name override for the model title.

    Returns
    -------
    str
        The generated Python source code.
    """
    schema = (
        model.model_json_schema()
        if hasattr(model, "model_json_schema")
        else model.schema()
    )

    if model_name:
        schema["title"] = model_name

    schema_path, temp_dir = _write_json_schema_to_tempfile(schema)

    output_file_path = Path(output_path) if output_path else None

    code = _run_datamodel_codegen(schema_path, output_file_path)

    return code
