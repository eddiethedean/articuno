import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union, Type

from datamodel_code_generator import InputFileType, generate
from pydantic import BaseModel

def _write_json_schema_to_tempfile(schema: dict) -> Tuple[Path, tempfile.TemporaryDirectory]:
    """
    Write the JSON schema dict to a temporary file.

    Returns:
        Tuple containing the Path to the JSON file and the TemporaryDirectory object.
        The TemporaryDirectory must be kept alive while the file is in use.
    """
    temp_dir = tempfile.TemporaryDirectory()
    json_path = Path(temp_dir.name) / "schema.json"
    json_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return json_path, temp_dir

def _run_datamodel_codegen(input_path: Path, output_path: Optional[Path] = None) -> str:
    """
    Run datamodel-code-generator on the input JSON schema file and return the generated code.

    Parameters:
        input_path: Path to the JSON schema file.
        output_path: Optional Path to save the generated Python code.

    Returns:
        The generated Python source code as a string.
    """
    generate(
        input_file=str(input_path),
        input_file_type=InputFileType.JsonSchema,
        output=str(output_path) if output_path else None,
    )
    if output_path:
        return output_path.read_text(encoding="utf-8")
    else:
        # If no output_path specified, read from the default output file "model.py" in cwd
        default_file = Path("model.py")
        return default_file.read_text(encoding="utf-8")

def generate_class_code(
    model: Type[BaseModel],
    output_path: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate Python class source code from a Pydantic model using datamodel-code-generator.

    Parameters:
        model: The Pydantic model class (can be dynamic).
        output_path: Optional path to write the generated Python code file.
        model_name: Optional model name to override the JSON schema title.

    Returns:
        The generated Python source code as a string.
    """
    # Get schema from model (pydantic v2 or v1 compatible)
    if hasattr(model, "model_json_schema"):
        schema = model.model_json_schema()
    else:
        schema = model.schema()

    if model_name:
        schema["title"] = model_name

    schema_path, temp_dir = _write_json_schema_to_tempfile(schema)

    output_file_path = Path(output_path) if output_path else None

    code = _run_datamodel_codegen(schema_path, output_file_path)

    # Keep temp_dir alive until after code generation to avoid deletion of temp file
    # temp_dir will be cleaned up automatically when this function returns and temp_dir goes out of scope

    return code
