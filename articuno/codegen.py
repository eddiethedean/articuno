import json
import tempfile
from pathlib import Path
from typing import Optional, Type, Union

from pydantic import BaseModel

try:
    import patito as pt
    _has_patito = True
except ImportError:
    _has_patito = False

from datamodel_code_generator import InputFileType, generate


def _write_json_schema_to_tempfile(schema: dict) -> Path:
    """
    Write a given JSON schema to a temporary file.

    :param schema: The JSON schema dictionary.
    :return: Path to the written temporary file.
    :rtype: Path
    """
    temp_dir = tempfile.TemporaryDirectory()
    json_path = Path(temp_dir.name) / "schema.json"
    json_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    json_path.temp_dir = temp_dir  # Keep reference to avoid deletion
    return json_path


def _run_datamodel_codegen(schema_path: Path, output_path: Optional[Path] = None) -> str:
    """
    Run datamodel-code-generator on a schema JSON file.

    :param schema_path: Path to the input JSON schema file.
    :param output_path: Optional output file path. If None, a temporary path is used.
    :return: The generated Python source code as a string.
    :rtype: str
    """
    if output_path is None:
        output_path = schema_path.parent / "model_output.py"

    generate(
        input=schema_path,
        input_file_type=InputFileType.JsonSchema,
        output=output_path,
    )
    return output_path.read_text(encoding="utf-8")


def generate_pydantic_class_code(
    model: Type[BaseModel],
    output_path: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate Python class source code from a Pydantic model.

    This function converts a dynamic Pydantic model to a JSON Schema,
    then uses `datamodel-code-generator` to produce Python source code.

    :param model: The Pydantic model class to convert.
    :type model: Type[BaseModel]
    :param output_path: Optional path to write the output `.py` file.
    :type output_path: Optional[Union[str, Path]]
    :param model_name: Optional override for the schema's title field.
    :type model_name: Optional[str]
    :return: The generated Python code as a string.
    :rtype: str
    """
    if hasattr(model, "model_json_schema"):
        schema = model.model_json_schema()
    else:
        schema = model.schema()

    if model_name:
        schema["title"] = model_name

    schema_path = _write_json_schema_to_tempfile(schema)
    return _run_datamodel_codegen(schema_path, Path(output_path) if output_path else None)


def generate_patito_class_code(
    model: Type["pt.Model"],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Generate Python class source code from a Patito model.

    This serializes the Patito model to JSON Schema and uses
    `datamodel-code-generator` to produce corresponding Python code.

    :param model: A Patito model class.
    :type model: Type[pt.Model]
    :param output_path: Optional path to save the generated code.
    :type output_path: Optional[Union[str, Path]]
    :return: The generated Python source code as a string.
    :rtype: str
    :raises ImportError: If Patito is not installed.
    """
    if not _has_patito:
        raise ImportError("Patito is not installed. Try `pip install patito`.")

    schema = model.model_json_schema()
    schema_path = _write_json_schema_to_tempfile(schema)
    return _run_datamodel_codegen(schema_path, Path(output_path) if output_path else None)
