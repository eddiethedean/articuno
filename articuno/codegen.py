"""
Code generation utilities for converting Pydantic models into class definitions using datamodel-code-generator.

This module enables you to extract the JSON schema from a Pydantic model
and generate equivalent Python code using the powerful `datamodel-code-generator` package.
"""

import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union, Type

from datamodel_code_generator import InputFileType, generate
from pydantic import BaseModel


def _write_json_schema_to_tempfile(schema: dict) -> Tuple[Path, tempfile.TemporaryDirectory]:
    """
    Write a JSON schema to a temporary file.

    This function serializes the provided JSON schema dictionary to a `.json` file
    in a new temporary directory. The directory is returned to allow cleanup control.

    Parameters
    ----------
    schema : dict
        A JSON schema dictionary typically produced by `model.model_json_schema()` or `model.schema()`.

    Returns
    -------
    Tuple[Path, tempfile.TemporaryDirectory]
        A tuple containing the path to the written JSON schema file and the corresponding temporary directory.
    """
    temp_dir = tempfile.TemporaryDirectory()
    json_path = Path(temp_dir.name) / "schema.json"
    json_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return json_path, temp_dir


def _run_datamodel_codegen(input_path: Path) -> str:
    """
    Run `datamodel-code-generator` on a JSON schema file and return the generated code as a string.

    This function creates a temporary output file, invokes the generator,
    and reads the resulting Python code.

    Parameters
    ----------
    input_path : Path
        Path to a valid JSON schema file.

    Returns
    -------
    str
        The Python class definitions generated from the schema.
    """
    with tempfile.TemporaryDirectory() as temp_output_dir:
        temp_output_path = Path(temp_output_dir) / "model.py"
        generate(
            input_=input_path,
            input_file_type=InputFileType.JsonSchema,
            output=temp_output_path,
        )
        return temp_output_path.read_text(encoding="utf-8")


def generate_class_code(
    model: Type[BaseModel],
    output_path: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate Python class code from a Pydantic model using datamodel-code-generator.

    This function extracts the JSON schema from a Pydantic model and uses
    `datamodel-code-generator` to convert it into Python class code.

    If `output_path` is provided, the generated code is written to that file.
    Otherwise, the code is returned as a string.

    Parameters
    ----------
    model : Type[BaseModel]
        A Pydantic model class to convert to source code.
    output_path : Union[str, Path], optional
        If provided, the code will be written to this file path.
    model_name : str, optional
        If provided, overrides the default model name in the generated schema.

    Returns
    -------
    str
        The generated Python class code as a string.
    """
    schema = (
        model.model_json_schema()
        if hasattr(model, "model_json_schema")
        else model.schema()
    )

    if model_name:
        schema["title"] = model_name

    schema_path, temp_dir = _write_json_schema_to_tempfile(schema)

    if output_path:
        generate(
            input_=schema_path,
            input_file_type=InputFileType.JsonSchema,
            output=Path(output_path),
        )
        return Path(output_path).read_text(encoding="utf-8")
    else:
        return _run_datamodel_codegen(schema_path)
