import json
import tempfile
from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel
from datamodel_code_generator import InputFileType, generate


def generate_pydantic_class_code(
    model: Type[BaseModel],
    output_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate Python class source code from a Pydantic model using datamodel-code-generator's API.

    Args:
        model: A Pydantic model class (can be dynamic).
        output_path: Optional file path to write the generated code to.
        model_name: Optional override for the class name in the output.

    Returns:
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
            output=output_file,  # Pass Path object here, not str
        )

        code = output_file.read_text(encoding="utf-8")

        if output_path:
            Path(output_path).write_text(code, encoding="utf-8")

        return code
