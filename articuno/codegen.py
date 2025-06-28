import json
from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel
from datamodel_code_generator import InputFileType, generate


def generate_pydantic_class_code(
    model: Type[BaseModel],
    output_path: Optional[str | Path] = None,
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
    # Extract model schema depending on Pydantic version
    if hasattr(model, 'model_json_schema'):
        schema = model.model_json_schema()
    else:
        schema = model.schema()

    # Optionally rename the model in the schema title
    if model_name:
        schema['title'] = model_name

    # Dump the schema to a string
    schema_str = json.dumps(schema, indent=2)

    # Use a temporary file for the output code
    with Path().cwd().joinpath(".tmp_model_output.py").open("w+") as tmp_output:
        generate(
            input_text=schema_str,
            input_file_type=InputFileType.JsonSchema,
            output=tmp_output.name,
        )
        tmp_output.seek(0)
        code = tmp_output.read()

    # Save to final location if needed
    if output_path:
        Path(output_path).write_text(code)

    return code
