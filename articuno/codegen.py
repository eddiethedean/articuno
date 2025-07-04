from typing import Optional, Type
from pydantic import BaseModel
import json
import tempfile
from pathlib import Path
from datamodel_code_generator import InputFileType, generate


def generate_pydantic_class_code(
    model: Type[BaseModel],
    output_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate Python class source code from a Pydantic model using datamodel-code-generator.

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


def generate_patito_class_code(
    model: Type,
    output_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate Python class source code from a Patito model.

    This function requires Patito to be installed.

    Parameters
    ----------
    model : Type
        A Patito model class.
    output_path : str or Path, optional
        If given, saves the generated class code to this file path.
    model_name : str, optional
        Optionally override the class name before generation.

    Returns
    -------
    str
        The generated Python source code as a string.
    """
    try:
        import patito
    except ImportError as e:
        raise ImportError(
            "The 'patito' package is required to use generate_patito_class_code. "
            "Please install it with `pip install patito`."
        ) from e

    source_code = patito.get_source(model)
    if model_name:
        import re
        source_code = re.sub(r"class\s+\w+", f"class {model_name}", source_code)

    if output_path:
        Path(output_path).write_text(source_code, encoding="utf-8")

    return source_code
