import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union, Type

from datamodel_code_generator import InputFileType, generate
from pydantic import BaseModel

def _write_json_schema_to_tempfile(schema: dict) -> Tuple[Path, tempfile.TemporaryDirectory]:
    temp_dir = tempfile.TemporaryDirectory()
    json_path = Path(temp_dir.name) / "schema.json"
    json_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return json_path, temp_dir

def _run_datamodel_codegen(input_path: Path, output_path: Optional[Path] = None) -> str:
    generate(
        str(input_path),
        input_file_type=InputFileType.JsonSchema,
        output=output_path,
    )
    if output_path:
        return output_path.read_text(encoding="utf-8")
    else:
        default_file = Path("model.py")
        return default_file.read_text(encoding="utf-8")

def generate_class_code(
    model: Type[BaseModel],
    output_path: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
) -> str:
    if hasattr(model, "model_json_schema"):
        schema = model.model_json_schema()
    else:
        schema = model.schema()

    if model_name:
        schema["title"] = model_name

    schema_path, temp_dir = _write_json_schema_to_tempfile(schema)

    output_file_path = Path(output_path) if output_path else None

    code = _run_datamodel_codegen(schema_path, output_file_path)

    return code
