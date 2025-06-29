from typing import Any, Callable, Dict
import inspect


_inference_registry = []


def get_inference_registry():
    return _inference_registry


def infer_response_model(
    name: str,
    example_input: Dict[str, Any],
    models_path: str = "models.py"
) -> Callable:
    """
    Mark a FastAPI endpoint for automatic response model generation using Articuno.

    This decorator registers the endpoint function along with a model name and example input
    so the Articuno CLI can later call the function, inspect the Polars DataFrame output,
    and generate a matching Pydantic model.

    Args:
        name: The name of the Pydantic model to generate.
        example_input: A dictionary of example input values to call the endpoint with.
        models_path: Path to the file where the generated model should be written.
                     If relative (default: "models.py"), it's resolved relative to the app file.
    """
    def decorator(func: Callable) -> Callable:
        frame = inspect.currentframe().f_back
        filename = inspect.getfile(frame)
        lineno = frame.f_lineno

        _inference_registry.append({
            "func": func,
            "name": name,
            "example_input": example_input,
            "models_path": models_path,
            "source_file": filename,
            "source_line": lineno,
        })

        return func

    return decorator