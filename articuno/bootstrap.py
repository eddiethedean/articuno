import inspect
from typing import Any, Callable, Dict, List

_inference_registry: List[Dict[str, Any]] = []

def get_inference_registry() -> List[Dict[str, Any]]:
    """
    Retrieve the current registry of functions marked for response model inference.

    Returns
    -------
    List[Dict[str, Any]]
        A list of metadata dictionaries for registered endpoint functions,
        including function object, name, example input, models path,
        source file, and source line.
    """
    return _inference_registry

def infer_response_model(
    name: str,
    example_input: Dict[str, Any],
    models_path: str = "models.py"
) -> Callable:
    """
    Mark a FastAPI endpoint for automatic response model generation using Articuno.

    This decorator registers the endpoint function along with a model name and example input
    so that the Articuno CLI or tooling can later call the function, inspect the Polars
    DataFrame it returns, and generate a matching Pydantic model.

    Parameters
    ----------
    name : str
        The name of the Pydantic model to generate.
    example_input : dict[str, Any]
        A dictionary of example input values to call the endpoint with.
    models_path : str, optional
        Path to the file where the generated model should be written.
        If relative (default: "models.py"), it's resolved relative to the source file.

    Returns
    -------
    Callable
        The original endpoint function, unchanged.
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
