import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

from polars import DataFrame as PolarsDataFrame

_inference_registry: List[Dict[str, Any]] = []

def get_inference_registry() -> List[Dict[str, Any]]:
    """
    Returns the global list of registered endpoint metadata dictionaries.
    """
    return _inference_registry

def infer_response_model(
    name: str,
    example_input: Dict[str, Any],
    models_path: str = "models.py",
):
    """
    Decorator to mark a FastAPI endpoint function for automatic Pydantic/Patito model generation.

    Registers the function with its model name, example input, and target models file path.

    Parameters
    ----------
    name : str
        The name of the Pydantic/Patito model to generate.
    example_input : dict[str, Any]
        Example input data to call the function with.
    models_path : str, optional
        Path to the file where the generated model should be written. Defaults to "models.py".

    Returns
    -------
    Callable
        The original function, unchanged.
    """
    def decorator(func: Callable):
        _inference_registry.append({
            "func": func,
            "name": name,
            "example_input": example_input,
            "models_path": models_path,
            "source_file": inspect.getfile(func),
            "source_line": inspect.getsourcelines(func)[1],
        })
        return func
    return decorator
