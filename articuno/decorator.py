from typing import Callable, Optional, Dict, Any


def infer_response_model(
    name: str,
    example_input: Optional[Dict[str, Any]] = None,
    models_path: Optional[str] = None,
    use_patito: bool = False,
):
    """
    Decorator to mark FastAPI endpoint functions for Articuno model inference.

    Parameters
    ----------
    name : str
        Name of the generated model class.
    example_input : dict, optional
        Example input dictionary to call the endpoint for inference.
    models_path : str, optional
        Optional path to the models file.
    use_patito : bool, optional
        If True, use Patito models instead of Pydantic.

    Usage
    -----
    @infer_response_model(name="UserModel", example_input={"limit": 5})
    @app.get("/users")
    def get_users(limit: int):
        ...
    """
    def decorator(func: Callable):
        setattr(
            func,
            "__articuno_infer_response_model__",
            {
                "name": name,
                "example_input": example_input,
                "models_path": models_path,
                "use_patito": use_patito,
            },
        )
        return func
    return decorator
