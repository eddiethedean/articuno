# articuno/__init__.py

from .inference import df_to_pydantic, infer_pydantic_model
from .codegen import generate_class_code
from .iterable_infer import dicts_to_pydantic, infer_generic_model

# Optional conversion functions (available when dependencies are installed)
try:
    from .sqlalchemy_infer import (  # noqa: F401
        infer_pydantic_model_from_sqlalchemy,
        pydantic_to_sqlalchemy,
    )

    _SQLALCHEMY_AVAILABLE = True
except ImportError:
    _SQLALCHEMY_AVAILABLE = False

try:
    from .sqlmodel_infer import (  # noqa: F401
        infer_pydantic_model_from_sqlmodel,
        pydantic_to_sqlmodel,
    )

    _SQLMODEL_AVAILABLE = True
except ImportError:
    _SQLMODEL_AVAILABLE = False

try:
    from .pyspark_infer import pydantic_to_pyspark  # noqa: F401

    _PYSPARK_AVAILABLE = True
except ImportError:
    _PYSPARK_AVAILABLE = False

__all__ = [
    "df_to_pydantic",
    "generate_class_code",
    "infer_pydantic_model",
    "dicts_to_pydantic",
    "infer_generic_model",
]

# Add optional exports if available
if _SQLALCHEMY_AVAILABLE:
    __all__.extend(
        [
            "infer_pydantic_model_from_sqlalchemy",
            "pydantic_to_sqlalchemy",
        ]
    )

if _SQLMODEL_AVAILABLE:
    __all__.extend(
        [
            "infer_pydantic_model_from_sqlmodel",
            "pydantic_to_sqlmodel",
        ]
    )

if _PYSPARK_AVAILABLE:
    __all__.append("pydantic_to_pyspark")

__version__ = "0.10.0"
