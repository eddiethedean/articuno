"""
Pytest configuration and shared fixtures for articuno tests.
"""

import pytest
from typing import Type
from pydantic import BaseModel


# Optional imports
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import pyarrow as pa

    PYARROW_AVAILABLE = True
except ImportError:
    pa = None
    PYARROW_AVAILABLE = False


@pytest.fixture
def sample_polars_df():
    """Simple Polars DataFrame with basic types."""
    if not POLARS_AVAILABLE:
        pytest.skip("polars not installed")
    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 88.0, 92.3],
            "active": [True, False, True],
        }
    )


@pytest.fixture
def sample_pandas_df():
    """Simple Pandas DataFrame with basic types."""
    if not PANDAS_AVAILABLE:
        pytest.skip("pandas not installed")
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 88.0, 92.3],
            "active": [True, False, True],
        }
    )


@pytest.fixture
def sample_dicts():
    """Simple list of dictionaries."""
    return [
        {"id": 1, "name": "Alice", "score": 95.5, "active": True},
        {"id": 2, "name": "Bob", "score": 88.0, "active": False},
        {"id": 3, "name": "Charlie", "score": 92.3, "active": True},
    ]


def validate_model_schema(model: Type[BaseModel], expected_fields: dict):
    """
    Helper to validate that a Pydantic model has expected fields.

    Args:
        model: The Pydantic model class to validate
        expected_fields: Dict mapping field names to expected types
    """
    schema = (
        model.model_json_schema()
        if hasattr(model, "model_json_schema")
        else model.schema()
    )
    properties = schema.get("properties", {})

    for field_name, expected_type in expected_fields.items():
        assert field_name in properties, f"Field '{field_name}' not found in model"
        # Note: Full type validation would be more complex; this is a basic check


def assert_model_instance_valid(instance: BaseModel, expected_values: dict):
    """
    Helper to validate a Pydantic model instance has expected values.

    Args:
        instance: The Pydantic model instance
        expected_values: Dict of field names to expected values
    """
    for field_name, expected_value in expected_values.items():
        actual_value = getattr(instance, field_name)
        assert actual_value == expected_value, (
            f"Field '{field_name}': expected {expected_value}, got {actual_value}"
        )
