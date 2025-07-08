"""
Nested dict model inference utilities for Articuno.

Provides a helper to generate nested Pydantic models for dict columns,
extracted from pandas_infer for reuse in iterable_infer.
"""

from typing import Any, Dict, List, Optional
from pydantic import create_model


def _infer_dict_model(
    samples: List[Dict[str, Any]],
    field_name: str,
    force_optional: bool = False,
) -> Any:
    """
    Merge keys from multiple sample dicts to create a nested Pydantic model.
    Parameters
    ----------
    samples : List[Dict[str, Any]]
        Sample values from a dict column.
    field_name : str
        Name of the parent field, used to name the nested model class.
    force_optional : bool, optional
        If True, all nested fields will be Optional.
    Returns
    -------
    Any
        A dynamically created nested Pydantic model class for the dict column.
    """
    # Collect all keys present in samples
    merged_keys = set()
    for sample in samples:
        merged_keys.update(sample.keys())

    fields: Dict[str, tuple] = {}
    for key in merged_keys:
        # Determine presence and nullability
        always_present = all(key in sample for sample in samples)
        has_null = any(sample.get(key) is None for sample in samples)
        # Pick a representative non-null value
        non_null_vals = [sample[key] for sample in samples if sample.get(key) is not None]
        example = non_null_vals[0] if non_null_vals else None

        # Infer type
        if isinstance(example, int):
            typ = int
        elif isinstance(example, float):
            typ = float
        elif isinstance(example, str):
            typ = str
        elif isinstance(example, bool):
            typ = bool
        elif isinstance(example, list):
            typ = List[Any]
        else:
            typ = Any

        # Apply optionality
        if force_optional or not always_present or has_null:
            from typing import Optional as _Opt
            typ = _Opt[typ]
            default = None
        else:
            default = ...

        fields[key] = (typ, default)

    model_cls = create_model(f"{field_name}_NestedModel", **fields)
    return model_cls
