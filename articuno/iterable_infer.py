"""
Iterable inference utilities for Articuno.

Provides functions to infer Pydantic models strictly from iterables of dict records
using pydantic.create_model and the nested-dict logic extracted.
"""

from typing import Any, Dict, Iterable, Type, Optional, Generator, List, Union
from pydantic import BaseModel, create_model
import itertools
from collections.abc import Mapping

# Nested dict-model logic
from articuno.dict_model import _infer_dict_model


def infer_generic_model(
    records: Iterable[Dict[str, Any]],
    model_name: str = "AutoDictModel",
    scan_limit: int = 1000,
    force_optional: bool = False,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from an iterable of dict records.

    Parameters
    ----------
    records : Iterable[Dict[str, Any]]
        Iterable of dictionary records for schema inference.
    model_name : str, optional
        Name of the generated Pydantic model class.
    scan_limit : int, optional
        Maximum number of records to scan for inference.
    force_optional : bool, optional
        If True, all fields are made Optional regardless of data.

    Returns
    -------
    Type[BaseModel]
        Dynamically created Pydantic model class.

    Raises
    ------
    ValueError
        If no records are provided.
    """
    # Sample records
    sample = list(itertools.islice(records, scan_limit))
    if not sample:
        raise ValueError("Cannot infer schema from empty iterable of records.")

    # Gather all field names
    keys = set().union(*(rec.keys() for rec in sample))
    fields: Dict[str, tuple] = {}

    for key in keys:
        vals = [rec.get(key) for rec in sample]
        non_nulls = [v for v in vals if v is not None]
        always_present = all(key in rec for rec in sample)
        has_null = any(v is None for v in vals)

        # Determine type
        if all(isinstance(v, Mapping) or v is None for v in vals):
            # nested dict
            typ = _infer_dict_model(non_nulls, key, force_optional=force_optional)
        elif all(isinstance(v, list) or v is None for v in vals):
            typ = List[Any]
        else:
            sample_val = non_nulls[0] if non_nulls else None
            if isinstance(sample_val, int):
                typ = int
            elif isinstance(sample_val, float):
                typ = float
            elif isinstance(sample_val, bool):
                typ = bool
            elif isinstance(sample_val, str):
                typ = str
            else:
                typ = Any

        # Apply optional if needed
        if force_optional or has_null or not always_present:
            from typing import Optional as _Opt
            typ = _Opt[typ]
            default = None
        else:
            default = ...

        fields[key] = (typ, default)

    # Create model
    Model = create_model(model_name, **fields)
    return Model


def dicts_to_pydantic(
    records: Iterable[Dict[str, Any]],
    model: Optional[Type[BaseModel]] = None,
    model_name: str = "AutoDictModel",
    scan_limit: int = 1000,
    force_optional: bool = False,
) -> Generator[BaseModel, None, None]:
    """
    Convert an iterable of dicts into a generator of Pydantic model instances.

    Parameters
    ----------
    records : Iterable[Dict[str, Any]]
        Iterable of dictionary records to convert.
    model : Type[BaseModel], optional
        Pre-defined Pydantic model. If None, one is inferred.
    model_name : str, optional
        Name for the generated model if inferring.
    scan_limit : int, optional
        Maximum number of records to scan for inference.
    force_optional : bool, optional
        If True, all fields in the inferred model will be Optional.

    Yields
    ------
    BaseModel
        An instance of the Pydantic model for each input record.
    """
    # Preserve iterator
    records, scan_records = itertools.tee(records, 2)

    if model is None:
        sample = itertools.islice(scan_records, scan_limit)
        model = infer_generic_model(
            sample,
            model_name=model_name,
            scan_limit=scan_limit,
            force_optional=force_optional,
        )

    for record in records:
        yield model(**record)
