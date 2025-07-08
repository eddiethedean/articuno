"""
Unified inference utilities for Articuno.

This module provides high-level functions to infer Pydantic models from either
pandas or polars DataFrames — or directly from an iterable of dicts — with optional
support for nested columns, force_optional, and limited schema scan using the first N records.
For iterable dicts, inference is done strictly via Genson in the `iterable_infer` module.
Dependencies on pandas or polars are detected dynamically at call time.
"""

from typing import Any, Dict, Iterable, Generator, Optional, Type, Union

from pydantic import BaseModel

from articuno.iterable_infer import infer_generic_model, dicts_to_pydantic
from articuno.backend_detect import is_pandas_df, is_polars_df


def infer_pydantic_model(
    source: Union[Any, Iterable[Dict[str, Any]]],
    model_name: str = "AutoModel",
    force_optional: bool = False,
    max_scan: int = 1000,
) -> Type[BaseModel]:
    """
    Infer a Pydantic model class from the given source.

    Parameters
    ----------
    source : pandas.DataFrame, polars.DataFrame, or iterable of dict
        The input from which to infer a Pydantic model. Can be a pandas or polars DataFrame,
        or an iterable of dict records. Iterable inference scans up to `max_scan` records.
    model_name : str, default "AutoModel"
        Name to assign to the generated Pydantic model class.
    force_optional : bool, default False
        If True, forces all fields in the generated model to be Optional (applies to DataFrames).
    max_scan : int, default 1000
        Maximum number of records to scan when inferring from an iterable of dicts.

    Returns
    -------
    Type[BaseModel]
        Dynamically created Pydantic model class.

    Raises
    ------
    TypeError
        If `source` is not a supported DataFrame or iterable of dicts.
    """
    # Iterable of dicts → strict generic inference
    if isinstance(source, Iterable) and not is_pandas_df(source) \
       and not is_polars_df(source):
        return infer_generic_model(
            source,
            model_name=model_name,
            scan_limit=max_scan,
            force_optional=force_optional
        )

    # pandas DataFrame path
    if is_pandas_df(source):
        from articuno.pandas_infer import infer_pydantic_model as _infer_pd_model
        return _infer_pd_model(
            source,
            model_name=model_name,
            force_optional=force_optional
        )

    # polars DataFrame path
    if is_polars_df(source):
        from articuno.polars_infer import infer_pydantic_model as _infer_pl_model
        return _infer_pl_model(
            source,
            model_name=model_name,
            force_optional=force_optional
        )

    raise TypeError("Expected a pandas.DataFrame, polars.DataFrame, or iterable of dicts.")


def df_to_pydantic(
    source: Union[Any, Iterable[Dict[str, Any]]],
    model: Optional[Type[BaseModel]] = None,
    model_name: Optional[str] = None,
    force_optional: bool = False,
    max_scan: int = 1000,
) -> Generator[BaseModel, None, None]:
    """
    Convert a DataFrame or iterable of dicts into a generator of Pydantic model instances.

    Parameters
    ----------
    source : pandas.DataFrame, polars.DataFrame, or iterable of dict
        The input DataFrame or dict iterable.
    model : Type[BaseModel], optional
        Pre-existing Pydantic model class to use. If None, a model is inferred.
    model_name : str, optional
        Name for the auto-inferred model if `model` is None.
    force_optional : bool, default False
        If True, forces all fields in the inferred model to be Optional.
    max_scan : int, default 1000
        Maximum records to scan when inferring from a dict iterable.

    Returns
    -------
    Generator[BaseModel, None, None]
        A generator yielding Pydantic model instances for each row or record.

    Raises
    ------
    TypeError
        If `source` is not a supported DataFrame or iterable of dicts.
    """
    # Iterable-of-dicts path → generator
    if isinstance(source, Iterable) and not is_pandas_df(source) \
       and not is_polars_df(source):
        return dicts_to_pydantic(
            source,
            model=model,
            model_name=(model_name or "AutoDictModel"),
            force_optional=force_optional,
            scan_limit=max_scan
        )

    # DataFrame path: infer model if not provided
    if model is None:
        model = infer_pydantic_model(
            source,
            model_name or "AutoModel",
            force_optional=force_optional,
            max_scan=max_scan
        )

    # pandas DataFrame extraction → generator
    if is_pandas_df(source):
        rows = source.to_dict(orient="records")
        return (model(**row) for row in rows)

    # polars DataFrame extraction → generator
    if is_polars_df(source):
        rows = source.to_dicts()
        return (model(**row) for row in rows)

    raise TypeError("Expected a pandas.DataFrame, polars.DataFrame, or iterable of dicts.")