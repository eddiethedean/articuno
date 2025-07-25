"""
Backend detection utilities for Articuno.

Provides functions to detect pandas or polars DataFrame types at runtime,
allowing dynamic installation of dependencies after Articuno has been imported.
"""

from typing import Any


def is_pandas_df(obj: Any) -> bool:
    """
    Check if the given object is a pandas DataFrame.

    Returns False if pandas is not installed.
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return False
    return isinstance(obj, pd.DataFrame)


def is_polars_df(obj: Any) -> bool:
    """
    Check if the given object is a polars DataFrame.

    Returns False if polars is not installed.
    """
    try:
        import polars as pl  # type: ignore
    except ImportError:
        return False
    return isinstance(obj, pl.DataFrame)
