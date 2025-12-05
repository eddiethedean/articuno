"""
Backend detection utilities for Articuno.

Provides functions to detect pandas, polars, PySpark DataFrame types, and
SQLAlchemy/SQLModel model classes at runtime, allowing dynamic installation
of dependencies after Articuno has been imported.
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


def is_pyspark_df(obj: Any) -> bool:
    """
    Check if the given object is a PySpark DataFrame.

    Returns False if PySpark is not installed.
    """
    try:
        from pyspark.sql import DataFrame  # type: ignore
    except ImportError:
        return False
    return isinstance(obj, DataFrame)


def is_sqlalchemy_model(obj: Any) -> bool:
    """
    Check if the given object is a SQLAlchemy declarative model class.

    Returns False if SQLAlchemy is not installed.
    """
    try:
        from sqlalchemy.orm import DeclarativeBase
        from sqlalchemy import inspect
    except ImportError:
        return False

    # Check if it's a class (not an instance)
    if not isinstance(obj, type):
        return False

    # Check if it's a subclass of DeclarativeBase
    try:
        if not issubclass(obj, DeclarativeBase):
            return False
        # Check if it has a mapper (is a mapped class)
        try:
            inspect(obj)
            return True
        except Exception:
            return False
    except Exception:
        return False


def is_sqlmodel_model(obj: Any) -> bool:
    """
    Check if the given object is a SQLModel class.

    Returns False if SQLModel is not installed.
    """
    try:
        from sqlmodel import SQLModel  # type: ignore
    except ImportError:
        return False

    # Check if it's a class (not an instance)
    if not isinstance(obj, type):
        return False

    # Check if it's a subclass of SQLModel
    try:
        return issubclass(obj, SQLModel)
    except Exception:
        return False
