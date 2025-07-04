from typing import List, Optional, Type, Any
import polars as pl
from .models import df_to_models, infer_model


def infer_pydantic_model(
    df: pl.DataFrame,
    model_name: str = "AutoModel",
    _model_cache: Optional[dict] = None,
) -> Type[Any]:
    return infer_model(df, model_name=model_name, use_patito=False, _model_cache=_model_cache)

def df_to_pydantic(
    df: pl.DataFrame,
    model: Optional[Type] = None,
    model_name: Optional[str] = None,
) -> List[Any]:
    return df_to_models(df, model=model, model_name=model_name, use_patito=False)


def infer_patito_model(
    df: pl.DataFrame,
    model_name: str = "AutoPatitoModel",
    _model_cache: Optional[dict] = None,
) -> Type[Any]:
    return infer_model(df, model_name=model_name, use_patito=True, _model_cache=_model_cache)

def df_to_patito(
    df: pl.DataFrame,
    model: Optional[Type] = None,
    model_name: Optional[str] = None,
) -> List[Any]:
    return df_to_models(df, model=model, model_name=model_name, use_patito=True)
