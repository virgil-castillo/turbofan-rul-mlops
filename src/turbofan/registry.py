"""Compatibility facade for MLflow model registry helpers.

Pyfunc model packaging lives in :mod:`turbofan.registry_pyfunc`; registry URI
resolution, loading, and listing live in :mod:`turbofan.registry_store`. This
module preserves the historical ``turbofan.registry`` import path.
"""
from __future__ import annotations

from turbofan.registry_pyfunc import (
    PREDICTION_OUTPUT_COLUMNS,
    GRUFinalWindowModel,
    RidgeEngineModel,
    SequenceFinalWindowModel,
    log_and_register,
)
from turbofan.registry_store import (
    RegisteredModelInfo,
    latest_version,
    list_registered,
    load,
    load_predictor,
    load_predictor_from_uri,
    model_name,
    model_type_from_name,
    parse_models_uri,
    promote,
    resolve_uri,
)

_latest_version = latest_version
_parse_models_uri = parse_models_uri

__all__ = [
    "GRUFinalWindowModel",
    "PREDICTION_OUTPUT_COLUMNS",
    "RegisteredModelInfo",
    "RidgeEngineModel",
    "SequenceFinalWindowModel",
    "list_registered",
    "load",
    "load_predictor",
    "load_predictor_from_uri",
    "log_and_register",
    "model_name",
    "model_type_from_name",
    "promote",
    "resolve_uri",
]
