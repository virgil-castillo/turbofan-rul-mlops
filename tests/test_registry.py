"""Tests for turbofan.registry MLflow wrapper and pyfunc model packaging."""
from __future__ import annotations

from collections.abc import Sequence

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from turbofan.inference.schemas import FEATURE_COLUMNS
from turbofan.models.gru import GRURULRegressor
from turbofan.preprocessing.normalization import OperatingModeNormalizer


def _make_normalizer_payload(feature_cols: Sequence[str]) -> dict[str, object]:
    """Fit a minimal ``OperatingModeNormalizer`` and return its payload dict.

    Args:
        feature_cols: Feature columns to include in the normalizer.

    Returns:
        Payload dictionary produced by ``OperatingModeNormalizer.to_payload``.
    """
    normalizer = OperatingModeNormalizer(feature_cols=list(feature_cols))
    fit_df = pd.DataFrame({col: [0.0, 1.0] for col in feature_cols})
    fit_df["op_1"] = [0.0, 0.0]
    fit_df["op_2"] = [0.0, 0.0]
    fit_df["op_3"] = [0.0, 0.0]
    normalizer.fit(fit_df)
    return normalizer.to_payload()


def _record(
    *,
    engine_id: int,
    cycle: int,
    feature_value: float = 1.0,
) -> dict[str, object]:
    """Build one canonical inference record.

    Args:
        engine_id: Engine identifier.
        cycle: Cycle identifier.
        feature_value: Value used for all feature columns.

    Returns:
        Canonical inference record.
    """
    return {
        "engine_id": engine_id,
        "cycle": cycle,
        **{column: feature_value for column in FEATURE_COLUMNS},
    }


def _records_for_engine(
    engine_id: int,
    cycles: int,
    *,
    feature_value: float = 1.0,
) -> list[dict[str, object]]:
    """Build canonical records for one engine.

    Args:
        engine_id: Engine identifier.
        cycles: Number of cycles to generate.
        feature_value: Value used for all feature columns.

    Returns:
        Canonical records sorted by cycle.
    """
    return [
        _record(engine_id=engine_id, cycle=cycle, feature_value=feature_value)
        for cycle in range(1, cycles + 1)
    ]


def _raw_records_df() -> pd.DataFrame:
    """Build a multi-engine raw-records DataFrame with varied cycle lengths.

    Returns:
        Canonical raw records for engines 1 and 2.
    """
    return pd.DataFrame(
        [
            *_records_for_engine(1, 5, feature_value=1.0),
            *_records_for_engine(2, 4, feature_value=2.0),
        ]
    )


def _fitted_ridge_pipeline() -> Pipeline:
    """Fit a tiny self-contained Ridge pipeline on canonical raw records.

    Mirrors ``build_baseline_pipeline`` in that the pipeline consumes the full
    canonical frame (including ``engine_id``/``cycle``) and selects the feature
    columns internally, so it scores the validated frame exactly as the
    production pipeline does.

    Returns:
        A fitted sklearn ``Pipeline`` mapping canonical raw records to a target.
    """
    frame = _raw_records_df()
    target = frame["cycle"].to_numpy(dtype=float)
    pipeline = Pipeline(
        [
            (
                "features",
                ColumnTransformer([("keep", "passthrough", FEATURE_COLUMNS)]),
            ),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(frame, target)
    return pipeline


def _gru_payload(*, window_size: int = 3, max_rul: int = 125) -> dict[str, object]:
    """Build a tiny GRU checkpoint payload mirroring the production format.

    Args:
        window_size: Sequence window size.
        max_rul: Maximum-RUL cap applied during rescaling.

    Returns:
        Checkpoint payload dict with model_state_dict, feature_cols,
        normalizer_payload, sequence_config, and max_rul.
    """
    model = GRURULRegressor(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    return {
        "model_state_dict": model.state_dict(),
        "feature_cols": list(FEATURE_COLUMNS),
        "sequence_config": {
            "architecture": "gru",
            "window_size": window_size,
            "hidden_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "normalizer_type": "operating_mode",
        "normalizer_payload": _make_normalizer_payload(FEATURE_COLUMNS),
        "max_rul": max_rul,
    }


# ---------------------------------------------------------------------------
# model_name
# ---------------------------------------------------------------------------


def test_model_name_gru_lowercases_subset() -> None:
    """model_name builds the canonical GRU registered-model name."""
    from turbofan.registry import model_name

    assert model_name("gru", "FD001") == "turbofan-gru-fd001"


def test_model_name_ridge_variants() -> None:
    """model_name builds canonical Ridge names across subsets."""
    from turbofan.registry import model_name

    assert model_name("ridge", "FD001") == "turbofan-ridge-fd001"
    assert model_name("ridge", "fd003") == "turbofan-ridge-fd003"


# ---------------------------------------------------------------------------
# register + version increment
# ---------------------------------------------------------------------------


def test_log_and_register_ridge_increments_version() -> None:
    """Registering a Ridge model twice yields versions 1 then 2."""
    from turbofan.registry import log_and_register

    pipeline = _fitted_ridge_pipeline()
    with mlflow.start_run():
        first = log_and_register(pipeline, model_type="ridge", subset="FD001")
    with mlflow.start_run():
        second = log_and_register(pipeline, model_type="ridge", subset="FD001")

    assert first == 1
    assert second == 2


def test_log_and_register_gru_increments_version() -> None:
    """Registering a GRU payload twice yields versions 1 then 2."""
    from turbofan.registry import log_and_register

    payload = _gru_payload()
    with mlflow.start_run():
        first = log_and_register(payload, model_type="gru", subset="FD001")
    with mlflow.start_run():
        second = log_and_register(payload, model_type="gru", subset="FD001")

    assert first == 1
    assert second == 2


# ---------------------------------------------------------------------------
# promote + resolve + load
# ---------------------------------------------------------------------------


def test_promote_and_resolve_uri() -> None:
    """promote sets the alias and resolve_uri formats the models URI."""
    from turbofan.registry import (
        log_and_register,
        model_name,
        promote,
        resolve_uri,
    )

    name = model_name("ridge", "FD001")
    pipeline = _fitted_ridge_pipeline()
    with mlflow.start_run():
        version = log_and_register(pipeline, model_type="ridge", subset="FD001")

    promote(name, version)

    assert resolve_uri(name) == f"models:/{name}@production"


def test_load_returns_aliased_model() -> None:
    """load resolves and loads the aliased pyfunc model."""
    from turbofan.registry import load, log_and_register, model_name, promote

    name = model_name("ridge", "FD001")
    pipeline = _fitted_ridge_pipeline()
    with mlflow.start_run():
        version = log_and_register(pipeline, model_type="ridge", subset="FD001")
    promote(name, version)

    model = load(name)

    assert isinstance(model, mlflow.pyfunc.PyFuncModel)


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


def test_promote_rollback_repoints_alias_to_earlier_version() -> None:
    """Promoting v2 then v1 repoints the production alias back to v1."""
    from mlflow.tracking import MlflowClient

    from turbofan.registry import log_and_register, model_name, promote

    name = model_name("ridge", "FD001")
    pipeline = _fitted_ridge_pipeline()
    with mlflow.start_run():
        log_and_register(pipeline, model_type="ridge", subset="FD001")
    with mlflow.start_run():
        log_and_register(pipeline, model_type="ridge", subset="FD001")

    promote(name, 2)
    promote(name, 1)

    aliased = MlflowClient().get_model_version_by_alias(name, "production")
    assert int(aliased.version) == 1


# ---------------------------------------------------------------------------
# list_registered shape
# ---------------------------------------------------------------------------


def test_list_registered_reports_production_metric_and_run() -> None:
    """list_registered reports name, production version, val_rmse, and run id."""
    from turbofan.registry import (
        list_registered,
        log_and_register,
        model_name,
        promote,
    )

    name = model_name("ridge", "FD001")
    pipeline = _fitted_ridge_pipeline()
    with mlflow.start_run() as run:
        mlflow.log_metric("val_rmse", 12.5)
        version = log_and_register(pipeline, model_type="ridge", subset="FD001")
        run_id = run.info.run_id
    promote(name, version)

    entries = list_registered()

    matching = [entry for entry in entries if entry.name == name]
    assert len(matching) == 1
    entry = matching[0]
    assert entry.production_version == version
    assert entry.versions == [version]
    assert entry.val_rmse == pytest.approx(12.5)
    assert entry.run_id == run_id


def test_list_registered_handles_missing_production_alias() -> None:
    """list_registered reports None for production fields when no alias is set."""
    from turbofan.registry import list_registered, log_and_register, model_name

    name = model_name("gru", "FD002")
    payload = _gru_payload()
    with mlflow.start_run():
        log_and_register(payload, model_type="gru", subset="FD002")

    entries = list_registered()

    matching = [entry for entry in entries if entry.name == name]
    assert len(matching) == 1
    entry = matching[0]
    assert entry.production_version is None
    assert entry.val_rmse is None
    assert entry.run_id is None


# ---------------------------------------------------------------------------
# Wrapper parity — the critical guard
# ---------------------------------------------------------------------------


def test_ridge_wrapper_roundtrip_matches_in_process_predictor() -> None:
    """A logged+loaded Ridge model predicts identically to RidgePredictor logic."""
    from turbofan.inference.predictors import ridge_engine_predictions
    from turbofan.inference.schemas import validate_raw_records
    from turbofan.registry import load, log_and_register, model_name, promote

    pipeline = _fitted_ridge_pipeline()
    records = _raw_records_df()

    name = model_name("ridge", "FD001")
    with mlflow.start_run():
        version = log_and_register(pipeline, model_type="ridge", subset="FD001")
    promote(name, version)

    loaded = load(name)
    roundtrip = np.asarray(loaded.predict(records), dtype=np.float64).reshape(-1)

    validated = validate_raw_records(records)
    _, expected = ridge_engine_predictions(pipeline, validated.records)

    assert np.allclose(roundtrip, expected)


def test_gru_wrapper_roundtrip_matches_in_process_predictor() -> None:
    """A logged+loaded GRU model predicts identically to GRUPredictor logic."""
    from turbofan.inference.predictors import gru_final_window_predictions
    from turbofan.inference.schemas import validate_raw_records
    from turbofan.registry import load, log_and_register, model_name, promote

    window_size = 3
    payload = _gru_payload(window_size=window_size, max_rul=125)
    # Engines long enough for the window so no padding warnings are needed.
    records = pd.DataFrame(
        [
            *_records_for_engine(1, 5, feature_value=1.0),
            *_records_for_engine(2, 4, feature_value=2.0),
        ]
    )

    name = model_name("gru", "FD001")
    with mlflow.start_run():
        version = log_and_register(payload, model_type="gru", subset="FD001")
    promote(name, version)

    loaded = load(name)
    roundtrip = np.asarray(loaded.predict(records), dtype=np.float64).reshape(-1)

    validated = validate_raw_records(records)
    _, expected = gru_final_window_predictions(payload, validated.records)

    assert np.allclose(roundtrip, expected)
