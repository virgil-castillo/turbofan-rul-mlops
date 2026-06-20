"""Tests for LSTM support in the turbofan MLflow registry wrapper."""
from __future__ import annotations

from collections.abc import Sequence

import mlflow
import numpy as np
import pandas as pd
import torch

from turbofan.inference.schemas import FEATURE_COLUMNS
from turbofan.models.sequence_models import build_sequence_model
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
        **dict.fromkeys(FEATURE_COLUMNS, feature_value),
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


def _lstm_payload(*, window_size: int = 3, max_rul: int = 125) -> dict[str, object]:
    """Build a tiny LSTM checkpoint payload mirroring the production format.

    The torch RNG is seeded so the LSTM weights are fixed across runs and yield
    per-engine-distinct, positive predictions, so a clipped-to-zero degenerate
    round trip cannot let a parity assertion pass trivially.

    Args:
        window_size: Sequence window size.
        max_rul: Maximum-RUL cap applied during rescaling.

    Returns:
        Checkpoint payload dict.
    """
    torch.manual_seed(0)
    model = build_sequence_model(
        "lstm",
        input_size=len(FEATURE_COLUMNS),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    return {
        "model_state_dict": model.state_dict(),
        "feature_cols": list(FEATURE_COLUMNS),
        "sequence_config": {
            "architecture": "lstm",
            "window_size": window_size,
            "hidden_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "normalizer_type": "operating_mode",
        "normalizer_payload": _make_normalizer_payload(FEATURE_COLUMNS),
        "max_rul": max_rul,
    }


def test_model_name_lstm_lowercases_subset() -> None:
    """model_name builds the canonical LSTM registered-model name."""
    from turbofan.registry import model_name

    assert model_name("lstm", "FD001") == "turbofan-lstm-fd001"


def test_model_type_from_name_resolves_lstm() -> None:
    """model_type_from_name infers lstm from a canonical LSTM model name."""
    from turbofan.registry import model_type_from_name

    assert model_type_from_name("turbofan-lstm-fd001") == "lstm"


def test_log_and_register_lstm_increments_version() -> None:
    """Registering an LSTM payload twice yields versions 1 then 2."""
    from turbofan.registry import log_and_register

    payload = _lstm_payload()
    with mlflow.start_run():
        first = log_and_register(payload, model_type="lstm", subset="FD001")
    with mlflow.start_run():
        second = log_and_register(payload, model_type="lstm", subset="FD001")

    assert first == 1
    assert second == 2


def test_sequence_wrapper_is_shared_for_lstm() -> None:
    """The shared SequenceFinalWindowModel runs the LSTM final-window contract."""
    from turbofan.registry import SequenceFinalWindowModel

    assert hasattr(SequenceFinalWindowModel, "predict")


def test_lstm_wrapper_roundtrip_matches_in_process_predictor() -> None:
    """A logged+loaded LSTM model predicts identically to the shared compute."""
    from turbofan.inference.predictors import sequence_final_window_predictions
    from turbofan.inference.schemas import validate_raw_records
    from turbofan.registry import load, log_and_register, model_name, promote

    window_size = 3
    payload = _lstm_payload(window_size=window_size, max_rul=125)
    records = pd.DataFrame(
        [
            *_records_for_engine(1, 5, feature_value=1.0),
            *_records_for_engine(2, 4, feature_value=2.0),
        ]
    )

    name = model_name("lstm", "FD001")
    with mlflow.start_run():
        version = log_and_register(payload, model_type="lstm", subset="FD001")
    promote(name, version)

    loaded = load(name)
    output = loaded.predict(records)

    validated = validate_raw_records(records)
    expected_meta, expected = sequence_final_window_predictions(
        payload, validated.records
    )

    assert list(output.columns) == ["engine_id", "cycle", "prediction"]
    roundtrip = output["prediction"].to_numpy(dtype=np.float64)
    assert not np.allclose(roundtrip, 0.0)
    assert roundtrip[0] != roundtrip[1]
    assert np.allclose(roundtrip, expected)
    assert output["engine_id"].tolist() == expected_meta["engine_id"].tolist()


def test_load_predictor_resolves_registered_lstm() -> None:
    """A registered LSTM resolves and predicts through load_predictor."""
    from turbofan.registry import (
        load_predictor,
        log_and_register,
        model_name,
        promote,
    )

    payload = _lstm_payload(window_size=3)
    records = pd.DataFrame(_records_for_engine(1, 5, feature_value=1.0))

    name = model_name("lstm", "FD003")
    with mlflow.start_run():
        version = log_and_register(payload, model_type="lstm", subset="FD003")
    promote(name, version)

    predictor = load_predictor(name)
    result = predictor.predict(records)

    assert predictor.metadata.model_type == "lstm"
    assert predictor.metadata.prediction_scope == "final_window"
    assert len(result.predictions) == 1
    assert result.predictions[0].model_type == "lstm"
