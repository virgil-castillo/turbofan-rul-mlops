"""Tests for the generalized sequence inference compute (GRU + LSTM)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from turbofan.features.pipeline import build_feature_pipeline
from turbofan.models.sequence_models import build_sequence_model
from turbofan.predictions.compute import sequence_final_window_predictions
from turbofan.preprocessing.normalization import OperatingModeNormalizer
from turbofan.serving.pyfunc_adapter import _MODEL_SCOPES
from turbofan.serving.schemas import FEATURE_COLUMNS, validate_raw_records


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
    engine_id: int = 1,
    cycle: int = 1,
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


def _sequence_payload(
    *,
    architecture: str,
    window_size: int = 3,
    max_rul: int = 125,
    bias: float | None = None,
) -> dict[str, object]:
    """Build a tiny sequence checkpoint payload mirroring the production format.

    Args:
        architecture: Recurrent architecture name stored in sequence_config.
        window_size: Sequence window size.
        max_rul: Maximum RUL cap stored in the checkpoint.
        bias: Optional regressor bias; weights are zeroed when given so output
            is deterministic.

    Returns:
        Checkpoint payload mapping.
    """
    model = build_sequence_model(
        architecture,
        input_size=len(FEATURE_COLUMNS),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    if bias is not None:
        for parameter in model.parameters():
            parameter.data.zero_()
        model.regressor.bias.data.fill_(bias)
    return {
        "model_state_dict": model.state_dict(),
        "sequence_config": {
            "architecture": architecture,
            "window_size": window_size,
            "hidden_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "feature_cols": list(FEATURE_COLUMNS),
        "normalizer_type": "operating_mode",
        "normalizer_payload": _make_normalizer_payload(FEATURE_COLUMNS),
        "max_rul": max_rul,
    }


def test_lstm_scope_is_final_window() -> None:
    """The model-scope map routes lstm to the final-window scope."""
    assert _MODEL_SCOPES["lstm"] == "final_window"


def test_sequence_final_window_lstm_roundtrips_to_frame() -> None:
    """An LSTM payload produces one final-window prediction per eligible engine."""
    payload = _sequence_payload(architecture="lstm", window_size=3, bias=-2.0)
    records = pd.DataFrame(
        [
            *_records_for_engine(2, 4, feature_value=2.0),
            *_records_for_engine(1, 3, feature_value=1.0),
        ]
    )
    validated = validate_raw_records(records)

    metadata, predictions = sequence_final_window_predictions(
        payload, validated.records
    )

    assert metadata["engine_id"].tolist() == [1, 2]
    assert metadata["cycle"].tolist() == [3, 4]
    # Negative bias drives raw output below zero, so clipping floors at 0.
    assert np.allclose(predictions, [0.0, 0.0])


def test_sequence_final_window_lstm_rescales_by_max_rul() -> None:
    """LSTM compute multiplies raw output by max_rul before clipping."""
    payload = _sequence_payload(
        architecture="lstm", window_size=3, max_rul=125, bias=0.12
    )
    records = pd.DataFrame(_records_for_engine(1, 3, feature_value=0.0))
    validated = validate_raw_records(records)

    _, predictions = sequence_final_window_predictions(payload, validated.records)

    assert len(predictions) == 1
    assert 10.0 < predictions[0] < 20.0


def test_sequence_final_window_gru_still_works() -> None:
    """A GRU payload still round-trips through the generalized function."""
    payload = _sequence_payload(architecture="gru", window_size=3, bias=-2.0)
    records = pd.DataFrame(_records_for_engine(1, 3, feature_value=1.0))
    validated = validate_raw_records(records)

    metadata, predictions = sequence_final_window_predictions(
        payload, validated.records
    )

    assert metadata["engine_id"].tolist() == [1]
    assert np.allclose(predictions, [0.0])


def test_sequence_final_window_applies_fitted_feature_pipeline() -> None:
    """Engineered-feature payloads transform raw records before windowing."""
    raw_frame = pd.DataFrame(_records_for_engine(1, 4, feature_value=1.0))
    pipeline = build_feature_pipeline(
        feature_families=["raw", "rolling_slope"],
        windows=[2],
    )
    pipeline.fit(raw_frame)
    feature_cols = pipeline.named_steps["feature_engineer"].feature_cols_
    model = build_sequence_model(
        "gru",
        input_size=len(feature_cols),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    model.regressor.bias.data.fill_(0.1)
    payload = {
        "model_state_dict": model.state_dict(),
        "sequence_config": {
            "architecture": "gru",
            "window_size": 3,
            "hidden_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "feature_cols": feature_cols,
        "feature_pipeline": pipeline,
        "normalizer_type": "operating_mode",
        "normalizer_payload": _make_normalizer_payload(FEATURE_COLUMNS),
        "max_rul": 125,
    }

    validated = validate_raw_records(raw_frame)
    metadata, predictions = sequence_final_window_predictions(
        payload, validated.records
    )

    assert metadata["engine_id"].tolist() == [1]
    assert metadata["cycle"].tolist() == [4]
    assert np.allclose(predictions, [12.5])


def test_sequence_final_window_unknown_architecture_raises() -> None:
    """An unsupported architecture in the payload raises a clear ValueError."""
    payload = _sequence_payload(architecture="gru", window_size=3)
    sequence_config = payload["sequence_config"]
    assert isinstance(sequence_config, dict)
    sequence_config["architecture"] = "transformer"
    records = pd.DataFrame(_records_for_engine(1, 3))
    validated = validate_raw_records(records)

    with pytest.raises(ValueError, match="transformer"):
        sequence_final_window_predictions(payload, validated.records)
