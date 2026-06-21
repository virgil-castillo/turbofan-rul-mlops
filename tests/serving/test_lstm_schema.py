"""Tests for LSTM membership in the serving ModelType contract."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import get_args

from turbofan.predictions.contracts import ModelType, PredictionRow


def test_model_type_includes_lstm() -> None:
    """The serving ModelType literal admits ``lstm`` alongside ridge/gru."""
    assert set(get_args(ModelType)) == {"ridge", "gru", "lstm"}


def test_prediction_row_accepts_lstm_model_type() -> None:
    """A PredictionRow can carry ``lstm`` as its model type."""
    row = PredictionRow(
        engine_id=1,
        cycle=10,
        prediction=42.0,
        model_type="lstm",
        artifact_id="turbofan-lstm-fd001/1",
        prediction_scope="final_window",
        predicted_at=datetime.now(UTC),
    )

    assert row.model_type == "lstm"
    assert row.prediction_scope == "final_window"
