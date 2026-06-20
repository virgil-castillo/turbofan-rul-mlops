"""Tests for the shared inference compute in turbofan.inference.predictors."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
import torch

from turbofan.inference.predictors import (
    ridge_engine_predictions,
    sequence_final_window_predictions,
)
from turbofan.inference.schemas import FEATURE_COLUMNS, validate_raw_records
from turbofan.models.sequence_models import build_sequence_model
from turbofan.preprocessing.normalization import OperatingModeNormalizer


def _make_normalizer_payload(feature_cols: Sequence[str]) -> dict[str, object]:
    """Fit a minimal ``OperatingModeNormalizer`` and return its payload dict.

    Args:
        feature_cols: Feature columns to include in the normalizer.

    Returns:
        Payload dictionary produced by :meth:`OperatingModeNormalizer.to_payload`.
    """
    normalizer = OperatingModeNormalizer(feature_cols=list(feature_cols))
    fit_df = pd.DataFrame({col: [0.0, 1.0] for col in feature_cols})
    fit_df["op_1"] = [0.0, 0.0]
    fit_df["op_2"] = [0.0, 0.0]
    fit_df["op_3"] = [0.0, 0.0]
    normalizer.fit(fit_df)
    return normalizer.to_payload()


class _NegativeRidgePipeline:
    """Small estimator returning one negative prediction per row."""

    def predict(self, rows: pd.DataFrame) -> list[float]:
        """Return one negative prediction per row.

        Args:
            rows: Validated inference rows.

        Returns:
            Negative predictions used to verify clipping.
        """
        return [-float(index + 1) for index in range(len(rows))]


class _ConstantRidgePipeline:
    """Small estimator returning a fixed value for every row."""

    def __init__(self, value: float) -> None:
        """Store the constant prediction value.

        Args:
            value: Prediction returned for every input row.
        """
        self._value = value

    def predict(self, rows: pd.DataFrame) -> list[float]:
        """Return the constant value once per row.

        Args:
            rows: Validated inference rows.

        Returns:
            The stored constant for each row.
        """
        return [self._value for _ in range(len(rows))]


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


def _gru_payload(
    *,
    window_size: int = 3,
    max_rul: int = 125,
    bias: float | None = None,
    include_max_rul: bool = True,
) -> dict[str, object]:
    """Build a tiny GRU checkpoint payload mirroring the production format.

    Args:
        window_size: Sequence window size.
        max_rul: Maximum RUL cap stored in the checkpoint.
        bias: Optional regressor bias to set; weights are zeroed when given so
            the model output is deterministic.
        include_max_rul: Whether to include the ``max_rul`` field.

    Returns:
        Checkpoint payload mapping.
    """
    model = build_sequence_model(
        "gru",
        input_size=len(FEATURE_COLUMNS),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    if bias is not None:
        for parameter in model.parameters():
            parameter.data.zero_()
        model.regressor.bias.data.fill_(bias)
    payload: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "sequence_config": {
            "architecture": "gru",
            "window_size": window_size,
            "hidden_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "feature_cols": list(FEATURE_COLUMNS),
        "normalizer_type": "operating_mode",
        "normalizer_payload": _make_normalizer_payload(FEATURE_COLUMNS),
    }
    if include_max_rul:
        payload["max_rul"] = max_rul
    return payload


def test_ridge_engine_predictions_returns_clipped_last_cycle_per_engine() -> None:
    """Ridge compute returns one clipped prediction per engine (last cycle)."""
    records = pd.DataFrame(
        [
            _record(engine_id=1, cycle=1),
            _record(engine_id=1, cycle=2),
            _record(engine_id=1, cycle=3),
            _record(engine_id=2, cycle=1),
            _record(engine_id=2, cycle=2),
        ]
    )
    validated = validate_raw_records(records)

    metadata, predictions = ridge_engine_predictions(
        _NegativeRidgePipeline(), validated.records
    )

    assert metadata["engine_id"].tolist() == [1, 2]
    assert metadata["cycle"].tolist() == [3, 2]
    # Negative pipeline outputs are clipped to the non-negative floor.
    assert np.allclose(predictions, [0.0, 0.0])


def test_ridge_engine_predictions_caps_at_max_rul() -> None:
    """Ridge predictions above the RUL ceiling are capped at ``max_rul``."""
    records = pd.DataFrame(
        [_record(engine_id=1, cycle=1), _record(engine_id=2, cycle=1)]
    )
    validated = validate_raw_records(records)

    _, default_capped = ridge_engine_predictions(
        _ConstantRidgePipeline(500.0), validated.records
    )
    _, custom_capped = ridge_engine_predictions(
        _ConstantRidgePipeline(500.0), validated.records, max_rul=80
    )

    # 500 is clipped to the default ceiling (125) and to an explicit one (80).
    assert np.allclose(default_capped, [125.0, 125.0])
    assert np.allclose(custom_capped, [80.0, 80.0])


def test_sequence_final_window_gru_one_per_eligible_engine() -> None:
    """GRU compute returns one final-window prediction per eligible engine."""
    payload = _gru_payload(window_size=3, bias=-2.0)
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
    # Negative bias drives the raw output below zero, so clipping floors at 0.
    assert np.allclose(predictions, [0.0, 0.0])


def test_sequence_final_window_gru_rescales_output_by_max_rul() -> None:
    """GRU compute multiplies raw model output by max_rul before clipping."""
    # Zero weights with a +0.12 regressor bias yield a constant raw output of
    # 0.12; rescaling by max_rul=125 gives 15.0, distinguishable from the
    # unrescaled value and well above the clipping floor.
    payload = _gru_payload(window_size=3, max_rul=125, bias=0.12)
    records = pd.DataFrame(_records_for_engine(1, 3, feature_value=0.0))
    validated = validate_raw_records(records)

    _, predictions = sequence_final_window_predictions(payload, validated.records)

    assert len(predictions) == 1
    assert 10.0 < predictions[0] < 20.0


def test_sequence_final_window_gru_rejects_missing_max_rul() -> None:
    """GRU compute fails when the checkpoint payload omits max_rul."""
    payload = _gru_payload(include_max_rul=False)
    records = pd.DataFrame(_records_for_engine(1, 3))
    validated = validate_raw_records(records)

    with pytest.raises(KeyError):
        sequence_final_window_predictions(payload, validated.records)


def test_sequence_final_window_gru_torch_is_deterministic() -> None:
    """Repeated GRU compute on identical payload/input is bit-for-bit stable."""
    torch.manual_seed(0)
    payload = _gru_payload(window_size=3)
    records = pd.DataFrame(_records_for_engine(1, 4, feature_value=1.0))
    validated = validate_raw_records(records)

    _, first = sequence_final_window_predictions(payload, validated.records)
    _, second = sequence_final_window_predictions(payload, validated.records)

    assert np.allclose(first, second)
