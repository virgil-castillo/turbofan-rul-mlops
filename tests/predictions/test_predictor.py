"""Tests for the registry pyfunc prediction adapter."""
from __future__ import annotations

import pandas as pd

from turbofan.data.contracts import FEATURE_COLUMNS
from turbofan.predictions.predictor import PyfuncPredictor


class _CapturingModel:
    """Test model that records the frame received from the adapter."""

    def __init__(self) -> None:
        """Initialize without a received prediction frame."""
        self.received: pd.DataFrame | None = None

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Record the model input and return a valid pyfunc prediction frame.

        Args:
            frame: Canonical records received from ``PyfuncPredictor``.

        Returns:
            Prediction rows required by the pyfunc output contract.
        """
        self.received = frame.copy()
        return frame.loc[:, ["engine_id", "cycle"]].assign(prediction=1.0)


def _integer_record() -> dict[str, object]:
    """Build a canonical record whose every feature value is an integer.

    Returns:
        Canonical raw inference record with integer feature values.
    """
    return {
        "engine_id": 1,
        "cycle": 1,
        **dict.fromkeys(FEATURE_COLUMNS, 1),
    }


def test_pyfunc_predictor_coerces_integer_features_before_model_prediction() -> None:
    """Validated integer feature values reach the model as MLflow-compatible floats."""
    model = _CapturingModel()
    predictor = PyfuncPredictor(
        model,
        model_type="ridge",
        artifact_id="turbofan-ridge-fd001/1",
    )

    result = predictor.predict([_integer_record()])

    assert len(result.predictions) == 1
    assert model.received is not None
    assert all(
        pd.api.types.is_float_dtype(model.received[column])
        for column in FEATURE_COLUMNS
    )


def test_pyfunc_predictor_allow_partial_skips_invalid_row_and_warns() -> None:
    """``allow_partial=True`` drops invalid rows and surfaces their warnings."""
    valid = _integer_record()
    invalid = {**_integer_record(), "engine_id": -1}
    model = _CapturingModel()
    predictor = PyfuncPredictor(
        model,
        model_type="ridge",
        artifact_id="turbofan-ridge-fd001/1",
    )

    result = predictor.predict([valid, invalid], allow_partial=True)

    assert result.metadata.input_rows == 2
    assert result.metadata.prediction_rows == 1
    assert len(result.predictions) == 1
    assert any("Skipped row 1" in warning for warning in result.metadata.warnings)
