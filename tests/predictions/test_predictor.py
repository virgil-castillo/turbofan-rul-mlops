"""Tests for the registry pyfunc prediction adapter."""
from __future__ import annotations

import pandas as pd
import pytest

from turbofan.data.contracts import FEATURE_COLUMNS
from turbofan.predictions.predictor import PyfuncPredictor
from turbofan.predictions.validation import SchemaValidationError


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


class _NonFrameModel:
    """Test model whose ``predict`` returns a non-DataFrame value."""

    def predict(self, frame: pd.DataFrame) -> object:
        """Return a value that violates the pyfunc output contract.

        Args:
            frame: Canonical records received from ``PyfuncPredictor``.

        Returns:
            A list rather than the required prediction DataFrame.
        """
        return list(frame.index)


class _MalformedFrameModel:
    """Test model returning a DataFrame missing required output columns."""

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame lacking the ``prediction`` column.

        Args:
            frame: Canonical records received from ``PyfuncPredictor``.

        Returns:
            A frame with identifier columns but no ``prediction`` column.
        """
        return frame.loc[:, ["engine_id", "cycle"]].copy()


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


def test_pyfunc_predictor_rejects_non_dataframe_model_output() -> None:
    """A model returning a non-DataFrame raises a descriptive ValueError."""
    predictor = PyfuncPredictor(
        _NonFrameModel(),
        model_type="ridge",
        artifact_id="turbofan-ridge-fd001/1",
    )

    with pytest.raises(
        ValueError, match="Pyfunc model predict\\(\\) must return a DataFrame\\."
    ):
        predictor.predict([_integer_record()])


def test_pyfunc_predictor_strict_validation_propagates_schema_error() -> None:
    """Strict mode surfaces ``SchemaValidationError`` for invalid input."""
    invalid = {**_integer_record(), "engine_id": -1}
    predictor = PyfuncPredictor(
        _CapturingModel(),
        model_type="ridge",
        artifact_id="turbofan-ridge-fd001/1",
    )

    with pytest.raises(SchemaValidationError, match="positive integers"):
        predictor.predict([invalid])


def test_pyfunc_predictor_partial_all_invalid_raises_schema_error() -> None:
    """Partial mode with every row invalid raises the no-valid-rows error."""
    invalid = {**_integer_record(), "engine_id": -1}
    predictor = PyfuncPredictor(
        _CapturingModel(),
        model_type="ridge",
        artifact_id="turbofan-ridge-fd001/1",
    )

    with pytest.raises(
        SchemaValidationError, match="Partial validation left no valid rows\\."
    ):
        predictor.predict([invalid], allow_partial=True)


def test_pyfunc_predictor_malformed_output_frame_raises_key_error() -> None:
    """A model output frame missing required columns fails row extraction."""
    predictor = PyfuncPredictor(
        _MalformedFrameModel(),
        model_type="ridge",
        artifact_id="turbofan-ridge-fd001/1",
    )

    with pytest.raises(KeyError):
        predictor.predict([_integer_record()])


def test_pyfunc_predictor_accepts_dataframe_records_input() -> None:
    """DataFrame input is counted and predicted through the model adapter."""
    frame = pd.DataFrame([_integer_record()])
    predictor = PyfuncPredictor(
        _CapturingModel(),
        model_type="ridge",
        artifact_id="turbofan-ridge-fd001/1",
    )

    result = predictor.predict(frame)

    assert result.metadata.input_rows == 1
    assert len(result.predictions) == 1


def test_pyfunc_predictor_metadata_property_exposes_initial_metadata() -> None:
    """The ``metadata`` property returns the model's descriptive metadata."""
    predictor = PyfuncPredictor(
        _CapturingModel(),
        model_type="ridge",
        artifact_id="turbofan-ridge-fd001/1",
    )

    metadata = predictor.metadata

    assert metadata.model_type == "ridge"
    assert metadata.artifact_id == "turbofan-ridge-fd001/1"
    assert metadata.prediction_scope == "engine"
