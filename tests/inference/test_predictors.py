"""Tests for turbofan.inference.predictors."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import pandas as pd
import pytest
import torch

from turbofan.inference.schemas import (
    FEATURE_COLUMNS,
    PredictionResult,
    PredictionRow,
    SchemaValidationError,
)
from turbofan.models.gru import GRURULRegressor


class _NegativeRidgePipeline:
    """Small joblib-serializable estimator returning negative predictions."""

    def predict(self, rows: pd.DataFrame) -> list[float]:
        """Return one negative prediction per row.

        Args:
            rows: Validated inference rows.

        Returns:
            Negative predictions used to verify clipping.
        """
        return [-float(index + 1) for index in range(len(rows))]


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


def _write_manifest(
    directory: Path,
    *,
    model_type: str,
    artifact_id: str,
    prediction_scope: str,
    model_path: str,
) -> Path:
    """Write a model manifest for a synthetic artifact.

    Args:
        directory: Artifact directory.
        model_type: Model family identifier.
        artifact_id: Stable artifact identifier.
        prediction_scope: Prediction scope identifier.
        model_path: Manifest-relative model path.

    Returns:
        Path to the written manifest.
    """
    manifest_path = directory / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": model_type,
                "artifact_id": artifact_id,
                "prediction_scope": prediction_scope,
                "model_path": model_path,
            }
        )
    )
    return manifest_path


def _ridge_artifact(tmp_path: Path) -> Path:
    """Create a synthetic Ridge artifact.

    Args:
        tmp_path: Temporary test directory.

    Returns:
        Path to the manifest.
    """
    artifact_dir = tmp_path / "ridge"
    artifact_dir.mkdir()
    joblib.dump(_NegativeRidgePipeline(), artifact_dir / "model.joblib")
    return _write_manifest(
        artifact_dir,
        model_type="ridge",
        artifact_id="ridge-test",
        prediction_scope="engine",
        model_path="model.joblib",
    )


def _gru_artifact(
    tmp_path: Path, *, window_size: int = 3, max_rul: int = 125
) -> Path:
    """Create a synthetic GRU artifact.

    Args:
        tmp_path: Temporary test directory.
        window_size: Serialized sequence window size.
        max_rul: Maximum RUL cap stored in the checkpoint.

    Returns:
        Path to the manifest.
    """
    artifact_dir = tmp_path / "gru"
    artifact_dir.mkdir()
    model = GRURULRegressor(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    model.regressor.bias.data.fill_(-2.0)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sequence_config": {
                "architecture": "gru",
                "window_size": window_size,
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "feature_cols": FEATURE_COLUMNS,
            "normalizer_means": {column: 0.0 for column in FEATURE_COLUMNS},
            "normalizer_stds": {column: 1.0 for column in FEATURE_COLUMNS},
            "max_rul": max_rul,
        },
        artifact_dir / "model.pt",
    )
    return _write_manifest(
        artifact_dir,
        model_type="gru",
        artifact_id="gru-test",
        prediction_scope="final_window",
        model_path="model.pt",
    )


def _assert_prediction_rows(
    rows: Sequence[PredictionRow],
    *,
    artifact_id: str,
    model_type: str,
    prediction_scope: str,
) -> None:
    """Assert stable metadata on prediction rows.

    Args:
        rows: Prediction result rows.
        artifact_id: Expected artifact identifier.
        model_type: Expected model family.
        prediction_scope: Expected prediction scope.
    """
    assert rows
    for row in rows:
        assert isinstance(row, PredictionRow)
        assert row.prediction >= 0.0
        assert row.artifact_id == artifact_id
        assert row.model_type == model_type
        assert row.prediction_scope == prediction_scope
        assert row.predicted_at.tzinfo is not None


def _assert_response_metadata(
    result: PredictionResult,
    *,
    artifact_id: str,
    model_type: str,
    prediction_scope: str,
    input_rows: int,
    prediction_rows: int,
    warnings: list[str],
) -> None:
    """Assert response-level prediction metadata.

    Args:
        result: Prediction response object.
        artifact_id: Expected artifact identifier.
        model_type: Expected model family.
        prediction_scope: Expected prediction scope.
        input_rows: Expected raw input row count.
        prediction_rows: Expected prediction row count.
        warnings: Expected response warnings.
    """
    assert isinstance(result, PredictionResult)
    assert result.metadata.artifact_id == artifact_id
    assert result.metadata.model_type == model_type
    assert result.metadata.prediction_scope == prediction_scope
    assert result.metadata.input_rows == input_rows
    assert result.metadata.prediction_rows == prediction_rows
    assert result.metadata.warnings == warnings


def test_load_predictor_returns_ridge_prediction_per_engine_last_cycle(
    tmp_path: Path,
) -> None:
    """Ridge predictor returns one clipped prediction per engine (last cycle)."""
    from turbofan.inference.predictors import load_predictor

    predictor = load_predictor(_ridge_artifact(tmp_path))
    records = pd.DataFrame(
        [
            _record(engine_id=1, cycle=1),
            _record(engine_id=1, cycle=2),
            _record(engine_id=1, cycle=3),
            _record(engine_id=2, cycle=1),
            _record(engine_id=2, cycle=2),
        ]
    )

    result = predictor.predict(records)

    prediction_tuples = [
        (row.engine_id, row.cycle, row.prediction) for row in result.predictions
    ]
    assert prediction_tuples == [
        (1, 3, 0.0),
        (2, 2, 0.0),
    ]
    _assert_prediction_rows(
        result.predictions,
        artifact_id="ridge-test",
        model_type="ridge",
        prediction_scope="engine",
    )
    _assert_response_metadata(
        result,
        artifact_id="ridge-test",
        model_type="ridge",
        prediction_scope="engine",
        input_rows=5,
        prediction_rows=2,
        warnings=[],
    )


def test_load_predictor_returns_gru_prediction_for_each_eligible_engine(
    tmp_path: Path,
) -> None:
    """GRU predictor returns one final-window prediction per eligible engine."""
    from turbofan.inference.predictors import load_predictor

    predictor = load_predictor(_gru_artifact(tmp_path, window_size=3))
    records = pd.DataFrame(
        [
            *_records_for_engine(2, 4, feature_value=2.0),
            *_records_for_engine(1, 3, feature_value=1.0),
        ]
    )

    result = predictor.predict(records)

    prediction_tuples = [
        (row.engine_id, row.cycle, row.prediction) for row in result.predictions
    ]
    assert prediction_tuples == [
        (1, 3, 0.0),
        (2, 4, 0.0),
    ]
    _assert_prediction_rows(
        result.predictions,
        artifact_id="gru-test",
        model_type="gru",
        prediction_scope="final_window",
    )
    _assert_response_metadata(
        result,
        artifact_id="gru-test",
        model_type="gru",
        prediction_scope="final_window",
        input_rows=7,
        prediction_rows=2,
        warnings=[],
    )


def test_gru_predictor_strict_mode_fails_for_short_engines(tmp_path: Path) -> None:
    """GRU strict mode rejects inputs containing engines shorter than a window."""
    from turbofan.inference.predictors import load_predictor

    predictor = load_predictor(_gru_artifact(tmp_path, window_size=3))
    records = pd.DataFrame(
        [
            *_records_for_engine(1, 3),
            *_records_for_engine(2, 2),
        ]
    )

    with pytest.raises(SchemaValidationError, match="shorter than window_size"):
        predictor.predict(records)


def test_gru_predictor_partial_mode_warns_and_skips_short_engines(
    tmp_path: Path,
) -> None:
    """GRU partial mode skips short engines and warns about each skip."""
    from turbofan.inference.predictors import load_predictor

    predictor = load_predictor(_gru_artifact(tmp_path, window_size=3))
    records = pd.DataFrame(
        [
            *_records_for_engine(1, 3),
            *_records_for_engine(2, 2),
        ]
    )

    result = predictor.predict(records, allow_partial=True)

    assert [(row.engine_id, row.cycle) for row in result.predictions] == [(1, 3)]
    assert len(result.metadata.warnings) == 1
    assert "engine 2" in result.metadata.warnings[0]
    assert result.metadata.input_rows == 5
    assert result.metadata.prediction_rows == 1


def test_gru_predictor_partial_mode_fails_when_no_predictions_remain(
    tmp_path: Path,
) -> None:
    """GRU partial mode still fails when every engine is too short."""
    from turbofan.inference.predictors import load_predictor

    predictor = load_predictor(_gru_artifact(tmp_path, window_size=3))
    records = pd.DataFrame(_records_for_engine(1, 2))

    with pytest.raises(SchemaValidationError, match="No eligible sequence windows"):
        predictor.predict(records, allow_partial=True)


def test_gru_predictor_rescales_output_by_max_rul(tmp_path: Path) -> None:
    """GRU predictor multiplies raw model output by max_rul before clipping."""
    from turbofan.inference.predictors import load_predictor

    # Build an artifact with a model that produces a known positive raw output.
    # All weights are zero and the regressor bias is set to +0.12, so the model
    # always outputs 0.12 regardless of input. After rescaling by max_rul=125
    # the prediction is 0.12 * 125 = 15.0, which is well above the clipping
    # floor of 0.0 and therefore distinguishable from an unrescaled result.
    max_rul = 125
    artifact_dir = tmp_path / "gru_rescale"
    artifact_dir.mkdir()
    window_size = 3
    model = GRURULRegressor(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    model.regressor.bias.data.fill_(0.12)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sequence_config": {
                "architecture": "gru",
                "window_size": window_size,
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "feature_cols": FEATURE_COLUMNS,
            "normalizer_means": {column: 0.0 for column in FEATURE_COLUMNS},
            "normalizer_stds": {column: 1.0 for column in FEATURE_COLUMNS},
            "max_rul": max_rul,
        },
        artifact_dir / "model.pt",
    )
    manifest = _write_manifest(
        artifact_dir,
        model_type="gru",
        artifact_id="gru-rescale-test",
        prediction_scope="final_window",
        model_path="model.pt",
    )

    predictor = load_predictor(manifest)
    records = pd.DataFrame(_records_for_engine(1, window_size, feature_value=0.0))

    result = predictor.predict(records)

    assert len(result.predictions) == 1
    prediction = result.predictions[0].prediction
    # raw output = 0.12; rescaled = 0.12 * 125 = 15.0
    assert prediction > 10.0, f"Expected rescaled prediction > 10, got {prediction}"
    assert prediction < 20.0, f"Expected rescaled prediction < 20, got {prediction}"


def test_gru_predictor_rejects_checkpoint_without_max_rul(tmp_path: Path) -> None:
    """GRU predictor fails to load when checkpoint is missing max_rul."""
    from turbofan.inference.predictors import load_predictor

    artifact_dir = tmp_path / "gru_no_maxrul"
    artifact_dir.mkdir()
    model = GRURULRegressor(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sequence_config": {
                "architecture": "gru",
                "window_size": 3,
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "feature_cols": FEATURE_COLUMNS,
            "normalizer_means": {column: 0.0 for column in FEATURE_COLUMNS},
            "normalizer_stds": {column: 1.0 for column in FEATURE_COLUMNS},
        },
        artifact_dir / "model.pt",
    )
    _write_manifest(
        artifact_dir,
        model_type="gru",
        artifact_id="gru-no-maxrul",
        prediction_scope="final_window",
        model_path="model.pt",
    )

    with pytest.raises(ValueError, match="max_rul"):
        load_predictor(artifact_dir / "model_manifest.json")


def test_ridge_predictor_partial_mode_returns_validation_warnings(
    tmp_path: Path,
) -> None:
    """Ridge partial mode returns validation warnings in response metadata."""
    from turbofan.inference.predictors import load_predictor

    predictor = load_predictor(_ridge_artifact(tmp_path))
    bad_record = _record(engine_id=1, cycle=2)
    bad_record["s_21"] = "bad"
    records = pd.DataFrame(
        [
            _record(engine_id=1, cycle=1),
            bad_record,
        ]
    )

    result = predictor.predict(records, allow_partial=True)

    assert [(row.engine_id, row.cycle) for row in result.predictions] == [(1, 1)]
    assert result.metadata.input_rows == 2
    assert result.metadata.prediction_rows == 1
    assert len(result.metadata.warnings) == 1
    assert "row 1" in result.metadata.warnings[0]
    assert "numeric and finite" in result.metadata.warnings[0]
