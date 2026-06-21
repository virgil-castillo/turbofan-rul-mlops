"""Tests for turbofan.models.evaluate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from turbofan.config.schema import DataConfig, ModelConfig, ProjectConfig
from turbofan.models.baseline import build_ridge_estimator
from turbofan.models.evaluate import (
    align_official_test_labels,
    clip_rul_predictions,
    evaluate_rows,
    predict_ridge_official,
    predict_with_clipping,
    select_last_cycle_per_engine,
    split_features_target,
)
from turbofan.training.split import load_and_split


class FixedPredictor:
    """Predictor returning preconfigured values."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return predictions matching the requested row count.

        Args:
            X: Feature rows.

        Returns:
            Configured predictions as a float array.
        """
        return np.asarray(self.values[: len(X)], dtype=np.float64)


def test_split_features_target_removes_only_target() -> None:
    """Feature/target split preserves non-target columns."""
    df = pd.DataFrame(
        {"engine_id": [1], "cycle": [1], "s_1": [2.0], "rul": [3.0]}
    )
    X, y = split_features_target(df)
    assert list(X.columns) == ["engine_id", "cycle", "s_1"]
    assert list(y) == [3.0]


def test_split_features_target_missing_target_raises() -> None:
    """Missing target column raises KeyError."""
    df = pd.DataFrame({"engine_id": [1], "cycle": [1], "s_1": [2.0]})
    with pytest.raises(KeyError):
        split_features_target(df)


def test_select_last_cycle_per_engine() -> None:
    """Final row per engine is selected and sorted by engine_id."""
    df = pd.DataFrame(
        {
            "engine_id": [2, 1, 1, 2],
            "cycle": [1, 1, 3, 4],
            "s_1": [20.0, 10.0, 30.0, 40.0],
        }
    )
    result = select_last_cycle_per_engine(df)
    assert list(result["engine_id"]) == [1, 2]
    assert list(result["cycle"]) == [3, 4]


def test_align_official_test_labels() -> None:
    """Official labels align to one final-cycle row per engine."""
    last_rows = pd.DataFrame({"engine_id": [1, 2], "cycle": [3, 4]})
    labels = pd.Series([50, 60], name="rul")
    aligned = align_official_test_labels(last_rows, labels)
    assert list(aligned) == [50.0, 60.0]
    assert aligned.name == "rul"


def test_align_official_test_labels_count_mismatch_raises() -> None:
    """Official label count must match final-cycle row count."""
    last_rows = pd.DataFrame({"engine_id": [1, 2], "cycle": [3, 4]})
    labels = pd.Series([50], name="rul")
    with pytest.raises(ValueError, match="label count"):
        align_official_test_labels(last_rows, labels)


def test_official_labels_align_after_last_cycle_selection() -> None:
    """Labels align after selecting sorted final-cycle test rows."""
    test_rows = pd.DataFrame(
        {
            "engine_id": [2, 1, 1, 2],
            "cycle": [1, 2, 5, 4],
        }
    )
    last_rows = select_last_cycle_per_engine(test_rows)
    aligned = align_official_test_labels(last_rows, pd.Series([100, 200]))
    assert list(last_rows["engine_id"]) == [1, 2]
    assert list(aligned) == [100.0, 200.0]


def test_evaluate_rows_clips_negative_predictions() -> None:
    """Negative predictions are clipped to zero before metrics."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [1, 2],
            "s_1": [1.0, 2.0],
            "rul": [0.0, 10.0],
        }
    )
    metrics = evaluate_rows(FixedPredictor([-5.0, 10.0]), df)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0


def test_evaluate_rows_rejects_wrong_prediction_length() -> None:
    """Prediction count must match target row count."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [1, 2],
            "s_1": [1.0, 2.0],
            "rul": [0.0, 10.0],
        }
    )
    with pytest.raises(ValueError, match="same length"):
        evaluate_rows(FixedPredictor([1.0]), df)


# ---------------------------------------------------------------------------
# clip helpers
# ---------------------------------------------------------------------------


def test_clip_rul_predictions_bounds_values() -> None:
    """Clipping bounds predictions into ``[0, max_rul]`` as float64."""
    clipped = clip_rul_predictions(
        np.array([-5.0, 10.0, 200.0], dtype=np.float64), max_rul=125
    )

    assert clipped.dtype == np.float64
    assert clipped.tolist() == [0.0, 10.0, 125.0]


def test_predict_with_clipping_clips_estimator_output() -> None:
    """Predictions are clipped into the RUL range as float64."""

    class _Est:
        def predict(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
            """Return fixed out-of-range predictions.

            Args:
                x: Ignored feature rows.

            Returns:
                Predictions spanning below 0 and above the cap.
            """
            return np.array([-1.0, 50.0, 999.0], dtype=np.float64)

    out = predict_with_clipping(
        _Est(),  # type: ignore[arg-type]
        pd.DataFrame({"a": [1, 2, 3]}),
        max_rul=125,
        label="validation",
    )

    assert out.dtype == np.float64
    assert out.tolist() == [0.0, 50.0, 125.0]


# ---------------------------------------------------------------------------
# Ridge official-test prediction
# ---------------------------------------------------------------------------


def test_predict_ridge_official_caps_predictions(data_cfg: DataConfig) -> None:
    """Official ridge eval returns one capped prediction per test engine."""
    cfg = ProjectConfig(
        project_name="t", data=data_cfg, model=ModelConfig(alpha=1.0)
    )
    frames = load_and_split(
        data_cfg, max_rul=data_cfg.max_rul, test_size=0.4, split_seed=42
    )
    x_train, y_train = split_features_target(frames.train)
    estimator = build_ridge_estimator(cfg, seed=42)
    estimator.fit(x_train, y_train)

    official = predict_ridge_official(
        data_cfg, estimator=estimator, max_rul=data_cfg.max_rul
    )

    assert len(official.y_pred) == len(official.last_rows)
    assert np.all(official.y_pred >= 0.0)
    assert np.all(official.y_pred <= data_cfg.max_rul)
