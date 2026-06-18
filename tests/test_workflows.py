"""Unit tests for the shared train/evaluate workflow steps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import torch
from sklearn.pipeline import Pipeline

from turbofan import workflows
from turbofan.config.schema import (
    DataConfig,
    ModelConfig,
    ProjectConfig,
    SequenceConfig,
)
from turbofan.features.pipeline import build_feature_pipeline
from turbofan.models.evaluate import split_features_target
from turbofan.models.sequence_models import SequenceRULRegressor, build_sequence_model
from turbofan.models.sequence_training import TrainingResult


def _tiny_seq_cfg() -> SequenceConfig:
    """Return a minimal GRU sequence config for fast CPU training.

    Returns:
        A 1-epoch GRU config with tiny hyperparameters.
    """
    return SequenceConfig(
        architecture="gru",
        window_size=3,
        batch_size=4,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
        epochs=1,
        patience=1,
        device="cpu",
    )


def _prepare(
    data_cfg: DataConfig, *, data_seed: int = 42
) -> workflows.PreparedSequenceData:
    """Prepare tiny sequence data from the stub FD001 fixture.

    Args:
        data_cfg: Data config pointing at the stub files.
        data_seed: Seed for the split and feature pipeline.

    Returns:
        The prepared sequence data.
    """
    return workflows.prepare_sequence_data(
        data_cfg,
        feature_families=["raw"],
        windows=None,
        lag_steps=None,
        sensor_drop=None,
        n_modes=1,
        data_seed=data_seed,
        max_rul=125,
        test_size=0.4,
        window_size=3,
        batch_size=4,
    )


# ---------------------------------------------------------------------------
# load_and_split
# ---------------------------------------------------------------------------


def test_load_and_split_labels_and_splits_disjoint_engines(
    data_cfg: DataConfig,
) -> None:
    """The split adds RUL labels and partitions engines without overlap."""
    frames = workflows.load_and_split(
        data_cfg, max_rul=125, test_size=0.4, split_seed=42
    )

    assert "rul" in frames.train.columns
    assert "rul" in frames.val.columns
    train_engines = set(frames.train["engine_id"])
    val_engines = set(frames.val["engine_id"])
    assert train_engines and val_engines
    assert train_engines.isdisjoint(val_engines)
    assert frames.train["rul"].max() <= 125


# ---------------------------------------------------------------------------
# prepare_sequence_data — seed wiring
# ---------------------------------------------------------------------------


def test_prepare_sequence_data_uses_data_seed_for_split_and_pipeline(
    data_cfg: DataConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The data seed flows to both the engine split and the pipeline state."""
    captured: dict[str, object] = {}
    real_split = workflows.split_by_engine
    real_pipeline = workflows.build_feature_pipeline

    def fake_split(
        df: pd.DataFrame, *, test_size: float, random_seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured["split_seed"] = random_seed
        captured["test_size"] = test_size
        return real_split(df, test_size=test_size, random_seed=random_seed)

    def fake_pipeline(**kwargs: object) -> Pipeline:
        captured["random_state"] = kwargs["random_state"]
        return real_pipeline(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workflows, "split_by_engine", fake_split)
    monkeypatch.setattr(workflows, "build_feature_pipeline", fake_pipeline)

    prepared = workflows.prepare_sequence_data(
        data_cfg,
        feature_families=["raw"],
        windows=None,
        lag_steps=None,
        sensor_drop=None,
        n_modes=1,
        data_seed=7,
        max_rul=125,
        test_size=0.4,
        window_size=3,
        batch_size=4,
    )

    assert captured["split_seed"] == 7
    assert captured["random_state"] == 7
    assert captured["test_size"] == 0.4
    assert prepared.feature_cols
    assert prepared.train_windows.X.shape[0] > 0
    assert prepared.val_windows.X.shape[0] > 0


# ---------------------------------------------------------------------------
# train_prepared_sequence — model seed and architecture
# ---------------------------------------------------------------------------


def test_train_prepared_sequence_seeds_with_model_seed_and_builds_arch(
    data_cfg: DataConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training seeds with the model seed and builds the configured architecture."""
    prepared = _prepare(data_cfg)
    seeds: list[int] = []
    real_seed = workflows.seed_everything

    def fake_seed(seed: int) -> None:
        seeds.append(seed)
        real_seed(seed)

    monkeypatch.setattr(workflows, "seed_everything", fake_seed)

    result = workflows.train_prepared_sequence(
        prepared,
        _tiny_seq_cfg(),
        device=torch.device("cpu"),
        model_seed=123,
        max_rul=125,
    )

    assert isinstance(result, TrainingResult)
    assert seeds == [123]
    assert isinstance(result.model, SequenceRULRegressor)
    assert result.model.architecture == "gru"


# ---------------------------------------------------------------------------
# predict_sequence_official
# ---------------------------------------------------------------------------


def test_predict_sequence_official_returns_finite_aligned_predictions(
    data_cfg: DataConfig,
) -> None:
    """Official sequence eval yields non-negative predictions aligned to labels."""
    prepared = _prepare(data_cfg)
    result = workflows.train_prepared_sequence(
        prepared,
        _tiny_seq_cfg(),
        device=torch.device("cpu"),
        model_seed=42,
        max_rul=125,
    )

    official = workflows.predict_sequence_official(
        data_cfg,
        pipeline=prepared.pipeline,
        feature_cols=prepared.feature_cols,
        model=result.model,
        device=torch.device("cpu"),
        window_size=3,
        batch_size=4,
        max_rul=125,
    )

    assert len(official.y_pred) == len(official.y_true)
    assert np.all(np.isfinite(official.y_pred))
    assert np.all(official.y_pred >= 0.0)


def test_predict_sequence_official_missing_files_raises(tmp_path: Path) -> None:
    """Missing official test/RUL files surface as FileNotFoundError."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    data_cfg = DataConfig(
        raw_dir=raw_dir,
        processed_dir=tmp_path / "processed",
        interim_dir=tmp_path / "interim",
        fd_subset="FD001",
    )
    model = build_sequence_model(
        "gru", input_size=1, hidden_size=2, num_layers=1, dropout=0.0
    )

    with pytest.raises(FileNotFoundError):
        workflows.predict_sequence_official(
            data_cfg,
            pipeline=build_feature_pipeline(),
            feature_cols=["s_1"],
            model=model,
            device=torch.device("cpu"),
            window_size=3,
            batch_size=4,
            max_rul=125,
        )


# ---------------------------------------------------------------------------
# clip helpers
# ---------------------------------------------------------------------------


def test_clip_rul_predictions_bounds_values() -> None:
    """Clipping bounds predictions into ``[0, max_rul]`` as float64."""
    clipped = workflows.clip_rul_predictions(
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

    out = workflows.predict_with_clipping(
        _Est(),  # type: ignore[arg-type]
        pd.DataFrame({"a": [1, 2, 3]}),
        max_rul=125,
        label="validation",
    )

    assert out.dtype == np.float64
    assert out.tolist() == [0.0, 50.0, 125.0]


# ---------------------------------------------------------------------------
# Ridge helpers
# ---------------------------------------------------------------------------


def test_build_ridge_estimator_honors_alpha_and_seed(data_cfg: DataConfig) -> None:
    """The Ridge estimator carries the configured alpha and pipeline seed."""
    cfg = ProjectConfig(
        project_name="t", data=data_cfg, model=ModelConfig(alpha=37.0)
    )

    estimator = workflows.build_ridge_estimator(cfg, seed=11)

    assert estimator.named_steps["model"].alpha == 37.0
    normalizer = estimator.named_steps["features"].named_steps["normalizer"]
    assert normalizer.random_state == 11


def test_predict_ridge_official_caps_predictions(data_cfg: DataConfig) -> None:
    """Official ridge eval returns one capped prediction per test engine."""
    cfg = ProjectConfig(
        project_name="t", data=data_cfg, model=ModelConfig(alpha=1.0)
    )
    frames = workflows.load_and_split(
        data_cfg, max_rul=data_cfg.max_rul, test_size=0.4, split_seed=42
    )
    x_train, y_train = split_features_target(frames.train)
    estimator = workflows.build_ridge_estimator(cfg, seed=42)
    estimator.fit(x_train, y_train)

    official = workflows.predict_ridge_official(
        data_cfg, estimator=estimator, max_rul=data_cfg.max_rul
    )

    assert len(official.y_pred) == len(official.last_rows)
    assert np.all(official.y_pred >= 0.0)
    assert np.all(official.y_pred <= data_cfg.max_rul)
