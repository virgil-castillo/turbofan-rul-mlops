"""Unit tests for the shared sequence-model train/evaluate pipeline steps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.pipeline import Pipeline

from turbofan.config.schema import DataConfig, SequenceConfig
from turbofan.features import pipeline as feature_pipeline
from turbofan.features.pipeline import build_feature_pipeline
from turbofan.models.sequence_models import SequenceRULRegressor, build_sequence_model
from turbofan.training import sequence_pipeline, sequence_training, split
from turbofan.training.sequence_training import TrainingResult


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
) -> sequence_pipeline.PreparedSequenceData:
    """Prepare tiny sequence data from the stub FD001 fixture.

    Args:
        data_cfg: Data config pointing at the stub files.
        data_seed: Seed for the split and feature pipeline.

    Returns:
        The prepared sequence data.
    """
    return sequence_pipeline.prepare_sequence_data(
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
# prepare_sequence_data — seed wiring
# ---------------------------------------------------------------------------


def test_prepare_sequence_data_uses_data_seed_for_split_and_pipeline(
    data_cfg: DataConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The data seed flows to both the engine split and the pipeline state."""
    captured: dict[str, object] = {}
    real_split = split.split_by_engine
    real_pipeline = feature_pipeline.build_feature_pipeline

    def fake_split(
        df: pd.DataFrame, *, test_size: float, random_seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured["split_seed"] = random_seed
        captured["test_size"] = test_size
        return real_split(df, test_size=test_size, random_seed=random_seed)

    def fake_pipeline(**kwargs: object) -> Pipeline:
        captured["random_state"] = kwargs["random_state"]
        return real_pipeline(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(split, "split_by_engine", fake_split)
    monkeypatch.setattr(feature_pipeline, "build_feature_pipeline", fake_pipeline)

    prepared = sequence_pipeline.prepare_sequence_data(
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
    real_seed = sequence_training.seed_everything

    def fake_seed(seed: int) -> None:
        seeds.append(seed)
        real_seed(seed)

    monkeypatch.setattr(sequence_training, "seed_everything", fake_seed)

    result = sequence_pipeline.train_prepared_sequence(
        prepared,
        _tiny_seq_cfg(),
        device=torch.device("cpu"),
        model_seed=123,
        max_rul=125,
    )

    assert isinstance(result, TrainingResult)
    assert seeds and all(seed == 123 for seed in seeds)
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
    result = sequence_pipeline.train_prepared_sequence(
        prepared,
        _tiny_seq_cfg(),
        device=torch.device("cpu"),
        model_seed=42,
        max_rul=125,
    )

    official = sequence_pipeline.predict_sequence_official(
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
        sequence_pipeline.predict_sequence_official(
            data_cfg,
            pipeline=build_feature_pipeline(),
            feature_cols=["s_1"],
            model=model,
            device=torch.device("cpu"),
            window_size=3,
            batch_size=4,
            max_rul=125,
        )
