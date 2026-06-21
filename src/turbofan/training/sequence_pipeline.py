"""Shared sequence-model train/evaluate pipeline steps.

This module hosts the prepare/train/evaluate building blocks for sequence
(GRU/LSTM) models that were previously duplicated across the production
training CLI (:mod:`turbofan.cli.train_sequence`), the official-evaluation
sweep (:mod:`turbofan.benchmarks.official_jobs`), and the experiment harness
(:mod:`turbofan.experiments.feature_family_screen`). Sharing them keeps
production training, official evaluation, and experiment sweeps from drifting
apart.

Seed discipline is explicit: ``data_seed`` governs the engine train/val split
*and* the feature-pipeline ``random_state`` (the data-distribution seed), while
the model-initialisation/training seed is passed separately to
:func:`train_prepared_sequence`. The two coincide for production training but
diverge in the official-eval and screen sweeps, where the data seed is pinned
to 42 and only the model seed varies. The low-level helpers are imported at
module scope so tests can monkeypatch them on this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch import nn

from turbofan.config.schema import DataConfig, FeatureFamilyName, SequenceConfig
from turbofan.data import loader as data_loader
from turbofan.evaluation import metrics, sequence_official
from turbofan.features import pipeline as feature_pipeline
from turbofan.models import sequence_models
from turbofan.sequences import dataset, windowing
from turbofan.sequences.windowing import WindowedSequences
from turbofan.training import sequence_training, split
from turbofan.training.sequence_training import SequenceLoader, TrainingResult

#: Identifier columns carried alongside engineered features so windowing can
#: group per engine and read final-cycle targets.
_ID_COLS: list[str] = ["engine_id", "cycle", "rul"]


@dataclass(frozen=True)
class PreparedSequenceData:
    """Fitted feature pipeline plus windowed train/validation data.

    Args:
        pipeline: Fitted feature pipeline.
        feature_cols: Feature columns produced by the pipeline.
        train_windows: Sliding windows over the training split.
        val_windows: Sliding windows over the validation split.
        train_loader: Shuffled loader over ``train_windows``.
        val_loader: Sequential loader over ``val_windows``.
    """

    pipeline: Pipeline
    feature_cols: list[str]
    train_windows: WindowedSequences
    val_windows: WindowedSequences
    train_loader: SequenceLoader
    val_loader: SequenceLoader


def prepare_sequence_data(
    data_cfg: DataConfig,
    *,
    feature_families: list[FeatureFamilyName],
    windows: list[int] | None,
    lag_steps: list[int] | None,
    sensor_drop: list[str] | None,
    n_modes: int,
    data_seed: int,
    max_rul: int,
    test_size: float,
    window_size: int,
    batch_size: int,
) -> PreparedSequenceData:
    """Load, split, fit the feature pipeline, and build sequence windows/loaders.

    The engine split and the feature-pipeline ``random_state`` both use
    ``data_seed`` (a single parameter so they cannot drift apart); the model
    seed is supplied separately to :func:`train_prepared_sequence`.

    Args:
        data_cfg: Data layer config locating the raw training files.
        feature_families: Ordered feature families to engineer.
        windows: Rolling-window sizes, or ``None``.
        lag_steps: Lag offsets, or ``None``.
        sensor_drop: Sensor columns to drop before feature engineering.
        n_modes: Operating-mode count for the normalizer.
        data_seed: Seed for the engine split and pipeline ``random_state``.
        max_rul: Maximum-RUL cap for the labels.
        test_size: Fraction of engines held out for validation.
        window_size: Number of cycles per sequence window.
        batch_size: Loader batch size.

    Returns:
        The fitted pipeline and windowed train/validation data and loaders.
    """
    frames = split.load_and_split(
        data_cfg, max_rul=max_rul, test_size=test_size, split_seed=data_seed
    )
    pipeline = feature_pipeline.build_feature_pipeline(
        sensor_drop=sensor_drop,
        n_modes=n_modes,
        random_state=data_seed,
        feature_families=feature_families,
        windows=windows,
        lag_steps=lag_steps,
    )
    train_features = pipeline.fit_transform(frames.train)
    val_features = pipeline.transform(frames.val)
    feature_cols: list[str] = pipeline.named_steps["feature_engineer"].feature_cols_

    train_windows = windowing.build_sliding_windows(
        _join_ids(frames.train, train_features),
        feature_cols=feature_cols,
        window_size=window_size,
    )
    val_windows = windowing.build_sliding_windows(
        _join_ids(frames.val, val_features),
        feature_cols=feature_cols,
        window_size=window_size,
    )
    train_loader = dataset.build_sequence_loader(
        train_windows, batch_size=batch_size, shuffle=True
    )
    val_loader = dataset.build_sequence_loader(
        val_windows, batch_size=batch_size, shuffle=False
    )
    return PreparedSequenceData(
        pipeline=pipeline,
        feature_cols=feature_cols,
        train_windows=train_windows,
        val_windows=val_windows,
        train_loader=train_loader,
        val_loader=val_loader,
    )


def train_prepared_sequence(
    prepared: PreparedSequenceData,
    seq_cfg: SequenceConfig,
    *,
    device: torch.device,
    model_seed: int,
    max_rul: int,
) -> TrainingResult:
    """Seed, build, and train a sequence model on prepared data.

    ``model_seed`` governs only model initialisation and training; the data
    split and feature pipeline were already seeded in
    :func:`prepare_sequence_data`.

    Args:
        prepared: Prepared feature pipeline, windows, and loaders.
        seq_cfg: Sequence model configuration (architecture + hyperparameters).
        device: Torch device for training.
        model_seed: Seed for model initialisation and training.
        max_rul: Maximum RUL used to normalise targets and rescale predictions.

    Returns:
        The training result with the best restored model and metric history.
    """
    sequence_training.seed_everything(model_seed)
    model = sequence_models.build_sequence_model(
        seq_cfg.architecture,
        input_size=len(prepared.feature_cols),
        hidden_size=seq_cfg.hidden_size,
        num_layers=seq_cfg.num_layers,
        dropout=seq_cfg.dropout,
    )
    return sequence_training.train_sequence_model(
        model=model,
        train_loader=prepared.train_loader,
        validation_windows_loader=prepared.val_loader,
        config=seq_cfg,
        device=device,
        random_seed=model_seed,
        max_rul=max_rul,
    )


def evaluate_window_metrics(
    model: nn.Module,
    loader: SequenceLoader,
    windows: WindowedSequences,
    *,
    device: torch.device,
    max_rul: int,
) -> tuple[dict[str, float], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Predict labeled windows and compute regression metrics.

    Predictions are clipped to be non-negative and rescaled by ``max_rul``
    without an upper cap, matching the sequence training/eval contract.

    Args:
        model: Trained sequence model.
        loader: Sequential loader over ``windows``.
        windows: Labeled sequence windows.
        device: Torch device used for inference.
        max_rul: Maximum RUL used to rescale predictions.

    Returns:
        Metrics, ground-truth RUL values, and predicted RUL values.
    """
    y_pred = np.clip(
        sequence_training.predict_windows(model, loader, device, max_rul=max_rul),
        0.0,
        None,
    )
    y_true = windows.y.astype(np.float64)
    computed_metrics = metrics.regression_metrics(y_true, y_pred)
    return computed_metrics, y_true, y_pred


@dataclass(frozen=True)
class OfficialSequencePredictions:
    """Aligned official-test predictions for a sequence model.

    Args:
        windows: Final-cycle test windows (carry per-window metadata).
        y_true: Official RUL labels aligned to the eligible engines.
        y_pred: Non-negative predicted RUL values aligned to ``windows``.
    """

    windows: WindowedSequences
    y_true: pd.Series
    y_pred: npt.NDArray[np.float64]


def predict_sequence_official(
    data_cfg: DataConfig,
    *,
    pipeline: Pipeline,
    feature_cols: list[str],
    model: nn.Module,
    device: torch.device,
    window_size: int,
    batch_size: int,
    max_rul: int,
) -> OfficialSequencePredictions:
    """Evaluate a trained sequence model on the official C-MAPSS test set.

    Loads the official test/RUL files, transforms them with the fitted
    pipeline, builds one final window per eligible engine, predicts (clipped to
    be non-negative, rescaled by ``max_rul``), and aligns the official labels.

    Args:
        data_cfg: Data layer config locating the official test files.
        pipeline: Fitted feature pipeline used during training.
        feature_cols: Feature columns produced by the pipeline.
        model: Trained sequence model.
        device: Torch device used for inference.
        window_size: Number of cycles per sequence window.
        batch_size: Inference batch size.
        max_rul: Maximum RUL used to rescale predictions.

    Returns:
        The final-cycle test windows with aligned labels and predictions.

    Raises:
        FileNotFoundError: If the official test or RUL files are missing.
    """
    test_raw = data_loader.load_raw_test(data_cfg)
    rul_labels = data_loader.load_rul_labels(data_cfg)
    id_cols = [c for c in ("engine_id", "cycle") if c in test_raw.columns]
    test_features = pipeline.transform(test_raw)
    test_df = pd.concat(
        [
            test_raw[id_cols].reset_index(drop=True),
            test_features.reset_index(drop=True),
        ],
        axis=1,
    )
    test_windows = windowing.build_final_windows(
        test_df,
        feature_cols=feature_cols,
        window_size=window_size,
        target_col=None,
    )
    loader = dataset.build_sequence_loader(
        test_windows, batch_size=batch_size, shuffle=False
    )
    y_pred = np.clip(
        sequence_training.predict_windows(model, loader, device, max_rul=max_rul),
        0.0,
        None,
    )
    y_true = sequence_official.align_labels_to_eligible_engines(
        test_windows.metadata, rul_labels
    )
    return OfficialSequencePredictions(
        windows=test_windows, y_true=y_true, y_pred=y_pred
    )


def _join_ids(rows: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Concatenate identifier columns with engineered features.

    Args:
        rows: Source rows carrying ``engine_id``/``cycle``/``rul``.
        features: Engineered feature frame aligned to ``rows``.

    Returns:
        Combined frame with identifiers followed by features.
    """
    return pd.concat(
        [
            rows[_ID_COLS].reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )
