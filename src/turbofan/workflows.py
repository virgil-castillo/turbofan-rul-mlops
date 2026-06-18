"""Shared train/evaluate workflow steps.

This module hosts the train/evaluate building blocks that were previously
duplicated across the production training CLIs
(:mod:`turbofan.cli.train_sequence`, :mod:`turbofan.cli.train_baseline`), the
official-evaluation sweep (:mod:`turbofan.cli.export_official_eval`), and the
experiment harness (:mod:`turbofan.experiments.feature_family_screen`). Sharing
them keeps production training, official evaluation, and experiment sweeps from
drifting apart.

Seed discipline is explicit: ``data_seed`` governs the engine train/val split
*and* the feature-pipeline ``random_state`` (the data-distribution seed), while
the model-initialisation/training seed is passed separately to
:func:`train_prepared_sequence`. The two coincide for production training but
diverge in the official-eval and screen sweeps, where the data seed is pinned to
42 and only the model seed varies. The low-level helpers are imported at module
scope so tests can monkeypatch them on this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch import nn

from turbofan.config.schema import (
    DataConfig,
    FeatureFamilyName,
    ProjectConfig,
    SequenceConfig,
)
from turbofan.data.loader import load_raw_test, load_raw_train, load_rul_labels
from turbofan.features.pipeline import build_feature_pipeline
from turbofan.models.baseline import build_baseline_pipeline
from turbofan.models.evaluate import (
    add_rul_column,
    align_official_test_labels,
    select_last_cycle_per_engine,
)
from turbofan.models.metrics import regression_metrics
from turbofan.models.sequence_models import build_sequence_model
from turbofan.models.sequence_training import (
    SequenceLoader,
    TrainingResult,
    predict_windows,
    seed_everything,
    train_sequence_model,
)
from turbofan.models.split import split_by_engine
from turbofan.models.test_evaluation import align_labels_to_eligible_engines
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.windowing import (
    WindowedSequences,
    build_final_windows,
    build_sliding_windows,
)
from turbofan.utils.logging import get_logger

logger = get_logger(__name__)

#: Identifier columns carried alongside engineered features so windowing can
#: group per engine and read final-cycle targets.
_ID_COLS: list[str] = ["engine_id", "cycle", "rul"]


@dataclass(frozen=True)
class SplitFrames:
    """A labeled train/validation engine split.

    Args:
        train: Training rows with a computed ``rul`` column.
        val: Validation rows with a computed ``rul`` column.
    """

    train: pd.DataFrame
    val: pd.DataFrame


def load_and_split(
    data_cfg: DataConfig,
    *,
    max_rul: int,
    test_size: float,
    split_seed: int,
) -> SplitFrames:
    """Load raw training data, add RUL labels, and split by engine.

    Args:
        data_cfg: Data layer config locating the raw training files.
        max_rul: Maximum-RUL cap for the piecewise-linear labels.
        test_size: Fraction of engines held out for validation.
        split_seed: Random seed for the engine train/val split.

    Returns:
        The labeled train/validation split.
    """
    train_raw = load_raw_train(data_cfg)
    train_labeled = add_rul_column(train_raw, max_rul=max_rul)
    train_df, val_df = split_by_engine(
        train_labeled, test_size=test_size, random_seed=split_seed
    )
    return SplitFrames(train=train_df, val=val_df)


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
    frames = load_and_split(
        data_cfg, max_rul=max_rul, test_size=test_size, split_seed=data_seed
    )
    pipeline = build_feature_pipeline(
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

    train_windows = build_sliding_windows(
        _join_ids(frames.train, train_features),
        feature_cols=feature_cols,
        window_size=window_size,
    )
    val_windows = build_sliding_windows(
        _join_ids(frames.val, val_features),
        feature_cols=feature_cols,
        window_size=window_size,
    )
    train_loader = build_sequence_loader(
        train_windows, batch_size=batch_size, shuffle=True
    )
    val_loader = build_sequence_loader(
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
    seed_everything(model_seed)
    model = build_sequence_model(
        seq_cfg.architecture,
        input_size=len(prepared.feature_cols),
        hidden_size=seq_cfg.hidden_size,
        num_layers=seq_cfg.num_layers,
        dropout=seq_cfg.dropout,
    )
    return train_sequence_model(
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
        predict_windows(model, loader, device, max_rul=max_rul), 0.0, None
    )
    y_true = windows.y.astype(np.float64)
    metrics = regression_metrics(y_true, y_pred)
    return metrics, y_true, y_pred


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
    test_raw = load_raw_test(data_cfg)
    rul_labels = load_rul_labels(data_cfg)
    id_cols = [c for c in ("engine_id", "cycle") if c in test_raw.columns]
    test_features = pipeline.transform(test_raw)
    test_df = pd.concat(
        [
            test_raw[id_cols].reset_index(drop=True),
            test_features.reset_index(drop=True),
        ],
        axis=1,
    )
    test_windows = build_final_windows(
        test_df,
        feature_cols=feature_cols,
        window_size=window_size,
        target_col=None,
    )
    loader = build_sequence_loader(
        test_windows, batch_size=batch_size, shuffle=False
    )
    y_pred = np.clip(
        predict_windows(model, loader, device, max_rul=max_rul), 0.0, None
    )
    y_true = align_labels_to_eligible_engines(test_windows.metadata, rul_labels)
    return OfficialSequencePredictions(
        windows=test_windows, y_true=y_true, y_pred=y_pred
    )


def clip_rul_predictions(
    values: npt.ArrayLike,
    max_rul: int,
) -> npt.NDArray[np.float64]:
    """Clip raw predictions into ``[0, max_rul]`` as float64.

    Args:
        values: Raw model predictions.
        max_rul: Maximum allowed RUL value.

    Returns:
        Float64 predictions clipped to ``[0, max_rul]``.
    """
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, float(max_rul))


def predict_with_clipping(
    estimator: Pipeline,
    rows: pd.DataFrame,
    *,
    max_rul: int,
    label: str,
) -> npt.NDArray[np.float64]:
    """Predict rows, log the raw prediction range, and clip to valid RUL bounds.

    Args:
        estimator: Fitted sklearn estimator.
        rows: Feature rows to predict.
        max_rul: Maximum allowed RUL value.
        label: Human-readable prediction-set label for the debug log line.

    Returns:
        Float64 predictions clipped to ``[0, max_rul]``.
    """
    raw = np.asarray(estimator.predict(rows), dtype=np.float64)
    logger.debug(
        "%s raw prediction min/max: %.6f/%.6f", label, raw.min(), raw.max()
    )
    return clip_rul_predictions(raw, max_rul=max_rul)


def build_ridge_estimator(cfg: ProjectConfig, *, seed: int) -> Pipeline:
    """Build the unfitted Ridge feature-plus-model pipeline for a config.

    Args:
        cfg: Loaded project config (model + feature settings).
        seed: KMeans-normalizer ``random_state`` for the pipeline.

    Returns:
        The unfitted Ridge sklearn pipeline.
    """
    rf = cfg.features.for_model("ridge")
    return build_baseline_pipeline(
        model_name=cfg.model.name,
        alpha=cfg.model.alpha,
        feature_families=rf.feature_families,
        windows=rf.windows,
        lag_steps=rf.lag_steps,
        sensor_drop=cfg.features.sensor_cols_to_drop or None,
        n_modes=cfg.features.n_modes,
        random_state=seed,
    )


@dataclass(frozen=True)
class OfficialRidgePredictions:
    """Aligned official-test predictions for a Ridge model.

    Args:
        last_rows: One final-cycle row per test engine.
        y_true: Official RUL labels aligned to ``last_rows``.
        y_pred: Final-cycle predicted RUL values, clipped to ``[0, max_rul]``.
    """

    last_rows: pd.DataFrame
    y_true: pd.Series
    y_pred: npt.NDArray[np.float64]


def predict_ridge_official(
    data_cfg: DataConfig,
    *,
    estimator: Pipeline,
    max_rul: int,
) -> OfficialRidgePredictions:
    """Evaluate a fitted Ridge estimator on the official C-MAPSS test set.

    Predicts over each engine's full trajectory (so rolling/lag features keep
    their context), clips to ``[0, max_rul]``, then selects the final cycle per
    engine to compare against the official labels.

    Args:
        data_cfg: Data layer config locating the official test files.
        estimator: Fitted Ridge pipeline.
        max_rul: Maximum-RUL ceiling for clipping predictions.

    Returns:
        The final-cycle rows with aligned labels and clipped predictions.

    Raises:
        FileNotFoundError: If the official test or RUL files are missing.
    """
    test_raw = load_raw_test(data_cfg)
    rul_labels = load_rul_labels(data_cfg)
    last_rows = select_last_cycle_per_engine(test_raw)
    y_true = align_official_test_labels(last_rows, rul_labels)
    all_pred = predict_with_clipping(
        estimator, test_raw, max_rul=max_rul, label="official_test"
    )
    pred_rows = test_raw[["engine_id", "cycle"]].copy()
    pred_rows["prediction"] = all_pred
    last_pred = select_last_cycle_per_engine(pred_rows)
    y_pred = last_pred["prediction"].to_numpy(dtype=np.float64)
    return OfficialRidgePredictions(
        last_rows=last_rows, y_true=y_true, y_pred=y_pred
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
