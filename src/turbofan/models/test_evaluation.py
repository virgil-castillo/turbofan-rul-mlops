"""Shared official test set evaluation for sequence models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from turbofan.config.schema import DataConfig
from turbofan.data.loader import load_raw_test, load_rul_labels
from turbofan.models.evaluate import align_official_test_labels
from turbofan.models.gru import GRURULRegressor
from turbofan.models.metrics import regression_metrics
from turbofan.models.sequence_training import predict_windows
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.normalize import SequenceNormalizer
from turbofan.sequences.windowing import build_final_windows


def align_labels_to_eligible_engines(
    metadata: pd.DataFrame,
    rul_labels: pd.Series,
) -> pd.Series:
    """Align official RUL labels to eligible sequence test engines.

    C-MAPSS official RUL labels are ordered by engine ID, while final
    sequence windows can skip engines shorter than the window size. This
    selects labels for the eligible engine IDs before applying the
    standard count check.

    Args:
        metadata: Final-window metadata containing eligible ``engine_id``
            rows.
        rul_labels: Official RUL labels in full test engine order.

    Returns:
        Float RUL Series aligned to ``metadata``.

    Raises:
        ValueError: If an eligible engine ID cannot be mapped to a label
            row.
    """
    engine_ids = metadata["engine_id"].to_numpy(dtype=np.int64)
    label_positions = engine_ids - 1
    if np.any(label_positions < 0) or np.any(label_positions >= len(rul_labels)):
        raise ValueError(
            "Official RUL labels must include a row for every eligible test engine."
        )

    eligible_labels = rul_labels.iloc[label_positions].reset_index(drop=True)
    return align_official_test_labels(
        metadata.reset_index(drop=True), eligible_labels
    )


def evaluate_test_from_df(
    test_df: pd.DataFrame,
    rul_labels: pd.Series,
    model: GRURULRegressor,
    normalizer: SequenceNormalizer,
    feature_cols: list[str],
    device: torch.device,
    window_size: int,
    batch_size: int,
    max_rul: int,
) -> dict[str, float]:
    """Evaluate a trained model on pre-loaded test data.

    Normalizes the test DataFrame, builds final windows, predicts,
    aligns labels, and computes regression metrics. Use this when
    the test DataFrame has already been loaded and optionally
    feature-engineered (e.g. rolling features applied).

    Args:
        test_df: Raw or feature-engineered test DataFrame.
        rul_labels: Official RUL labels in full test engine order.
        model: Trained GRU model.
        normalizer: Fitted normalizer (trained on training data).
        feature_cols: Feature columns matching the model's input.
        device: Torch device for inference.
        window_size: Sequence window size.
        batch_size: Inference batch size.
        max_rul: Maximum RUL cap for prediction rescaling.

    Returns:
        Dict with ``test_rmse``, ``test_mae``, ``test_phm08_score``.
    """
    test_normalized = normalizer.transform(test_df)
    test_windows = build_final_windows(
        test_normalized,
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
    metrics = regression_metrics(y_true, y_pred)
    return {
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_phm08_score": metrics["phm08_score"],
    }


def evaluate_official_test(
    data_config: DataConfig,
    model: GRURULRegressor,
    normalizer: SequenceNormalizer,
    feature_cols: list[str],
    device: torch.device,
    window_size: int,
    batch_size: int,
) -> dict[str, float] | None:
    """Evaluate a trained model on official C-MAPSS test files.

    Loads the test data and RUL labels from paths derived from
    ``data_config``, then delegates to :func:`evaluate_test_from_df`.

    Args:
        data_config: Data config for file paths and ``max_rul``.
        model: Trained GRU model.
        normalizer: Fitted normalizer (trained on training data).
        feature_cols: Feature columns matching the model's input.
        device: Torch device for inference.
        window_size: Sequence window size.
        batch_size: Inference batch size.

    Returns:
        Dict with ``test_rmse``, ``test_mae``, ``test_phm08_score``,
        or ``None`` when test files are missing.
    """
    try:
        test_raw = load_raw_test(data_config)
        rul_labels = load_rul_labels(data_config)
    except FileNotFoundError:
        return None

    return evaluate_test_from_df(
        test_df=test_raw,
        rul_labels=rul_labels,
        model=model,
        normalizer=normalizer,
        feature_cols=feature_cols,
        device=device,
        window_size=window_size,
        batch_size=batch_size,
        max_rul=data_config.max_rul,
    )
