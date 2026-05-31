"""Fixed-length sequence window construction."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class WindowedSequences:
    """Container for sequence windows, labels, and row metadata.

    Args:
        X: Window features with shape ``(n_windows, window_size, n_features)``.
        y: Labels with shape ``(n_windows,)``.
        metadata: Final-timestep metadata for each window. Includes a
            ``padded`` boolean column flagging windows that were left-zero
            padded because the engine was shorter than ``window_size``.
        lengths: Actual (un-padded) timestep counts per window, dtype
            ``int64``. Full-length windows carry ``length == window_size``.
    """

    X: npt.NDArray[np.float32]
    y: npt.NDArray[np.float32]
    metadata: pd.DataFrame
    lengths: npt.NDArray[np.int64]


def build_sliding_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window_size: int,
    target_col: str = "rul",
) -> WindowedSequences:
    """Build fixed-length sliding windows per engine.

    Args:
        df: Input rows containing engine, cycle, feature, and target columns.
        feature_cols: Feature column names to include in each window.
        window_size: Number of cycles per window.
        target_col: Target column used for final-timestep labels.

    Returns:
        Fixed-length sequence windows with labels and final-row metadata.

    Raises:
        KeyError: If required engine, cycle, feature, or target columns are
            missing.
        ValueError: If ``window_size`` is not positive or no windows are built.
    """
    return _build_windows(
        df=df,
        feature_cols=feature_cols,
        window_size=window_size,
        target_col=target_col,
        final_only=False,
    )


def build_final_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window_size: int,
    target_col: str | None = "rul",
) -> WindowedSequences:
    """Build one final full window per eligible engine.

    Args:
        df: Input rows containing engine, cycle, feature, and optional target
            columns.
        feature_cols: Feature column names to include in each window.
        window_size: Number of cycles per window.
        target_col: Target column used for labels, or ``None`` for unlabeled
            prediction windows.

    Returns:
        Final fixed-length sequence windows with labels and final-row metadata.

    Raises:
        KeyError: If required engine, cycle, feature, or requested target
            columns are missing.
        ValueError: If ``window_size`` is not positive or no windows are built.
    """
    return _build_windows(
        df=df,
        feature_cols=feature_cols,
        window_size=window_size,
        target_col=target_col,
        final_only=True,
    )


def _build_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window_size: int,
    target_col: str | None,
    *,
    final_only: bool,
) -> WindowedSequences:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    feature_list = list(feature_cols)
    _validate_columns(df, feature_list, target_col)

    n_features = len(feature_list)

    X_chunks: list[npt.NDArray[np.float32]] = []
    y_chunks: list[npt.NDArray[np.float32]] = []
    meta_rows: list[pd.DataFrame] = []
    length_chunks: list[npt.NDArray[np.int64]] = []

    for engine_id, group in df.groupby("engine_id", sort=True):
        group = group.sort_values("cycle").reset_index(drop=True)
        n_rows = len(group)
        features = group[feature_list].to_numpy(dtype=np.float32)

        if target_col is None:
            # Unlabeled: use NaN for all labels
            targets = np.full(n_rows, float("nan"), dtype=np.float32)
        else:
            targets = group[target_col].to_numpy(dtype=np.float32)

        if n_rows < window_size:
            # Left-zero-pad the short engine
            padded_features = np.zeros((window_size, n_features), dtype=np.float32)
            padded_features[-n_rows:, :] = features
            X_chunks.append(padded_features[np.newaxis, ...])
            y_chunks.append(np.asarray([targets[-1]], dtype=np.float32))
            length_chunks.append(np.asarray([n_rows], dtype=np.int64))
            meta_rows.append(
                pd.DataFrame(
                    {
                        "engine_id": [int(cast(int, engine_id))],
                        "cycle": [int(group["cycle"].iloc[-1])],
                        "padded": [True],
                    }
                )
            )
            continue

        if final_only:
            start_indices: range = range(n_rows - window_size, n_rows - window_size + 1)
        else:
            start_indices = range(0, n_rows - window_size + 1)

        for start in start_indices:
            end = start + window_size
            X_chunks.append(features[start:end][np.newaxis, ...])
            y_chunks.append(np.asarray([targets[end - 1]], dtype=np.float32))
            length_chunks.append(np.asarray([window_size], dtype=np.int64))
            meta_rows.append(
                pd.DataFrame(
                    {
                        "engine_id": [int(cast(int, engine_id))],
                        "cycle": [int(group["cycle"].iloc[end - 1])],
                        "padded": [False],
                    }
                )
            )

    if not X_chunks:
        raise ValueError("No eligible sequence windows could be built.")

    return WindowedSequences(
        X=np.concatenate(X_chunks, axis=0),
        y=np.concatenate(y_chunks, axis=0),
        metadata=pd.concat(meta_rows, ignore_index=True),
        lengths=np.concatenate(length_chunks, axis=0),
    )


def _validate_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str | None,
) -> None:
    """Raise ``KeyError`` if required columns are absent from ``df``.

    Args:
        df: DataFrame to validate.
        feature_cols: Feature column names that must be present.
        target_col: Target column name, or ``None`` if labels are not needed.

    Raises:
        KeyError: If any required column is missing from ``df``.
    """
    required_cols = ["engine_id", "cycle", *feature_cols]
    if target_col is not None:
        required_cols.append(target_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise KeyError(f"Missing sequence window columns: {missing}")
