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
        metadata: Final-timestep metadata for each window.
    """

    X: npt.NDArray[np.float32]
    y: npt.NDArray[np.float32]
    metadata: pd.DataFrame


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

    sorted_df = df.sort_values(["engine_id", "cycle"])
    windows: list[npt.NDArray[np.float32]] = []
    labels: list[float] = []
    metadata_rows: list[dict[str, int]] = []

    for _, group in sorted_df.groupby("engine_id", sort=True):
        if len(group) < window_size:
            continue

        starts: range | list[int]
        if final_only:
            starts = [len(group) - window_size]
        else:
            starts = range(0, len(group) - window_size + 1)

        for start in starts:
            end = start + window_size
            window = group.iloc[start:end]
            features = cast(
                npt.NDArray[np.float32],
                window.loc[:, feature_list].to_numpy(dtype=np.float32),
            )
            final_row = window.iloc[-1]

            windows.append(features)
            if target_col is None:
                labels.append(float("nan"))
            else:
                labels.append(float(final_row[target_col]))
            metadata_rows.append(
                {
                    "engine_id": int(final_row["engine_id"]),
                    "cycle": int(final_row["cycle"]),
                }
            )

    if not windows:
        raise ValueError("No eligible sequence windows could be built.")

    return WindowedSequences(
        X=np.stack(windows).astype(np.float32),
        y=np.array(labels, dtype=np.float32),
        metadata=pd.DataFrame(metadata_rows).reset_index(drop=True),
    )


def _validate_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str | None,
) -> None:
    required_cols = ["engine_id", "cycle", *feature_cols]
    if target_col is not None:
        required_cols.append(target_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise KeyError(f"Missing sequence window columns: {missing}")
