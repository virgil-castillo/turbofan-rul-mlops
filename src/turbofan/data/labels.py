"""RUL label computation for C-MAPSS turbofan training data."""
from __future__ import annotations

import pandas as pd


def compute_rul_labels(df: pd.DataFrame, max_rul: int = 125) -> pd.Series:  # type: ignore[type-arg]
    """Compute piecewise-linear RUL labels for training data.

    For each engine: RUL = min(max_rul, max_cycle - current_cycle).
    Returns a Series aligned to df.index.

    Args:
        df: DataFrame with columns ``engine_id`` and ``cycle``.
        max_rul: Maximum RUL cap. Default 125 matches FD001 literature.

    Returns:
        pd.Series of integer RUL values with the same index as ``df``,
        named ``"rul"``.
    """
    max_cycles: pd.Series = df.groupby("engine_id")["cycle"].transform("max")  # type: ignore[assignment]
    rul: pd.Series = (max_cycles - df["cycle"]).clip(upper=max_rul)  # type: ignore[assignment]
    return rul.rename("rul")
