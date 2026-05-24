"""Engine-level train/validation splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_engine(
    df: pd.DataFrame,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training DataFrame into train/validation engine groups.

    Args:
        df: Training rows containing an ``engine_id`` column.
        test_size: Fraction of engines assigned to validation.
        random_seed: Seed used to shuffle engine IDs.

    Returns:
        Tuple of train rows and validation rows with reset indexes.

    Raises:
        ValueError: If ``engine_id`` is missing, ``test_size`` is outside
            ``(0, 1)``, fewer than two engines are present, or the split would
            create an empty output.
    """
    if "engine_id" not in df.columns:
        raise ValueError("DataFrame must contain an engine_id column.")
    if test_size <= 0.0 or test_size >= 1.0:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    engine_ids = np.asarray(sorted(df["engine_id"].unique()))
    if len(engine_ids) < 2:
        raise ValueError("Engine-level split requires at least two engines.")

    n_val = int(round(len(engine_ids) * test_size))
    n_val = max(1, n_val)
    if n_val >= len(engine_ids):
        raise ValueError("Validation split would leave no training engines.")

    rng = np.random.default_rng(random_seed)
    shuffled = engine_ids.copy()
    rng.shuffle(shuffled)

    val_ids = set(shuffled[:n_val].tolist())
    val_mask = df["engine_id"].isin(val_ids)
    train = df.loc[~val_mask].reset_index(drop=True)
    val = df.loc[val_mask].reset_index(drop=True)
    return train, val
