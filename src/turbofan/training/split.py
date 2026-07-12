"""Engine-level train/validation splitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from turbofan.config.schema import DataConfig
from turbofan.data import labels
from turbofan.data import loader as data_loader


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
    train_raw = data_loader.load_raw_train(data_cfg)
    train_labeled = labels.add_rul_column(train_raw, max_rul=max_rul)
    train_df, val_df = split_by_engine(
        train_labeled, test_size=test_size, random_seed=split_seed
    )
    return SplitFrames(train=train_df, val=val_df)


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
