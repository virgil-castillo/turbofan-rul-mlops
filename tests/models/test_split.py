"""Tests for turbofan.models.split."""
from __future__ import annotations

import pandas as pd
import pytest

from turbofan.config.schema import DataConfig
from turbofan.models.split import load_and_split, split_by_engine


def test_load_and_split_labels_and_splits_disjoint_engines(
    data_cfg: DataConfig,
) -> None:
    """The split adds RUL labels and partitions engines without overlap."""
    frames = load_and_split(data_cfg, max_rul=125, test_size=0.4, split_seed=42)

    assert "rul" in frames.train.columns
    assert "rul" in frames.val.columns
    train_engines = set(frames.train["engine_id"])
    val_engines = set(frames.val["engine_id"])
    assert train_engines and val_engines
    assert train_engines.isdisjoint(val_engines)
    assert frames.train["rul"].max() <= 125


def _make_df(n_engines: int = 5, n_cycles: int = 3) -> pd.DataFrame:
    """Build synthetic engine-cycle rows."""
    rows = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, n_cycles + 1):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "s_1": float(engine_id * cycle),
                }
            )
    return pd.DataFrame(rows)


def test_split_preserves_all_rows() -> None:
    """Train and validation outputs contain every input row once."""
    df = _make_df()
    train, val = split_by_engine(df, test_size=0.4, random_seed=42)
    assert len(train) + len(val) == len(df)
    combined_keys = set(zip(train["engine_id"], train["cycle"], strict=True)) | set(
        zip(val["engine_id"], val["cycle"], strict=True)
    )
    expected_keys = set(zip(df["engine_id"], df["cycle"], strict=True))
    assert combined_keys == expected_keys


def test_no_engine_leakage_between_splits() -> None:
    """No engine_id appears in both train and validation outputs."""
    df = _make_df()
    train, val = split_by_engine(df, test_size=0.4, random_seed=42)
    assert set(train["engine_id"]).isdisjoint(set(val["engine_id"]))


def test_split_is_deterministic_for_seed() -> None:
    """Same seed produces the same engine assignment."""
    df = _make_df()
    train_a, val_a = split_by_engine(df, test_size=0.4, random_seed=7)
    train_b, val_b = split_by_engine(df, test_size=0.4, random_seed=7)
    assert list(train_a["engine_id"]) == list(train_b["engine_id"])
    assert list(val_a["engine_id"]) == list(val_b["engine_id"])


def test_split_resets_indexes() -> None:
    """Returned DataFrames have clean RangeIndex values."""
    df = _make_df()
    train, val = split_by_engine(df, test_size=0.4, random_seed=42)
    assert list(train.index) == list(range(len(train)))
    assert list(val.index) == list(range(len(val)))


def test_requires_engine_id_column() -> None:
    """Missing engine_id raises ValueError."""
    df = pd.DataFrame({"cycle": [1, 2], "s_1": [1.0, 2.0]})
    with pytest.raises(ValueError, match="engine_id"):
        split_by_engine(df, test_size=0.5, random_seed=42)


def test_requires_at_least_two_engines() -> None:
    """A validation split needs at least two engines."""
    df = _make_df(n_engines=1)
    with pytest.raises(ValueError, match="at least two"):
        split_by_engine(df, test_size=0.5, random_seed=42)


def test_rejects_test_size_that_consumes_all_engines() -> None:
    """Validation split cannot consume all engines."""
    df = _make_df(n_engines=2)
    with pytest.raises(ValueError, match="no training engines"):
        split_by_engine(df, test_size=0.75, random_seed=42)


def test_rejects_non_positive_test_size() -> None:
    """Validation split fraction must be positive."""
    df = _make_df()
    with pytest.raises(ValueError, match="test_size"):
        split_by_engine(df, test_size=0.0, random_seed=42)
