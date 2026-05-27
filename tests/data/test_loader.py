"""Tests for turbofan.data.loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from turbofan.config.schema import DataConfig
from turbofan.data.loader import (
    COLUMN_NAMES,
    load_raw_test,
    load_raw_train,
    load_rul_labels,
)


def test_load_raw_train_column_names(data_cfg: DataConfig) -> None:
    """load_raw_train assigns the correct 26 column names."""
    df = load_raw_train(data_cfg)
    assert list(df.columns) == COLUMN_NAMES


def test_load_raw_train_returns_nonempty_dataframe(data_cfg: DataConfig) -> None:
    """load_raw_train returns a non-empty DataFrame."""
    df = load_raw_train(data_cfg)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_raw_train_engine_id_column(data_cfg: DataConfig) -> None:
    """engine_id column contains the expected engine IDs."""
    df = load_raw_train(data_cfg)
    assert set(df["engine_id"].unique()) == {1, 2, 3}


def test_load_raw_test_column_names(data_cfg: DataConfig) -> None:
    """load_raw_test assigns the correct 26 column names."""
    df = load_raw_test(data_cfg)
    assert list(df.columns) == COLUMN_NAMES


def test_load_raw_train_missing_file_raises(tmp_path: Path) -> None:
    """load_raw_train raises FileNotFoundError when file is missing."""
    cfg = DataConfig(
        raw_dir=tmp_path / "nonexistent",
        processed_dir=tmp_path,
        interim_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="turbofan-download-data"):
        load_raw_train(cfg)


def test_load_raw_test_missing_file_raises(tmp_path: Path) -> None:
    """load_raw_test raises FileNotFoundError when file is missing."""
    cfg = DataConfig(
        raw_dir=tmp_path / "nonexistent",
        processed_dir=tmp_path,
        interim_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="turbofan-download-data"):
        load_raw_test(cfg)


def test_load_rul_labels_returns_series(data_cfg: DataConfig) -> None:
    """load_rul_labels returns a pd.Series."""
    rul = load_rul_labels(data_cfg)
    assert isinstance(rul, pd.Series)


def test_load_rul_labels_length(data_cfg: DataConfig) -> None:
    """load_rul_labels returns one value per test engine (3 in stub file)."""
    rul = load_rul_labels(data_cfg)
    assert len(rul) == 3


def test_load_rul_labels_missing_file_raises(tmp_path: Path) -> None:
    """load_rul_labels raises FileNotFoundError when file is missing."""
    cfg = DataConfig(
        raw_dir=tmp_path / "nonexistent",
        processed_dir=tmp_path,
        interim_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="turbofan-download-data"):
        load_rul_labels(cfg)
