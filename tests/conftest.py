"""Shared pytest fixtures for the turbofan test suite."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from turbofan.config.schema import DataConfig

SENSOR_COLS = {f"s_{i}": float(i) for i in range(1, 22)}


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """Minimal valid train DataFrame: 3 engines with varied cycle lengths.

    Engine 1: 10 cycles, Engine 2: 7 cycles, Engine 3: 15 cycles.
    """
    rows = []
    for engine_id, n_cycles in [(1, 10), (2, 7), (3, 15)]:
        for cycle in range(1, n_cycles + 1):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "op_1": 0.0,
                    "op_2": 0.0,
                    "op_3": 0.0,
                    **SENSOR_COLS,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_data_dir(tmp_path: Path, sample_train_df: pd.DataFrame) -> Path:
    """Temp directory with correctly-named stub C-MAPSS .txt files for FD001."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    train_path = raw_dir / "train_FD001.txt"
    sample_train_df.to_csv(train_path, sep=" ", header=False, index=False)

    test_path = raw_dir / "test_FD001.txt"
    sample_train_df.to_csv(test_path, sep=" ", header=False, index=False)

    rul_path = raw_dir / "RUL_FD001.txt"
    rul_path.write_text("112\n98\n42\n")

    return raw_dir


@pytest.fixture
def data_cfg(tmp_data_dir: Path) -> DataConfig:
    """DataConfig pointing at the tmp_data_dir stub files."""
    return DataConfig(
        raw_dir=tmp_data_dir,
        processed_dir=tmp_data_dir / "processed",
        interim_dir=tmp_data_dir / "interim",
        fd_subset="FD001",
    )
