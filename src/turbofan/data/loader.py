"""Raw data loaders for the NASA C-MAPSS turbofan dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from turbofan.config.schema import DataConfig
from turbofan.data.contracts import CANONICAL_COLUMNS

_DOWNLOAD_HINT: str = (
    "Run `turbofan-download-data --kaggle` to download the dataset, "
    "or `turbofan-download-data --check` to verify files are present."
)


def _load_txt(path: Path) -> pd.DataFrame:
    """Read a space-delimited C-MAPSS file and assign column names.

    Args:
        path: Path to the .txt file.

    Returns:
        DataFrame with CANONICAL_COLUMNS columns.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}\n{_DOWNLOAD_HINT}"
        )
    df: pd.DataFrame = pd.read_csv(
        path, sep=r"\s+", header=None, index_col=False
    )
    df = df.iloc[:, : len(CANONICAL_COLUMNS)]
    df.columns = pd.Index(CANONICAL_COLUMNS)
    return df


def load_raw_train(cfg: DataConfig) -> pd.DataFrame:
    """Load raw training data for the configured FD subset.

    Args:
        cfg: DataConfig with ``raw_dir`` and ``fd_subset``.

    Returns:
        DataFrame with columns: engine_id, cycle, op_1–op_3, s_1–s_21.

    Raises:
        FileNotFoundError: If the training file is not found.
    """
    return _load_txt(cfg.raw_dir / f"train_{cfg.fd_subset}.txt")


def load_raw_test(cfg: DataConfig) -> pd.DataFrame:
    """Load raw test data for the configured FD subset.

    Args:
        cfg: DataConfig with ``raw_dir`` and ``fd_subset``.

    Returns:
        DataFrame with columns: engine_id, cycle, op_1–op_3, s_1–s_21.

    Raises:
        FileNotFoundError: If the test file is not found.
    """
    return _load_txt(cfg.raw_dir / f"test_{cfg.fd_subset}.txt")


def load_rul_labels(cfg: DataConfig) -> pd.Series[int]:
    """Load ground-truth RUL values for test engines.

    Args:
        cfg: DataConfig with ``raw_dir`` and ``fd_subset``.

    Returns:
        pd.Series of integer RUL values, one per test engine, named ``"rul"``.

    Raises:
        FileNotFoundError: If the RUL file is not found.
    """
    path = cfg.raw_dir / f"RUL_{cfg.fd_subset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected RUL file not found: {path}\n{_DOWNLOAD_HINT}"
        )
    return pd.read_csv(path, header=None).iloc[:, 0].rename("rul")
