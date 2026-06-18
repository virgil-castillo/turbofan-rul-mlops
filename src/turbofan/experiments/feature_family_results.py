"""CSV resume storage for the sequence feature-family screen."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

CSV_COLUMNS: list[str] = [
    "architecture",
    "subset",
    "feature_config",
    "rolling_window",
    "lag_step",
    "sequence_window",
    "hidden_size",
    "learning_rate",
    "seed",
    "n_features",
    "n_train_windows",
    "n_val_windows",
    "best_epoch",
    "val_rmse",
    "val_mae",
    "training_duration_seconds",
]


def csv_path(results_dir: Path, architecture: str, subset: str) -> Path:
    """Return the output CSV path for a given architecture and subset.

    Args:
        results_dir: Root directory for result CSV files.
        architecture: Architecture name, e.g. ``"gru"`` or ``"lstm"``.
        subset: C-MAPSS subset identifier, e.g. ``"FD001"``.

    Returns:
        Path to the CSV file for this architecture/subset combination.
    """
    return results_dir / f"feature_family_screen_{architecture}_{subset}.csv"


def completed_keys(path: Path) -> set[tuple[str, str, str, str, str]]:
    """Read an existing result CSV and return completed cell keys.

    Rows with missing identity fields, such as a half-written trailing row, are
    silently skipped.

    Args:
        path: Path to the CSV file. A missing file yields an empty set.

    Returns:
        Set of completed cell identity keys.
    """
    if not path.exists():
        return set()

    keys: set[tuple[str, str, str, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fc = row["feature_config"]
            rw = row["rolling_window"]
            ls = row["lag_step"]
            sw = row["sequence_window"]
            sd = row["seed"]
            if None in (fc, rw, ls, sw, sd):
                continue
            keys.add((str(fc), str(rw), str(ls), str(sw), str(sd)))
    return keys


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append one result row to the output CSV, flushing immediately.

    If the file does not exist or is empty, the header is written first. The
    parent directory is created if needed.

    Args:
        path: Destination CSV file path.
        row: Mapping from column name to value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
