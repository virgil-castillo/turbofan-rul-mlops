"""Feature-family screen scaffold: cell enumeration and CSV resume layer.

This module provides the pure, IO-light building blocks for the feature-family
sweep experiment: cell definition, grid enumeration, stable resume keys, and
the CSV append/read layer. Model training (``run_cell``) and full orchestration
(``run_screen``) are implemented in a separate task.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from turbofan.config.schema import FeatureFamilyName

__all__ = [
    "CSV_COLUMNS",
    "ScreenCell",
    "append_row",
    "cell_key",
    "completed_keys",
    "csv_path",
    "enumerate_cells",
]

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


@dataclass(frozen=True)
class ScreenCell:
    """One cell in the feature-family sweep grid.

    A cell fully specifies a single training run: which architecture, which
    C-MAPSS subset, which feature configuration, which rolling window or lag
    step (when applicable), which sequence window, and which random seed.

    Args:
        architecture: Sequence model architecture, ``"gru"`` or ``"lstm"``.
        subset: C-MAPSS fault dataset subset, e.g. ``"FD001"``.
        feature_config: Stable string label for the feature configuration,
            e.g. ``"raw"``, ``"raw+rolling_slope"``, ``"raw+lag"``.
        feature_families: Ordered list of feature family names passed to the
            pipeline, e.g. ``["raw"]`` or ``["raw", "rolling_slope"]``.
        rolling_window: Rolling window size when applicable, else ``None``.
        lag_step: Lag step when applicable, else ``None``.
        sequence_window: Number of cycles per sequence window.
        seed: Random seed for reproducibility.
    """

    architecture: str
    subset: str
    feature_config: str
    feature_families: list[FeatureFamilyName]
    rolling_window: int | None
    lag_step: int | None
    sequence_window: int
    seed: int

    def __post_init__(self) -> None:
        """Validate that feature_families is stored as a list (not other seq)."""
        object.__setattr__(self, "feature_families", list(self.feature_families))


# ---------------------------------------------------------------------------
# Feature-config table
# ---------------------------------------------------------------------------

# Each entry: (label, families, swept_factor)
# swept_factor is one of "rolling_window", "lag_step", or "none"
_FEATURE_CONFIGS: list[tuple[str, list[FeatureFamilyName], str]] = [
    ("raw", ["raw"], "none"),
    ("raw+rolling_mean", ["raw", "rolling_mean"], "rolling_window"),
    ("raw+rolling_std", ["raw", "rolling_std"], "rolling_window"),
    ("raw+rolling_min", ["raw", "rolling_min"], "rolling_window"),
    ("raw+rolling_max", ["raw", "rolling_max"], "rolling_window"),
    ("raw+rolling_slope", ["raw", "rolling_slope"], "rolling_window"),
    ("raw+rolling_delta", ["raw", "rolling_delta"], "rolling_window"),
    ("raw+lag", ["raw", "lag"], "lag_step"),
]


def enumerate_cells(
    architectures: list[str],
    subsets: list[str],
    sequence_windows: list[int],
    rolling_windows: list[int],
    lag_steps: list[int],
    seeds: list[int],
) -> list[ScreenCell]:
    """Enumerate every cell of the feature-family sweep grid.

    For each (architecture, subset, sequence_window, seed) combination:
    - Emits 1 raw cell (rolling_window=None, lag_step=None).
    - For each of the 6 rolling-* configs, emits one cell per value in
      ``rolling_windows`` (lag_step=None).
    - For the lag config, emits one cell per value in ``lag_steps``
      (rolling_window=None).

    With the spec defaults (``rolling_windows=[5,20]``, ``lag_steps=[1,5]``,
    ``sequence_windows=[30,60]``, ``architectures=["gru","lstm"]``,
    ``subsets=["FD001","FD002","FD003","FD004"]``, ``seeds=[42]``) this
    produces exactly 240 cells (15 per arch/subset/sequence_window slice).

    Args:
        architectures: List of architecture names (``"gru"`` or ``"lstm"``).
        subsets: List of C-MAPSS subset identifiers.
        sequence_windows: List of sequence window sizes.
        rolling_windows: List of rolling window sizes for rolling-* configs.
        lag_steps: List of lag steps for the lag config.
        seeds: List of random seeds.

    Returns:
        List of :class:`ScreenCell` instances, one per grid cell.
    """
    cells: list[ScreenCell] = []
    for arch in architectures:
        for subset in subsets:
            for sw in sequence_windows:
                for seed in seeds:
                    for label, families, swept in _FEATURE_CONFIGS:
                        if swept == "none":
                            cells.append(
                                ScreenCell(
                                    architecture=arch,
                                    subset=subset,
                                    feature_config=label,
                                    feature_families=families,
                                    rolling_window=None,
                                    lag_step=None,
                                    sequence_window=sw,
                                    seed=seed,
                                )
                            )
                        elif swept == "rolling_window":
                            for rw in rolling_windows:
                                cells.append(
                                    ScreenCell(
                                        architecture=arch,
                                        subset=subset,
                                        feature_config=label,
                                        feature_families=families,
                                        rolling_window=rw,
                                        lag_step=None,
                                        sequence_window=sw,
                                        seed=seed,
                                    )
                                )
                        else:  # lag_step
                            for ls in lag_steps:
                                cells.append(
                                    ScreenCell(
                                        architecture=arch,
                                        subset=subset,
                                        feature_config=label,
                                        feature_families=families,
                                        rolling_window=None,
                                        lag_step=ls,
                                        sequence_window=sw,
                                        seed=seed,
                                    )
                                )
    return cells


def cell_key(cell: ScreenCell) -> tuple[str, str, str, str, str]:
    """Return the 5-string resume identity key for a cell.

    Inapplicable (``None``) factors are rendered as the empty string ``""`` so
    that the key compares equal to values parsed back from CSV text.

    Args:
        cell: The screen cell to compute the key for.

    Returns:
        A 5-tuple of strings:
        ``(feature_config, rolling_window, lag_step, sequence_window, seed)``.
    """
    rw = "" if cell.rolling_window is None else str(cell.rolling_window)
    ls = "" if cell.lag_step is None else str(cell.lag_step)
    return (
        cell.feature_config,
        rw,
        ls,
        str(cell.sequence_window),
        str(cell.seed),
    )


def csv_path(results_dir: Path, architecture: str, subset: str) -> Path:
    """Return the output CSV path for a given architecture and subset.

    Args:
        results_dir: Root directory for result CSV files.
        architecture: Architecture name (e.g. ``"gru"`` or ``"lstm"``).
        subset: C-MAPSS subset identifier (e.g. ``"FD001"``).

    Returns:
        Path to the CSV file for this (architecture, subset) combination.
    """
    return results_dir / f"feature_family_screen_{architecture}_{subset}.csv"


def completed_keys(path: Path) -> set[tuple[str, str, str, str, str]]:
    """Read an existing result CSV and return the set of completed cell keys.

    Each key is a 5-string tuple matching the format produced by
    :func:`cell_key`: ``(feature_config, rolling_window, lag_step,
    sequence_window, seed)``.  Rows with the wrong number of fields (e.g. a
    half-written trailing row) are silently skipped.

    Args:
        path: Path to the CSV file. If the file does not exist, an empty set
            is returned.

    Returns:
        Set of completed cell keys (as all-string 5-tuples).
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
            # DictReader fills short rows with None; skip them
            if None in (fc, rw, ls, sw, sd):
                continue
            key: tuple[str, str, str, str, str] = (
                str(fc),
                str(rw),
                str(ls),
                str(sw),
                str(sd),
            )
            keys.add(key)
    return keys


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append one result row to the output CSV, flushing immediately.

    If the file does not exist or is empty, the header is written first.
    The parent directory is created if needed. Each row is flushed to the OS
    buffer immediately after writing so that a crash cannot lose a completed
    cell.

    Args:
        path: Destination CSV file path.
        row: Mapping from column name to value; must contain all keys in
            :data:`CSV_COLUMNS`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
