"""Feature-family screen: cell enumeration, CSV resume layer, and harness.

This module provides:

- The :class:`ScreenCell` frozen dataclass defining one training run.
- :func:`enumerate_cells` to build the full sweep grid.
- :func:`cell_key`, :func:`csv_path`, :func:`completed_keys`, and
  :func:`append_row` for the CSV resume layer.
- :func:`run_cell` to train one cell and return a result row dict.
- :func:`run_screen` to orchestrate the full sweep with resume.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import pandas as pd

from turbofan.config.schema import (
    DataConfig,
    FeatureFamilyName,
    SequenceConfig,
    load_config,
)
from turbofan.data.loader import load_raw_train
from turbofan.features.pipeline import build_feature_pipeline
from turbofan.models.evaluate import add_rul_column
from turbofan.models.sequence_models import build_sequence_model
from turbofan.models.sequence_training import (
    resolve_device,
    seed_everything,
    train_sequence_model,
)
from turbofan.models.split import split_by_engine
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.windowing import build_sliding_windows
from turbofan.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Spec-fixed hyperparameters (held constant across the entire sweep)
# ---------------------------------------------------------------------------

HIDDEN_SIZE: int = 64
"""Recurrent hidden state width, fixed for the feature-family sweep."""

NUM_LAYERS: int = 1
"""Number of stacked recurrent layers, fixed for the sweep."""

DROPOUT: float = 0.0
"""Dropout probability, fixed for the sweep."""

LEARNING_RATE: float = 1e-3
"""Adam optimizer learning rate, fixed for the sweep."""

BATCH_SIZE: int = 64
"""Training and evaluation batch size, fixed for the sweep."""

EPOCHS: int = 50
"""Maximum training epochs, fixed for the sweep."""

PATIENCE: int = 8
"""Early-stopping patience in epochs, fixed for the sweep."""

MAX_RUL: int = 125
"""Maximum RUL cap for piecewise-linear labels, fixed for the sweep."""

TEST_SIZE: float = 0.2
"""Fraction of engines held out for validation, fixed for the sweep."""

SPLIT_SEED: int = 42
"""Random seed for the engine train/val split, fixed for the sweep.

This seed is NEVER cell.seed. The split (and KMeans normaliser) use this
constant so that all cells see identical train/val engine assignments and
feature distributions. Only model init and training consume cell.seed.
"""

__all__ = [
    "BATCH_SIZE",
    "CSV_COLUMNS",
    "DROPOUT",
    "EPOCHS",
    "HIDDEN_SIZE",
    "LEARNING_RATE",
    "MAX_RUL",
    "NUM_LAYERS",
    "PATIENCE",
    "SPLIT_SEED",
    "TEST_SIZE",
    "ScreenCell",
    "append_row",
    "cell_key",
    "completed_keys",
    "csv_path",
    "enumerate_cells",
    "run_cell",
    "run_screen",
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


# ---------------------------------------------------------------------------
# Identity columns joined during the pipeline data-flow
# ---------------------------------------------------------------------------
_ID_COLS: list[str] = ["engine_id", "cycle", "rul"]


def run_cell(
    cell: ScreenCell,
    *,
    configs_dir: Path = Path("configs/subsets"),
    device: str = "cpu",
) -> dict[str, Any]:
    """Train one cell and return a result row dict matching :data:`CSV_COLUMNS`.

    All sweep-fixed hyperparameters (:data:`HIDDEN_SIZE`, :data:`EPOCHS`, etc.)
    are read from the module constants.  The engine split and KMeans normaliser
    always use :data:`SPLIT_SEED` (42); only model init and training consume
    ``cell.seed``.  No MLflow logging, no model registry, no checkpoint writes.

    Args:
        cell: The fully-specified sweep cell to execute.
        configs_dir: Directory containing per-subset YAML configs (each named
            ``{subset_lower}.yaml``, e.g. ``fd001.yaml``).
        device: Requested compute device (``"cpu"``, ``"cuda"``, or
            ``"auto"``). ``"auto"`` resolves to CUDA when available and falls
            back to CPU otherwise, so the same cell runs on GPU and CPU nodes.

    Returns:
        A dict with exactly the keys in :data:`CSV_COLUMNS`.  The
        ``rolling_window`` and ``lag_step`` fields are ``""`` when the cell's
        value is ``None``.
    """
    # 1. Inherit sensor_cols_to_drop and n_modes from the subset config.
    subset_cfg = load_config(configs_dir / f"{cell.subset.lower()}.yaml")
    sensor_cols_to_drop = subset_cfg.features.sensor_cols_to_drop
    n_modes = subset_cfg.features.n_modes

    # 2. Build DataConfig for loading — use subset config's data paths.
    data_cfg = DataConfig(
        raw_dir=subset_cfg.data.raw_dir,
        processed_dir=subset_cfg.data.processed_dir,
        interim_dir=subset_cfg.data.interim_dir,
        fd_subset=cell.subset,  # type: ignore[arg-type]
        max_rul=MAX_RUL,
        test_size=TEST_SIZE,
        random_seed=SPLIT_SEED,
    )

    # 3. Resolve the device first so "auto" collapses to a concrete name before
    #    SequenceConfig (whose device field is a Pydantic-validated literal).
    dev = resolve_device(device)  # type: ignore[arg-type]
    device_name: Literal["cpu", "cuda"] = "cuda" if dev.type == "cuda" else "cpu"

    # 4. Build SequenceConfig for training (all spec-fixed values).
    seq_cfg = SequenceConfig(
        architecture=cell.architecture,  # type: ignore[arg-type]
        window_size=cell.sequence_window,
        batch_size=BATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        patience=PATIENCE,
        device=device_name,
    )

    # 5. Data pipeline — split seed and pipeline random_state are ALWAYS 42.
    train_raw = load_raw_train(data_cfg)
    train_labeled = add_rul_column(train_raw, max_rul=MAX_RUL)
    train_df, val_df = split_by_engine(
        train_labeled, test_size=TEST_SIZE, random_seed=SPLIT_SEED
    )

    # Build per-cell pipeline arguments.
    windows: list[int] | None = (
        [cell.rolling_window] if cell.rolling_window is not None else None
    )
    lag_steps: list[int] | None = (
        [cell.lag_step] if cell.lag_step is not None else None
    )

    pipeline = build_feature_pipeline(
        sensor_drop=sensor_cols_to_drop or None,
        n_modes=n_modes,
        random_state=SPLIT_SEED,
        feature_families=cell.feature_families,
        windows=windows,
        lag_steps=lag_steps,
    )

    train_features = pipeline.fit_transform(train_df)
    val_features = pipeline.transform(val_df)
    feature_cols: list[str] = (
        pipeline.named_steps["feature_engineer"].feature_cols_
    )

    train_normalized = pd.concat(
        [
            train_df[_ID_COLS].reset_index(drop=True),
            train_features.reset_index(drop=True),
        ],
        axis=1,
    )
    val_normalized = pd.concat(
        [
            val_df[_ID_COLS].reset_index(drop=True),
            val_features.reset_index(drop=True),
        ],
        axis=1,
    )

    train_windows = build_sliding_windows(
        train_normalized,
        feature_cols=feature_cols,
        window_size=cell.sequence_window,
    )
    validation_windows = build_sliding_windows(
        val_normalized,
        feature_cols=feature_cols,
        window_size=cell.sequence_window,
    )

    train_loader = build_sequence_loader(
        train_windows, batch_size=BATCH_SIZE, shuffle=True
    )
    validation_windows_loader = build_sequence_loader(
        validation_windows, batch_size=BATCH_SIZE, shuffle=False
    )

    # 6. Model — only cell.seed governs randomness from here on.
    seed_everything(cell.seed)
    model = build_sequence_model(
        cell.architecture,
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    started = perf_counter()
    result = train_sequence_model(
        model=model,
        train_loader=train_loader,
        validation_windows_loader=validation_windows_loader,
        config=seq_cfg,
        device=dev,
        random_seed=cell.seed,
        max_rul=MAX_RUL,
    )
    duration = perf_counter() - started

    # 7. Read metrics from training result (no extra forward pass).
    val_rmse = float(result.best_metric)
    best_row = result.history.loc[result.history["epoch"] == result.best_epoch]
    val_mae = float(best_row["validation_windows_mae"].iloc[0])

    # 8. Build the result row.
    rw_val: int | str = "" if cell.rolling_window is None else cell.rolling_window
    ls_val: int | str = "" if cell.lag_step is None else cell.lag_step

    return {
        "architecture": cell.architecture,
        "subset": cell.subset,
        "feature_config": cell.feature_config,
        "rolling_window": rw_val,
        "lag_step": ls_val,
        "sequence_window": cell.sequence_window,
        "hidden_size": HIDDEN_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": cell.seed,
        "n_features": len(feature_cols),
        "n_train_windows": int(train_windows.X.shape[0]),
        "n_val_windows": int(validation_windows.X.shape[0]),
        "best_epoch": result.best_epoch,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "training_duration_seconds": duration,
    }


def run_screen(
    architectures: list[str],
    subsets: list[str],
    sequence_windows: list[int],
    rolling_windows: list[int],
    lag_steps: list[int],
    seeds: list[int],
    *,
    results_dir: Path = Path("results"),
    configs_dir: Path = Path("configs/subsets"),
    device: str = "cpu",
) -> None:
    """Orchestrate the full feature-family sweep with CSV resume.

    Enumerates all cells with :func:`enumerate_cells`, checks each
    (architecture, subset) CSV for already-completed keys, skips completed
    cells, and calls :func:`run_cell` for the rest.  Each completed row is
    appended and flushed immediately so a crash loses at most the cell
    currently training.

    Args:
        architectures: Architecture names to sweep (``"gru"`` and/or
            ``"lstm"``).
        subsets: C-MAPSS subset identifiers to sweep.
        sequence_windows: Sequence window sizes to sweep.
        rolling_windows: Rolling window sizes for rolling-* configs.
        lag_steps: Lag steps for the lag config.
        seeds: Random seeds for model init / training.
        results_dir: Root directory for result CSV files.
        configs_dir: Directory containing per-subset YAML configs.
        device: Requested compute device (``"cpu"``, ``"cuda"``, or
            ``"auto"``) forwarded to :func:`run_cell` for every cell.
    """
    cells = enumerate_cells(
        architectures=architectures,
        subsets=subsets,
        sequence_windows=sequence_windows,
        rolling_windows=rolling_windows,
        lag_steps=lag_steps,
        seeds=seeds,
    )

    # Cache of completed keys per (arch, subset).
    done: dict[tuple[str, str], set[tuple[str, str, str, str, str]]] = {}
    for arch in architectures:
        for subset in subsets:
            p = csv_path(results_dir, arch, subset)
            done[(arch, subset)] = completed_keys(p)

    total = len(cells)
    for idx, cell in enumerate(cells, start=1):
        key = cell_key(cell)
        cache = done[(cell.architecture, cell.subset)]
        # A cell's full identity is (config, rolling_window, lag_step,
        # sequence_window, seed); log all of them so adjacent cells that share a
        # feature_config (e.g. different rolling windows) are distinguishable.
        cell_desc = (
            f"{cell.architecture} {cell.subset} {cell.feature_config} "
            f"rw={cell.rolling_window} lag={cell.lag_step} "
            f"sw={cell.sequence_window} seed={cell.seed}"
        )
        if key in cache:
            logger.info("skipping completed cell %d/%d: %s", idx, total, cell_desc)
            continue

        logger.info(
            "running cell %d/%d on device=%s: %s", idx, total, device, cell_desc
        )
        row = run_cell(cell, configs_dir=configs_dir, device=device)
        p = csv_path(results_dir, cell.architecture, cell.subset)
        append_row(p, row)
        cache.add(key)
        logger.info(
            "completed cell %d/%d: %s -> val_rmse=%.4f val_mae=%.4f (%.1fs)",
            idx,
            total,
            cell_desc,
            row["val_rmse"],
            row["val_mae"],
            row["training_duration_seconds"],
        )
