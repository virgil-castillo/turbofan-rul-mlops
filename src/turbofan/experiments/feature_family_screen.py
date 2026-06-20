"""Feature-family screen execution harness.

The grid definition and CSV resume layer live in smaller sibling modules. This
module keeps the training boundary and top-level orchestration together, while
re-exporting the original public helpers for compatibility.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from turbofan import workflows
from turbofan.config import schema
from turbofan.config.schema import (
    DataConfig,
    DeviceRequest,
    FDSubset,
    SequenceArchitecture,
    SequenceConfig,
)
from turbofan.experiments.feature_family_grid import (
    ScreenCell,
    cell_key,
    enumerate_cells,
)
from turbofan.experiments.feature_family_results import (
    CSV_COLUMNS,
    append_row,
    completed_keys,
    csv_path,
)
from turbofan.models import sequence_training
from turbofan.utils import logging as turbofan_logging

logger = turbofan_logging.get_logger(__name__)

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
"""Random seed for the engine train/val split, fixed for the sweep."""

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


def run_cell(
    cell: ScreenCell,
    *,
    configs_dir: Path = Path("configs/subsets"),
    device: DeviceRequest = "cpu",
) -> dict[str, Any]:
    """Train one cell and return a result row matching :data:`CSV_COLUMNS`.

    All sweep-fixed hyperparameters are read from module constants. The engine
    split and KMeans normalizer always use :data:`SPLIT_SEED`; only model init
    and training consume ``cell.seed``. No MLflow logging, model registry, or
    checkpoint writes are performed.

    Args:
        cell: The fully-specified sweep cell to execute.
        configs_dir: Directory containing per-subset YAML configs.
        device: Requested compute device (``"cpu"``, ``"cuda"``, or
            ``"auto"``).

    Returns:
        A dict with exactly the keys in :data:`CSV_COLUMNS`.
    """
    subset_cfg = schema.load_config(configs_dir / f"{cell.subset.lower()}.yaml")
    sensor_cols_to_drop = subset_cfg.features.sensor_cols_to_drop
    n_modes = subset_cfg.features.n_modes

    data_cfg = DataConfig(
        raw_dir=subset_cfg.data.raw_dir,
        processed_dir=subset_cfg.data.processed_dir,
        interim_dir=subset_cfg.data.interim_dir,
        fd_subset=cell.subset,
        max_rul=MAX_RUL,
        test_size=TEST_SIZE,
        random_seed=SPLIT_SEED,
    )

    dev = sequence_training.resolve_device(device)
    device_name: Literal["cpu", "cuda"] = "cuda" if dev.type == "cuda" else "cpu"

    seq_cfg = SequenceConfig(
        architecture=cell.architecture,
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

    windows: list[int] | None = (
        [cell.rolling_window] if cell.rolling_window is not None else None
    )
    lag_steps: list[int] | None = (
        [cell.lag_step] if cell.lag_step is not None else None
    )

    prepared = workflows.prepare_sequence_data(
        data_cfg,
        feature_families=cell.feature_families,
        windows=windows,
        lag_steps=lag_steps,
        sensor_drop=sensor_cols_to_drop or None,
        n_modes=n_modes,
        data_seed=SPLIT_SEED,
        max_rul=MAX_RUL,
        test_size=TEST_SIZE,
        window_size=cell.sequence_window,
        batch_size=BATCH_SIZE,
    )

    started = perf_counter()
    result = workflows.train_prepared_sequence(
        prepared,
        seq_cfg,
        device=dev,
        model_seed=cell.seed,
        max_rul=MAX_RUL,
    )
    duration = perf_counter() - started

    val_rmse = float(result.best_metric)
    best_row = result.history.loc[result.history["epoch"] == result.best_epoch]
    val_mae = float(best_row["validation_windows_mae"].iloc[0])

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
        "n_features": len(prepared.feature_cols),
        "n_train_windows": int(prepared.train_windows.X.shape[0]),
        "n_val_windows": int(prepared.val_windows.X.shape[0]),
        "best_epoch": result.best_epoch,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "training_duration_seconds": duration,
    }


def run_screen(
    architectures: Sequence[SequenceArchitecture],
    subsets: Sequence[FDSubset],
    sequence_windows: Sequence[int],
    rolling_windows: Sequence[int],
    lag_steps: Sequence[int],
    seeds: Sequence[int],
    *,
    results_dir: Path = Path("outputs/results"),
    configs_dir: Path = Path("configs/subsets"),
    device: DeviceRequest = "cpu",
) -> None:
    """Orchestrate the full feature-family sweep with CSV resume.

    Args:
        architectures: Architecture names to sweep.
        subsets: C-MAPSS subset identifiers to sweep.
        sequence_windows: Sequence window sizes to sweep.
        rolling_windows: Rolling window sizes for rolling feature configs.
        lag_steps: Lag steps for the lag config.
        seeds: Random seeds for model init and training.
        results_dir: Root directory for generated result CSV files.
        configs_dir: Directory containing per-subset YAML configs.
        device: Requested compute device forwarded to :func:`run_cell`.
    """
    cells = enumerate_cells(
        architectures=architectures,
        subsets=subsets,
        sequence_windows=sequence_windows,
        rolling_windows=rolling_windows,
        lag_steps=lag_steps,
        seeds=seeds,
    )

    done: dict[tuple[str, str], set[tuple[str, str, str, str, str]]] = {}
    for arch in architectures:
        for subset in subsets:
            path = csv_path(results_dir, arch, subset)
            done[(arch, subset)] = completed_keys(path)

    total = len(cells)
    for idx, cell in enumerate(cells, start=1):
        key = cell_key(cell)
        cache = done[(cell.architecture, cell.subset)]
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
        path = csv_path(results_dir, cell.architecture, cell.subset)
        append_row(path, row)
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
