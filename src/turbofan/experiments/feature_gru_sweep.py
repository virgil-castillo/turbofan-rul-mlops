"""Sweep GRU feature engineering configurations on the validation split."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import numpy as np
import pandas as pd

from turbofan.config.schema import load_config
from turbofan.data.loader import load_raw_train
from turbofan.models.evaluate import add_rul_column
from turbofan.models.gru import GRURULRegressor
from turbofan.models.metrics import regression_metrics
from turbofan.models.sequence_training import (
    predict_windows,
    resolve_device,
    seed_everything,
    train_gru_model,
)
from turbofan.models.split import split_by_engine
from turbofan.models.training_log import append_training_log, build_log_entry
from turbofan.preprocessing.normalization import OperatingModeNormalizer
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.feature_selection import select_correlated_sensors
from turbofan.sequences.windowing import build_sliding_windows

VALID_FEATURE_SETS = frozenset(
    {"raw", "raw_plus_rolling", "top_corr", "top_corr_rolling"}
)
RESULT_COLUMNS = [
    "feature_set",
    "corr_threshold",
    "n_features",
    "best_epoch",
    "rmse",
    "mae",
]

_ALL_SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
_OP_COLS = ["op_1", "op_2", "op_3"]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML project config.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=sorted(VALID_FEATURE_SETS),
        help="Feature set families to evaluate.",
    )
    parser.add_argument(
        "--corr-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="Correlation thresholds for top_corr/top_corr_rolling feature sets.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Rolling window size for rolling feature sets.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Torch device for training.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for sweep results.",
    )
    return parser.parse_args()


def _validate_inputs(
    feature_sets: list[str],
    corr_thresholds: list[float],
    rolling_window: int,
) -> None:
    """Validate feature sweep input values.

    Args:
        feature_sets: Feature set family names to evaluate.
        corr_thresholds: Correlation thresholds for top_corr/top_corr_rolling sets.
        rolling_window: Rolling window size in cycles.

    Raises:
        ValueError: If any input values are invalid.
    """
    invalid = set(feature_sets) - VALID_FEATURE_SETS
    if invalid:
        raise ValueError(f"Invalid feature sets: {sorted(invalid)}")
    if not corr_thresholds or any(t <= 0.0 or t >= 1.0 for t in corr_thresholds):
        raise ValueError("Correlation thresholds must be in (0, 1).")
    if rolling_window <= 0:
        raise ValueError("Rolling window must be positive.")


def _build_sweep_grid(
    feature_sets: list[str],
    corr_thresholds: list[float],
) -> list[tuple[str, float | None]]:
    """Build the list of (feature_set, corr_threshold) run specifications.

    For ``raw`` and ``raw_plus_rolling`` feature sets, one entry with
    ``corr_threshold=None`` is added. For ``top_corr`` and
    ``top_corr_rolling``, one entry per threshold value is added.

    Args:
        feature_sets: Feature set family names to include.
        corr_thresholds: Correlation thresholds for filtered sets.

    Returns:
        List of (feature_set, corr_threshold) tuples defining the sweep.
    """
    grid: list[tuple[str, float | None]] = []
    for fs in feature_sets:
        if fs in {"raw", "raw_plus_rolling"}:
            grid.append((fs, None))
        else:
            for threshold in corr_thresholds:
                grid.append((fs, threshold))
    return grid


def _rolling_feature_cols(sensor_cols: list[str], rolling_window: int) -> list[str]:
    """Return expected rolling feature column names for the given sensors.

    Args:
        sensor_cols: Sensor column names to generate rolling features for.
        rolling_window: Rolling window size used during extraction.

    Returns:
        Rolling feature column names in order: rmean, rstd, rmin, rmax
        per sensor.
    """
    cols: list[str] = []
    for sensor in sensor_cols:
        for stat in ["rmean", "rstd", "rmin", "rmax"]:
            cols.append(f"{sensor}_{stat}_{rolling_window}")
    return cols


def _add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int,
) -> pd.DataFrame:
    """Append rolling-statistic columns to a sensor DataFrame.

    Computes per-engine rolling mean, std, min, and max for each sensor
    using ``min_periods=1`` so early cycles receive values instead of NaN.

    Args:
        df: DataFrame with ``engine_id`` and sensor columns.
        sensor_cols: Sensor column names to compute rolling statistics for.
        window: Rolling window size in cycles.

    Returns:
        Copy of ``df`` with additional ``{sensor}_{stat}_{window}`` columns.
    """
    extra: dict[str, pd.Series] = {}
    for col in sensor_cols:
        grp = df.groupby("engine_id")[col]
        extra[f"{col}_rmean_{window}"] = grp.transform(
            lambda s, _w=window: s.rolling(_w, min_periods=1).mean()
        )
        extra[f"{col}_rstd_{window}"] = grp.transform(
            lambda s, _w=window: s.rolling(_w, min_periods=1).std()
        ).fillna(0.0)
        extra[f"{col}_rmin_{window}"] = grp.transform(
            lambda s, _w=window: s.rolling(_w, min_periods=1).min()
        )
        extra[f"{col}_rmax_{window}"] = grp.transform(
            lambda s, _w=window: s.rolling(_w, min_periods=1).max()
        )
    return pd.concat([df.copy(), pd.DataFrame(extra, index=df.index)], axis=1)


def _append_incremental_row(
    row: dict[str, object],
    output_path: Path,
    *,
    append: bool,
) -> None:
    """Append one completed result row to an incremental CSV.

    Args:
        row: Completed sweep result row.
        output_path: Destination CSV path.
        append: Whether to append to an existing incremental CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(
        output_path,
        mode="a" if append else "w",
        header=not append,
        index=False,
    )


def _device_name(device: object) -> str:
    """Return a stable display name for a resolved training device.

    Args:
        device: Resolved device object.

    Returns:
        Device type string when available, otherwise ``str(device)``.
    """
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type
    return str(device)


def run_feature_sweep(
    config_path: Path,
    feature_sets: list[str],
    corr_thresholds: list[float],
    rolling_window: int,
    device: str,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Train and evaluate a GRU validation feature engineering sweep.

    Evaluates one or more feature set families across correlation threshold
    values and returns sorted results. For ``raw`` and ``raw_plus_rolling``,
    one run each. For ``top_corr`` and ``top_corr_rolling``, one run per
    threshold value.

    Args:
        config_path: Project config path.
        feature_sets: Feature set family names to evaluate.
        corr_thresholds: Correlation thresholds for top_corr/top_corr_rolling
            feature sets.
        rolling_window: Rolling window size in cycles for rolling feature sets.
        device: Requested torch device, either ``"cpu"`` or ``"cuda"``.
        output_path: Optional CSV path for incremental and final results.

    Returns:
        Results sorted by validation RMSE.

    Raises:
        ValueError: If sweep inputs are invalid.
    """
    _validate_inputs(feature_sets, corr_thresholds, rolling_window)
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'.")

    cfg = load_config(config_path)
    if cfg.sequence.architecture != "gru":
        raise ValueError("Feature sweep requires sequence architecture='gru'.")

    torch_device = resolve_device(cast(Literal["cpu", "cuda"], device))
    train_raw = load_raw_train(cfg.data)
    train_labeled = add_rul_column(train_raw, max_rul=cfg.data.max_rul)
    train_df, val_df = split_by_engine(
        train_labeled,
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
    )

    grid = _build_sweep_grid(feature_sets, corr_thresholds)
    total_runs = len(grid)
    rows: list[dict[str, object]] = []

    for run_idx, (feature_set, corr_threshold) in enumerate(grid, 1):
        use_rolling = feature_set in {"raw_plus_rolling", "top_corr_rolling"}
        use_corr = feature_set in {"top_corr", "top_corr_rolling"}

        if use_corr:
            sensor_cols = select_correlated_sensors(
                train_df,
                threshold=cast(float, corr_threshold),
            )
        else:
            sensor_cols = list(_ALL_SENSOR_COLS)

        base_feature_cols = list(_OP_COLS) + list(sensor_cols)
        feature_cols = list(base_feature_cols)

        current_train_df = train_df
        current_val_df = val_df

        if use_rolling:
            current_train_df = _add_rolling_features(
                train_df, sensor_cols, rolling_window
            )
            current_val_df = _add_rolling_features(
                val_df, sensor_cols, rolling_window
            )
            feature_cols = list(base_feature_cols) + _rolling_feature_cols(
                sensor_cols, rolling_window
            )

        normalizer = OperatingModeNormalizer(
            feature_cols=feature_cols,
            n_modes=cfg.features.n_modes,
            random_state=cfg.data.random_seed,
        )
        train_normalized = normalizer.fit_transform(current_train_df)
        val_normalized = normalizer.transform(current_val_df)

        window_size = cfg.sequence.window_size
        train_windows = build_sliding_windows(
            train_normalized,
            feature_cols=feature_cols,
            window_size=window_size,
        )
        validation_windows = build_sliding_windows(
            val_normalized,
            feature_cols=feature_cols,
            window_size=window_size,
        )

        train_loader = build_sequence_loader(
            train_windows,
            batch_size=cfg.sequence.batch_size,
            shuffle=True,
        )
        validation_windows_loader = build_sequence_loader(
            validation_windows,
            batch_size=cfg.sequence.batch_size,
            shuffle=False,
        )

        seed_everything(cfg.data.random_seed)
        model = GRURULRegressor(
            input_size=len(feature_cols),
            hidden_size=cfg.sequence.hidden_size,
            num_layers=cfg.sequence.num_layers,
            dropout=cfg.sequence.dropout,
        )
        training_start = perf_counter()
        result = train_gru_model(
            model=model,
            train_loader=train_loader,
            validation_windows_loader=validation_windows_loader,
            config=cfg.sequence,
            device=torch_device,
            random_seed=cfg.data.random_seed,
            max_rul=cfg.data.max_rul,
        )
        training_duration_seconds = perf_counter() - training_start

        predictions = np.clip(
            predict_windows(
                result.model,
                validation_windows_loader,
                torch_device,
                max_rul=cfg.data.max_rul,
            ),
            0.0,
            None,
        )
        metrics = regression_metrics(
            validation_windows.y.astype(np.float64),
            predictions,
        )

        threshold_val: float | None = corr_threshold
        row: dict[str, object] = {
            "feature_set": feature_set,
            "corr_threshold": threshold_val,
            "n_features": len(feature_cols),
            "best_epoch": result.best_epoch,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
        }

        rows.append(row)
        if output_path is not None:
            _append_incremental_row(row, output_path, append=len(rows) > 1)

        extra_dict: dict[str, object] = {
            "feature_set": feature_set,
            "corr_threshold": threshold_val,
            "n_features": len(feature_cols),
            "rolling_window": rolling_window,
        }

        log_entry = build_log_entry(
            model_type="gru",
            dataset=cfg.data.fd_subset,
            random_seed=cfg.data.random_seed,
            hyperparameters={
                "window_size": window_size,
                "hidden_size": cfg.sequence.hidden_size,
                "learning_rate": cfg.sequence.learning_rate,
                "num_layers": cfg.sequence.num_layers,
                "dropout": cfg.sequence.dropout,
                "batch_size": cfg.sequence.batch_size,
                "epochs": cfg.sequence.epochs,
                "patience": cfg.sequence.patience,
            },
            metrics=metrics,
            training_duration_seconds=training_duration_seconds,
            device=_device_name(torch_device),
            run_dir=None,
            best_epoch=result.best_epoch,
            extra=extra_dict,
        )
        append_training_log(log_entry)
        print(
            f"run {run_idx}/{total_runs}: "
            f"feature_set={feature_set} "
            f"corr_threshold={corr_threshold} "
            f"n_features={len(feature_cols)} "
            f"rmse={metrics['rmse']:.6f}"
        )

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return results.sort_values("rmse").reset_index(drop=True)


def main() -> None:
    """Run the GRU feature engineering sweep CLI."""
    args = _parse_args()
    results = run_feature_sweep(
        config_path=args.config,
        feature_sets=args.feature_sets,
        corr_thresholds=args.corr_thresholds,
        rolling_window=args.rolling_window,
        device=args.device,
        output_path=args.output,
    )
    print(results.to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
