"""Unified feature engineering sweep for Ridge and GRU models."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from turbofan.config.schema import ProjectConfig, load_config
from turbofan.data.loader import load_raw_train
from turbofan.features.engineering import FeatureSet
from turbofan.features.pipeline import build_feature_pipeline
from turbofan.models.baseline import build_baseline_pipeline
from turbofan.models.evaluate import add_rul_column, split_features_target
from turbofan.models.metrics import regression_metrics
from turbofan.models.split import split_by_engine
from turbofan.models.training_log import append_training_log, build_log_entry

SUPPORTED_FEATURE_SETS: frozenset[str] = frozenset(
    {
        "raw",
        "rolling_mean",
        "rolling_stats",
        "raw_plus_rolling_mean",
        "raw_plus_rolling_stats",
        "lag",
        "raw_plus_lag",
    }
)

_ROLLING_FEATURE_SETS: frozenset[str] = frozenset(
    {
        "rolling_mean",
        "rolling_stats",
        "raw_plus_rolling_mean",
        "raw_plus_rolling_stats",
    }
)

_LAG_FEATURE_SETS: frozenset[str] = frozenset({"lag", "raw_plus_lag"})


class ExperimentSpec(NamedTuple):
    """Single feature sweep experiment specification.

    Args:
        feature_set: Feature family to evaluate.
        windows: Rolling window sizes. Non-empty only for rolling feature sets.
        lag_steps: Lag offsets. Non-empty only for the lag feature set.
    """

    feature_set: FeatureSet
    windows: tuple[int, ...]
    lag_steps: tuple[int, ...]


def _format_tuple(values: tuple[int, ...]) -> str:
    """Format an integer tuple as a comma-separated string.

    Args:
        values: Integer tuple to format.

    Returns:
        Comma-separated string, or empty string for an empty tuple.
    """
    return ",".join(str(v) for v in values)


def _build_experiment_specs(
    feature_sets: list[FeatureSet],
    windows: list[int],
    lag_steps: list[int],
) -> list[ExperimentSpec]:
    """Build independent experiment specs from requested feature families.

    Args:
        feature_sets: Validated feature-set names.
        windows: Positive rolling window sizes.
        lag_steps: Positive lag offsets.

    Returns:
        Experiment specs. ``raw`` yields one spec; rolling families yield one
        spec per window; lag families yield one spec per lag step.
    """
    specs: list[ExperimentSpec] = []
    for fs in feature_sets:
        if fs == "raw":
            specs.append(ExperimentSpec(feature_set=fs, windows=(), lag_steps=()))
        elif fs in _LAG_FEATURE_SETS:
            for ls in lag_steps:
                specs.append(
                    ExperimentSpec(feature_set=fs, windows=(), lag_steps=(ls,))
                )
        elif fs in _ROLLING_FEATURE_SETS:
            for w in windows:
                specs.append(
                    ExperimentSpec(feature_set=fs, windows=(w,), lag_steps=())
                )
    return specs


def _validate_inputs(
    model: str,
    feature_sets: list[str],
    windows: list[int],
    lag_steps: list[int],
    n_jobs: int,
) -> list[FeatureSet]:
    """Validate sweep inputs.

    Args:
        model: Model type, either ``"ridge"`` or ``"gru"``.
        feature_sets: Requested feature-set names.
        windows: Requested rolling window sizes.
        lag_steps: Requested lag offsets.
        n_jobs: Joblib parallel worker count (used by Ridge path).

    Returns:
        Validated feature-set names cast to FeatureSet.

    Raises:
        ValueError: If any input value is invalid.
    """
    if model not in {"ridge", "gru"}:
        raise ValueError(f"Model must be 'ridge' or 'gru', got {model!r}.")
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero.")
    if any(w <= 0 for w in windows):
        raise ValueError("All window sizes must be positive.")
    if any(ls <= 0 for ls in lag_steps):
        raise ValueError("All lag steps must be positive.")
    invalid = sorted(set(feature_sets) - SUPPORTED_FEATURE_SETS)
    if invalid:
        raise ValueError(f"Unsupported feature_set: {invalid[0]}")
    return list(feature_sets)  # type: ignore[arg-type]


def _evaluate_ridge_spec(
    spec: ExperimentSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series[float],
    X_val: pd.DataFrame,
    y_val: pd.Series[float],
    alpha: float,
    sensor_drop: list[str] | None,
    n_modes: int,
    random_state: int,
    max_rul: int,
) -> dict[str, object]:
    """Fit and evaluate one Ridge sweep experiment.

    Args:
        spec: Experiment specification.
        X_train: Training feature rows.
        y_train: Training RUL labels.
        X_val: Validation feature rows.
        y_val: Validation RUL labels.
        alpha: Ridge regularization strength.
        sensor_drop: Sensor columns to drop before feature engineering.
        n_modes: Number of operating-mode clusters for normalisation.
        random_state: Random seed.
        max_rul: Maximum RUL cap for prediction clipping.

    Returns:
        Result row with feature metadata and validation metrics.
    """
    estimator = build_baseline_pipeline(
        model_name="ridge",
        alpha=alpha,
        sensor_drop=sensor_drop,
        n_modes=n_modes,
        random_state=random_state,
        feature_set=spec.feature_set,
        windows=list(spec.windows) or None,
        lag_steps=list(spec.lag_steps) or None,
    )
    estimator.fit(X_train, y_train)
    raw_pred = np.asarray(estimator.predict(X_val), dtype=np.float64)
    clipped = np.clip(raw_pred, 0.0, float(max_rul))
    metrics = regression_metrics(y_val, clipped)
    ridge_model = estimator.named_steps["model"]
    return {
        "model": "ridge",
        "feature_set": spec.feature_set,
        "windows": _format_tuple(spec.windows),
        "lag_steps": _format_tuple(spec.lag_steps),
        "n_features": int(len(ridge_model.feature_names_in_)),
        "alpha": alpha,
        **metrics,
    }


def _evaluate_gru_spec(
    spec: ExperimentSpec,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: ProjectConfig,
    device: object,
) -> dict[str, object]:
    """Fit and evaluate one GRU sweep experiment.

    Args:
        spec: Experiment specification.
        train_df: Labeled training DataFrame with RUL column.
        val_df: Labeled validation DataFrame with RUL column.
        cfg: Loaded project configuration.
        device: Resolved torch device for training.

    Returns:
        Result row with feature metadata and validation metrics.
    """
    import torch

    from turbofan.models.gru import GRURULRegressor
    from turbofan.models.sequence_training import (
        predict_windows,
        seed_everything,
        train_gru_model,
    )
    from turbofan.sequences.dataset import build_sequence_loader
    from turbofan.sequences.windowing import build_sliding_windows

    assert isinstance(device, torch.device)

    _id_cols = ["engine_id", "cycle", "rul"]
    pipeline = build_feature_pipeline(
        sensor_drop=list(cfg.features.sensor_cols_to_drop) or None,
        n_modes=cfg.features.n_modes,
        random_state=cfg.data.random_seed,
        feature_set=spec.feature_set,
        windows=list(spec.windows) or None,
        lag_steps=list(spec.lag_steps) or None,
    )
    train_features = pipeline.fit_transform(train_df)
    val_features = pipeline.transform(val_df)
    feature_cols: list[str] = pipeline.named_steps["feature_engineer"].feature_cols_

    train_normalized = pd.concat(
        [
            train_df[_id_cols].reset_index(drop=True),
            train_features.reset_index(drop=True),
        ],
        axis=1,
    )
    val_normalized = pd.concat(
        [
            val_df[_id_cols].reset_index(drop=True),
            val_features.reset_index(drop=True),
        ],
        axis=1,
    )

    train_windows = build_sliding_windows(
        train_normalized,
        feature_cols=feature_cols,
        window_size=cfg.sequence.window_size,
    )
    val_windows = build_sliding_windows(
        val_normalized,
        feature_cols=feature_cols,
        window_size=cfg.sequence.window_size,
    )
    val_metadata = val_windows.metadata
    n_engines_total = int(val_metadata["engine_id"].nunique())
    padded_engines = val_metadata.loc[val_metadata["padded"], "engine_id"].unique()
    n_engines_padded = int(len(padded_engines))
    n_engines_full = n_engines_total - n_engines_padded

    train_loader = build_sequence_loader(
        train_windows, batch_size=cfg.sequence.batch_size, shuffle=True
    )
    val_loader = build_sequence_loader(
        val_windows, batch_size=cfg.sequence.batch_size, shuffle=False
    )

    seed_everything(cfg.data.random_seed)
    model = GRURULRegressor(
        input_size=len(feature_cols),
        hidden_size=cfg.sequence.hidden_size,
        num_layers=cfg.sequence.num_layers,
        dropout=cfg.sequence.dropout,
    )
    t0 = perf_counter()
    result = train_gru_model(
        model=model,
        train_loader=train_loader,
        validation_windows_loader=val_loader,
        config=cfg.sequence,
        device=device,
        random_seed=cfg.data.random_seed,
        max_rul=cfg.data.max_rul,
    )
    training_duration = perf_counter() - t0

    predictions = np.clip(
        predict_windows(result.model, val_loader, device, max_rul=cfg.data.max_rul),
        0.0,
        None,
    )
    metrics = regression_metrics(val_windows.y.astype(np.float64), predictions)

    log_entry = build_log_entry(
        model_type="gru",
        dataset=cfg.data.fd_subset,
        random_seed=cfg.data.random_seed,
        hyperparameters={
            "window_size": cfg.sequence.window_size,
            "hidden_size": cfg.sequence.hidden_size,
            "learning_rate": cfg.sequence.learning_rate,
            "num_layers": cfg.sequence.num_layers,
            "dropout": cfg.sequence.dropout,
            "batch_size": cfg.sequence.batch_size,
            "epochs": cfg.sequence.epochs,
            "patience": cfg.sequence.patience,
        },
        metrics=metrics,
        training_duration_seconds=training_duration,
        device=device.type,
        run_dir=None,
        best_epoch=result.best_epoch,
        extra={
            "feature_set": spec.feature_set,
            "windows": _format_tuple(spec.windows),
            "lag_steps": _format_tuple(spec.lag_steps),
            "n_engines_total": n_engines_total,
            "n_engines_padded": n_engines_padded,
            "n_engines_full": n_engines_full,
        },
    )
    append_training_log(log_entry)

    return {
        "model": "gru",
        "feature_set": spec.feature_set,
        "windows": _format_tuple(spec.windows),
        "lag_steps": _format_tuple(spec.lag_steps),
        "n_features": len(feature_cols),
        "best_epoch": result.best_epoch,
        "n_engines_total": n_engines_total,
        "n_engines_padded": n_engines_padded,
        "n_engines_full": n_engines_full,
        **metrics,
    }


def run_feature_sweep(
    config_path: Path,
    model: Literal["ridge", "gru"],
    feature_sets: list[str],
    windows: list[int],
    lag_steps: list[int],
    n_jobs: int = 1,
    device: str = "cpu",
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Train and evaluate a unified feature engineering sweep.

    Args:
        config_path: Project config path.
        model: Model type to sweep, either ``"ridge"`` or ``"gru"``.
        feature_sets: Feature families to evaluate.
        windows: Rolling window sizes to evaluate independently.
        lag_steps: Lag offsets to evaluate independently.
        n_jobs: Parallel job count (Ridge only; GRU is always sequential).
        device: Torch device string (GRU only).
        output_path: Optional CSV path for sweep results.

    Returns:
        Results sorted ascending by validation PHM08 score.

    Raises:
        ValueError: If any sweep input is invalid.
    """
    validated_feature_sets = _validate_inputs(
        model, feature_sets, windows, lag_steps, n_jobs=n_jobs
    )
    specs = _build_experiment_specs(validated_feature_sets, windows, lag_steps)

    cfg = load_config(config_path)
    train_raw = load_raw_train(cfg.data)
    train_labeled = add_rul_column(train_raw, max_rul=cfg.data.max_rul)
    train_df, val_df = split_by_engine(
        train_labeled,
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
    )

    rows: list[dict[str, object]]

    if model == "ridge":
        X_train, y_train = split_features_target(train_df)
        X_val, y_val = split_features_target(val_df)
        sensor_drop = list(cfg.features.sensor_cols_to_drop) or None
        rows = list(
            Parallel(n_jobs=n_jobs)(
                delayed(_evaluate_ridge_spec)(
                    spec,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    cfg.model.alpha,
                    sensor_drop,
                    cfg.features.n_modes,
                    cfg.data.random_seed,
                    cfg.data.max_rul,
                )
                for spec in specs
            )
        )
    else:
        from typing import cast

        from turbofan.models.sequence_training import resolve_device

        torch_device = resolve_device(cast(Literal["cpu", "cuda"], device))
        rows = []
        for i, spec in enumerate(specs, 1):
            row = _evaluate_gru_spec(spec, train_df, val_df, cfg, torch_device)
            rows.append(row)
            print(
                f"run {i}/{len(specs)}: "
                f"feature_set={spec.feature_set} "
                f"windows={_format_tuple(spec.windows)} "
                f"lag_steps={_format_tuple(spec.lag_steps)} "
                f"phm08_score={row['phm08_score']:.6f}"
            )

    results = pd.DataFrame(rows).sort_values("phm08_score").reset_index(drop=True)
    if output_path is None:
        output_path = Path(
            f"results/feature_sweep_{model}_{cfg.data.fd_subset.lower()}.csv"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


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
        "--model",
        choices=["ridge", "gru"],
        required=True,
        help="Model type to sweep.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["raw", "rolling_mean", "lag"],
        help="Feature families to evaluate.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Rolling window sizes to evaluate independently.",
    )
    parser.add_argument(
        "--lag-steps",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Lag offsets to evaluate independently.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel job count (Ridge only).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Torch device for GRU training.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for sweep results.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the unified feature engineering sweep CLI."""
    args = _parse_args()
    results = run_feature_sweep(
        config_path=args.config,
        model=args.model,
        feature_sets=args.feature_sets,
        windows=args.windows,
        lag_steps=args.lag_steps,
        n_jobs=args.n_jobs,
        device=args.device,
        output_path=args.output,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
