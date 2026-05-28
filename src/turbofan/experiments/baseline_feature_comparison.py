"""Compare baseline Ridge models across sensor feature families."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from turbofan.config.schema import load_config
from turbofan.data.loader import load_raw_train
from turbofan.features.engineering import FeatureSet
from turbofan.models.baseline import build_baseline_pipeline
from turbofan.models.evaluate import add_rul_column, split_features_target
from turbofan.models.metrics import regression_metrics
from turbofan.models.split import split_by_engine

SUPPORTED_FEATURE_SETS: set[str] = {
    "raw",
    "raw_plus_rolling_mean",
    "raw_plus_rolling_stats",
    "rolling_mean",
    "rolling_stats",
    "lag",
}


class ExperimentSpec(NamedTuple):
    """Single baseline feature comparison experiment.

    Args:
        feature_set: Sensor-derived feature family to train.
        windows: Rolling window sizes for the feature family.
    """

    feature_set: FeatureSet
    windows: tuple[int, ...]


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
        default=["raw", "raw_plus_rolling_mean", "rolling_mean"],
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
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel job count for independent experiment rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for comparison results.",
    )
    return parser.parse_args()


def _validate_feature_sets(
    feature_sets: list[str],
) -> list[FeatureSet]:
    """Validate feature-set names.

    Args:
        feature_sets: Requested feature-set names.

    Returns:
        Validated feature-set names cast to FeatureSet.

    Raises:
        ValueError: If a feature-set name is unsupported.
    """
    invalid = sorted(set(feature_sets) - SUPPORTED_FEATURE_SETS)
    if invalid:
        raise ValueError(f"Unsupported feature_set: {invalid[0]}")
    return list(feature_sets)  # type: ignore[return-value]


def _validate_inputs(
    feature_sets: list[str],
    windows: list[int],
    n_jobs: int,
) -> list[FeatureSet]:
    """Validate comparison inputs.

    Args:
        feature_sets: Requested feature-set names.
        windows: Requested rolling windows.
        n_jobs: Joblib parallel worker count.

    Returns:
        Validated feature-set names.

    Raises:
        ValueError: If any input is invalid.
    """
    if any(window <= 0 for window in windows):
        raise ValueError("All window sizes must be positive.")
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero.")
    return _validate_feature_sets(feature_sets)


def _build_experiment_specs(
    feature_sets: list[FeatureSet],
    windows: list[int],
) -> list[ExperimentSpec]:
    """Build independent experiment specs from requested feature families.

    Args:
        feature_sets: Validated feature-set names.
        windows: Positive rolling window sizes.

    Returns:
        Experiment specs. Raw is evaluated once; rolling variants are
        evaluated once per window.
    """
    specs: list[ExperimentSpec] = []
    for feature_set in feature_sets:
        if feature_set == "raw":
            specs.append(ExperimentSpec(feature_set=feature_set, windows=()))
            continue
        specs.extend(
            ExperimentSpec(feature_set=feature_set, windows=(window,))
            for window in windows
        )
    return specs


def _format_windows(windows: tuple[int, ...]) -> str:
    """Format window tuple for tabular output.

    Args:
        windows: Rolling window sizes.

    Returns:
        Comma-separated window string, or an empty string for no windows.
    """
    return ",".join(str(window) for window in windows)


def _evaluate_spec(
    spec: ExperimentSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series[float],
    X_val: pd.DataFrame,
    y_val: pd.Series[float],
    model_name: Literal["ridge"],
    alpha: float,
    sensor_drop: list[str] | None,
    max_rul: int,
) -> dict[str, float | int | str]:
    """Fit and evaluate one feature comparison experiment.

    Args:
        spec: Experiment configuration.
        X_train: Training feature rows.
        y_train: Training RUL labels.
        X_val: Validation feature rows.
        y_val: Validation RUL labels.
        model_name: Baseline model name.
        alpha: Ridge regularization strength.
        sensor_drop: Sensor column names to remove before feature engineering.
        max_rul: Maximum configured RUL value.

    Returns:
        Result row with feature metadata and validation metrics.
    """
    estimator = build_baseline_pipeline(
        model_name=model_name,
        alpha=alpha,
        windows=list(spec.windows) or None,
        sensor_drop=sensor_drop,
        feature_set=spec.feature_set,
    )
    estimator.fit(X_train, y_train)
    raw_pred = np.asarray(estimator.predict(X_val), dtype=np.float64)
    clipped_pred = np.clip(raw_pred, 0.0, float(max_rul))
    metrics = regression_metrics(y_val, clipped_pred)
    model = estimator.named_steps["model"]
    return {
        "feature_set": spec.feature_set,
        "windows": _format_windows(spec.windows),
        "alpha": alpha,
        "n_features": int(len(model.feature_names_in_)),
        "raw_prediction_min": float(raw_pred.min()),
        "raw_prediction_max": float(raw_pred.max()),
        **metrics,
    }


def run_feature_comparison(
    config_path: Path,
    feature_sets: list[str],
    windows: list[int],
    n_jobs: int,
) -> pd.DataFrame:
    """Train and evaluate baseline feature-family comparisons.

    Args:
        config_path: Project config path.
        feature_sets: Feature families to evaluate.
        windows: Rolling window sizes to evaluate independently.
        n_jobs: Parallel job count for independent experiments.

    Returns:
        Results sorted by validation PHM08 score.

    Raises:
        ValueError: If inputs are invalid.
    """
    validated_feature_sets = _validate_inputs(feature_sets, windows, n_jobs)
    specs = _build_experiment_specs(validated_feature_sets, windows)

    cfg = load_config(config_path)
    train_raw = load_raw_train(cfg.data)
    train_labeled = add_rul_column(train_raw, max_rul=cfg.data.max_rul)
    train_df, val_df = split_by_engine(
        train_labeled,
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
    )
    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)

    sensor_drop = cfg.features.sensor_cols_to_drop or None
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_spec)(
            spec,
            X_train,
            y_train,
            X_val,
            y_val,
            cfg.model.name,
            cfg.model.alpha,
            sensor_drop,
            cfg.data.max_rul,
        )
        for spec in specs
    )
    return pd.DataFrame(rows).sort_values("phm08_score").reset_index(drop=True)


def main() -> None:
    """Run the baseline feature comparison CLI."""
    args = _parse_args()
    results = run_feature_comparison(
        config_path=args.config,
        feature_sets=args.feature_sets,
        windows=args.windows,
        n_jobs=args.n_jobs,
    )
    print(results.to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
