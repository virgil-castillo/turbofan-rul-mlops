"""Compare Ridge alpha values for the tabular baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from turbofan.config.schema import load_config
from turbofan.data.loader import load_raw_train
from turbofan.models.baseline import build_baseline_pipeline
from turbofan.models.evaluate import add_rul_column, split_features_target
from turbofan.models.metrics import regression_metrics
from turbofan.models.split import split_by_engine


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
        "--alphas",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 100.0, 1000.0],
        help="Ridge alpha values to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for sweep results.",
    )
    return parser.parse_args()


def run_alpha_sweep(config_path: Path, alphas: list[float]) -> pd.DataFrame:
    """Train and evaluate the baseline for multiple Ridge alpha values.

    Args:
        config_path: Project config path.
        alphas: Positive Ridge alpha values to evaluate.

    Returns:
        DataFrame with one row per alpha and validation metrics.

    Raises:
        ValueError: If any alpha is non-positive.
    """
    if any(alpha <= 0.0 for alpha in alphas):
        raise ValueError("All alpha values must be positive.")

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

    rows: list[dict[str, float]] = []
    for alpha in alphas:
        estimator = build_baseline_pipeline(
            model_name=cfg.model.name,
            alpha=alpha,
            sensor_drop=cfg.features.sensor_cols_to_drop or None,
        )
        estimator.fit(X_train, y_train)
        raw_pred = np.asarray(estimator.predict(X_val), dtype=np.float64)
        clipped_pred = np.clip(raw_pred, 0.0, float(cfg.data.max_rul))
        metrics = regression_metrics(y_val, clipped_pred)
        rows.append(
            {
                "alpha": alpha,
                "raw_prediction_min": float(raw_pred.min()),
                "raw_prediction_max": float(raw_pred.max()),
                **metrics,
            }
        )

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def main() -> None:
    """Run the alpha sweep CLI."""
    args = _parse_args()
    results = run_alpha_sweep(args.config, args.alphas)
    print(results.to_string(index=False))
    if args.output is not None:
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
