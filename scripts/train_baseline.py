"""Train a tabular baseline RUL model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from turbofan.config.schema import ProjectConfig, load_config
from turbofan.data.loader import load_raw_test, load_raw_train, load_rul_labels
from turbofan.models.artifacts import (
    create_run_dir,
    save_json,
    save_model,
    save_predictions,
)
from turbofan.models.baseline import build_baseline_pipeline
from turbofan.models.evaluate import (
    add_rul_column,
    align_official_test_labels,
    select_last_cycle_per_engine,
    split_features_target,
)
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
    return parser.parse_args()


def _config_to_dict(cfg: ProjectConfig) -> dict[str, object]:
    """Convert config to JSON-friendly dict.

    Args:
        cfg: Project config.

    Returns:
        Dictionary with JSON-friendly values.
    """
    return cfg.model_dump(mode="json")


def _manifest_payload(run_dir: Path, model_type: str) -> dict[str, object]:
    """Build a schema-version-1 model manifest payload.

    Args:
        run_dir: Created training run directory.
        model_type: Persisted baseline model type.

    Returns:
        JSON-serializable model manifest.
    """
    return {
        "schema_version": 1,
        "model_type": model_type,
        "artifact_id": f"baseline/{run_dir.name}",
        "prediction_scope": "row",
        "model_path": "model.joblib",
        "config_path": "config.json",
        "metrics_path": "metrics.json",
    }


def _prediction_frame(
    rows: pd.DataFrame,
    y_true: pd.Series,
    y_pred: npt.NDArray[np.float64],
) -> pd.DataFrame:
    """Build a prediction artifact DataFrame.

    Args:
        rows: Feature rows used for prediction.
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        DataFrame with identifiers, targets, and predictions.
    """
    return pd.DataFrame(
        {
            "engine_id": rows["engine_id"].to_numpy(),
            "cycle": rows["cycle"].to_numpy(),
            "rul": y_true.to_numpy(dtype=np.float64),
            "prediction": y_pred,
        }
    )


def _evaluate_official_test(
    cfg: ProjectConfig,
    estimator: Pipeline,
) -> tuple[dict[str, float], pd.DataFrame] | None:
    """Evaluate final-cycle official test labels when files exist.

    Args:
        cfg: Project config.
        estimator: Fitted sklearn estimator.

    Returns:
        Metrics and prediction rows, or None when official files are missing.
    """
    try:
        test_raw = load_raw_test(cfg.data)
        rul_labels = load_rul_labels(cfg.data)
    except FileNotFoundError:
        return None

    last_rows = select_last_cycle_per_engine(test_raw)
    y_true = align_official_test_labels(last_rows, rul_labels)
    all_pred = np.clip(
        np.asarray(estimator.predict(test_raw), dtype=np.float64),
        0.0,
        None,
    )
    pred_rows = test_raw[["engine_id", "cycle"]].copy()
    pred_rows["prediction"] = all_pred
    last_pred_rows = select_last_cycle_per_engine(pred_rows)
    y_pred = np.clip(
        last_pred_rows["prediction"].to_numpy(dtype=np.float64),
        0.0,
        None,
    )
    metrics = regression_metrics(y_true, y_pred)
    predictions = _prediction_frame(last_rows, y_true, y_pred)
    return metrics, predictions


def main() -> None:
    """Train, evaluate, and persist a baseline model run."""
    args = _parse_args()
    cfg = load_config(args.config)

    train_raw = load_raw_train(cfg.data)
    train_labeled = add_rul_column(train_raw, max_rul=cfg.data.max_rul)
    train_df, val_df = split_by_engine(
        train_labeled,
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
    )

    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)

    estimator = build_baseline_pipeline(
        model_name=cfg.model.name,
        alpha=cfg.model.alpha,
        sensor_std_threshold=cfg.features.sensor_std_threshold,
        sensor_keep=cfg.features.sensor_keep,
    )
    estimator.fit(X_train, y_train)

    val_pred = np.clip(
        np.asarray(estimator.predict(X_val), dtype=np.float64),
        0.0,
        None,
    )
    val_metrics = regression_metrics(y_val, val_pred)
    val_predictions = _prediction_frame(X_val, y_val, val_pred)

    run_dir = create_run_dir(cfg.model.artifact_dir, "baseline")
    metrics_payload: dict[str, object] = {"validation": val_metrics}

    official = _evaluate_official_test(cfg, estimator)
    if official is not None:
        official_metrics, official_predictions = official
        metrics_payload["official_test"] = official_metrics
        save_predictions(
            official_predictions,
            run_dir / "official_test_predictions.csv",
        )
    else:
        print("official test evaluation skipped: test or RUL files not found")

    save_model(estimator, run_dir / "model.joblib")
    save_json(metrics_payload, run_dir / "metrics.json")
    save_json(_config_to_dict(cfg), run_dir / "config.json")
    save_json(
        _manifest_payload(run_dir, cfg.model.name),
        run_dir / "model_manifest.json",
    )
    save_predictions(val_predictions, run_dir / "validation_predictions.csv")

    print(f"run_dir: {run_dir}")
    print(f"validation rmse: {val_metrics['rmse']:.6f}")
    print(f"validation mae: {val_metrics['mae']:.6f}")
    print(f"validation phm08_score: {val_metrics['phm08_score']:.6f}")


if __name__ == "__main__":
    main()
