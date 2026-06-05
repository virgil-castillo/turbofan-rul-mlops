"""Train a tabular baseline RUL model."""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from turbofan import registry, tracking
from turbofan.config.schema import ProjectConfig, load_config
from turbofan.data.loader import load_raw_test, load_raw_train, load_rul_labels
from turbofan.models.artifacts import (
    create_run_dir,
    save_json,
    save_predictions,
)
from turbofan.models.baseline import build_baseline_pipeline
from turbofan.models.evaluate import (
    add_rul_column,
    align_official_test_labels,
    select_last_cycle_per_engine,
    split_features_target,
)
from turbofan.models.metrics import official_test_metrics, regression_metrics
from turbofan.models.split import split_by_engine
from turbofan.utils.logging import get_logger, run_file_logging, setup_logging

logger = get_logger(__name__)


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
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging verbosity (falls back to the LOG_LEVEL env var or INFO).",
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


def _clip_rul_predictions(
    values: npt.ArrayLike,
    rul_cap: int,
) -> npt.NDArray[np.float64]:
    """Clip raw predictions into the configured RUL range.

    Args:
        values: Raw model predictions.
        rul_cap: Maximum allowed RUL value.

    Returns:
        Float64 predictions clipped to ``[0, rul_cap]``.
    """
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, float(rul_cap))


def _predict_with_clipping(
    estimator: Pipeline,
    rows: pd.DataFrame,
    rul_cap: int,
    label: str,
) -> npt.NDArray[np.float64]:
    """Predict rows, log raw prediction range, and clip to valid RUL bounds.

    Args:
        estimator: Fitted sklearn estimator.
        rows: Feature rows to predict.
        rul_cap: Maximum allowed RUL value.
        label: Human-readable prediction set label for logs.

    Returns:
        Float64 predictions clipped to ``[0, rul_cap]``.
    """
    raw = np.asarray(estimator.predict(rows), dtype=np.float64)
    logger.debug(
        "%s raw prediction min/max: %.6f/%.6f", label, raw.min(), raw.max()
    )
    return _clip_rul_predictions(raw, rul_cap=rul_cap)


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
    all_pred = _predict_with_clipping(
        estimator,
        test_raw,
        rul_cap=cfg.data.max_rul,
        label="official_test",
    )
    pred_rows = test_raw[["engine_id", "cycle"]].copy()
    pred_rows["prediction"] = all_pred
    last_pred_rows = select_last_cycle_per_engine(pred_rows)
    y_pred = last_pred_rows["prediction"].to_numpy(dtype=np.float64)
    metrics = official_test_metrics(y_true, y_pred)
    predictions = _prediction_frame(last_rows, y_true, y_pred)
    return metrics, predictions


def main() -> None:
    """Train, evaluate, and persist a baseline model run."""
    args = _parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    tmp_log_dir = Path(tempfile.mkdtemp())
    tmp_run_log = tmp_log_dir / "run.log"
    try:
        with run_file_logging(tmp_run_log):
            logger.info("loading training data for %s", cfg.data.fd_subset)
            train_raw = load_raw_train(cfg.data)
            train_labeled = add_rul_column(train_raw, max_rul=cfg.data.max_rul)
            train_df, val_df = split_by_engine(
                train_labeled,
                test_size=cfg.data.test_size,
                random_seed=cfg.data.random_seed,
            )

            X_train, y_train = split_features_target(train_df)
            X_val, y_val = split_features_target(val_df)

            rf = cfg.features.for_model("ridge")
            estimator = build_baseline_pipeline(
                model_name=cfg.model.name,
                alpha=cfg.model.alpha,
                feature_families=rf.feature_families,
                windows=rf.windows,
                lag_steps=rf.lag_steps,
                sensor_drop=cfg.features.sensor_cols_to_drop or None,
                n_modes=cfg.features.n_modes,
                random_state=cfg.data.random_seed,
            )
            logger.info("fitting %s baseline pipeline", cfg.model.name)
            training_started = perf_counter()
            estimator.fit(X_train, y_train)
            training_duration_seconds = perf_counter() - training_started

            val_pred = _predict_with_clipping(
                estimator,
                X_val,
                rul_cap=cfg.data.max_rul,
                label="validation",
            )
            val_metrics = regression_metrics(y_val, val_pred)
            val_predictions = _prediction_frame(X_val, y_val, val_pred)

            tracking.configure_mlflow()
            mlflow.set_experiment(tracking.TRAINING_EXPERIMENT)
            with mlflow.start_run():
                run_dir = create_run_dir(cfg.model.artifact_dir, "baseline")
                metrics_payload: dict[str, object] = {"validation": val_metrics}
                run_metrics: dict[str, float] = {
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "training_duration_seconds": training_duration_seconds,
                }

                official = _evaluate_official_test(cfg, estimator)
                if official is not None:
                    official_metrics, official_predictions = official
                    metrics_payload["official_test"] = official_metrics
                    run_metrics["official_rmse"] = official_metrics["rmse"]
                    run_metrics["official_mae"] = official_metrics["mae"]
                    run_metrics["official_phm08"] = official_metrics["phm08_score"]
                    save_predictions(
                        official_predictions,
                        run_dir / "official_test_predictions.csv",
                    )
                else:
                    logger.warning(
                        "official test evaluation skipped: "
                        "test or RUL files not found"
                    )

                save_json(metrics_payload, run_dir / "metrics.json")
                save_json(_config_to_dict(cfg), run_dir / "config.json")
                save_predictions(
                    val_predictions, run_dir / "validation_predictions.csv"
                )
                logger.info("saved baseline run to %s", run_dir)

                tracking.log_params(
                    {
                        "alpha": cfg.model.alpha,
                        "feature_families": rf.feature_families,
                        "windows": rf.windows,
                        "lag_steps": rf.lag_steps,
                        "seed": cfg.data.random_seed,
                    }
                )
                tracking.log_metrics(run_metrics)
                tracking.set_tags(
                    {
                        "model_type": "ridge",
                        "run_type": "production",
                        "run_dir": str(run_dir),
                    }
                )
                mlflow.log_artifact(str(tmp_run_log), artifact_path="logs")

                version = registry.log_and_register(
                    estimator, model_type="ridge", subset=cfg.data.fd_subset
                )
                mlflow.log_artifact(
                    str(run_dir / "validation_predictions.csv"),
                    artifact_path="predictions",
                )
                if official is not None:
                    mlflow.log_artifact(
                        str(run_dir / "official_test_predictions.csv"),
                        artifact_path="predictions",
                    )
                logger.info(
                    "registered %s version %d",
                    registry.model_name("ridge", cfg.data.fd_subset),
                    version,
                )

                print(f"run_dir: {run_dir}")
                print(f"validation rmse: {val_metrics['rmse']:.6f}")
                print(f"validation mae: {val_metrics['mae']:.6f}")
    finally:
        shutil.rmtree(tmp_log_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
