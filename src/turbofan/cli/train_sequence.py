"""Train a sequence RUL model (GRU or LSTM) selected by config.

The architecture comes from ``sequence.architecture`` in the project config; the
model is constructed through the sequence registry and registered under its
per-architecture registered-model name (``turbofan-<arch>-<subset>``). The
``turbofan-train-sequence-gru`` console script is a backward-compatible alias
pointing at this module's :func:`main`.
"""
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
import torch
from sklearn.pipeline import Pipeline
from torch import nn

from turbofan import registry, tracking
from turbofan.config.schema import ProjectConfig, load_config
from turbofan.data.loader import load_raw_test, load_raw_train, load_rul_labels
from turbofan.features.pipeline import build_feature_pipeline
from turbofan.models.artifacts import create_run_dir, save_json, save_predictions
from turbofan.models.evaluate import add_rul_column
from turbofan.models.metrics import official_test_metrics, regression_metrics
from turbofan.models.sequence_models import build_sequence_model
from turbofan.models.sequence_training import (
    predict_windows,
    resolve_device,
    seed_everything,
    train_sequence_model,
)
from turbofan.models.split import split_by_engine
from turbofan.models.test_evaluation import align_labels_to_eligible_engines
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.windowing import (
    WindowedSequences,
    build_final_windows,
    build_sliding_windows,
)
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
    """Convert config to a JSON-friendly dict.

    Args:
        cfg: Project config.

    Returns:
        Dictionary with JSON-friendly values.
    """
    return cfg.model_dump(mode="json")


def _prediction_frame(
    windows: WindowedSequences,
    y_true: npt.NDArray[np.float64] | pd.Series,
    y_pred: npt.NDArray[np.float64],
) -> pd.DataFrame:
    """Build a prediction artifact DataFrame.

    Args:
        windows: Sequence windows used for prediction.
        y_true: Ground-truth RUL values aligned to windows.
        y_pred: Predicted RUL values aligned to windows.

    Returns:
        DataFrame with identifiers, targets, and predictions.
    """
    return pd.DataFrame(
        {
            "engine_id": windows.metadata["engine_id"].to_numpy(),
            "cycle": windows.metadata["cycle"].to_numpy(),
            "rul": np.asarray(y_true, dtype=np.float64),
            "prediction": y_pred,
        }
    )


def _evaluate_windows(
    model: nn.Module,
    windows: WindowedSequences,
    device: torch.device,
    batch_size: int,
    max_rul: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate labeled sequence windows.

    Args:
        model: Trained sequence model.
        windows: Labeled sequence windows.
        device: Torch device used for inference.
        batch_size: Prediction batch size.
        max_rul: Maximum RUL cap for prediction rescaling.

    Returns:
        Metrics and prediction artifact rows.
    """
    loader = build_sequence_loader(windows, batch_size=batch_size, shuffle=False)
    y_pred = np.clip(
        predict_windows(model, loader, device, max_rul=max_rul), 0.0, None
    )
    y_true = windows.y.astype(np.float64)
    metrics = regression_metrics(y_true, y_pred)
    return metrics, _prediction_frame(windows, y_true, y_pred)


def _evaluate_official_test(
    cfg: ProjectConfig,
    model: nn.Module,
    pipeline: Pipeline,
    feature_cols: list[str],
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame] | None:
    """Evaluate final-cycle official test labels when files exist.

    Args:
        cfg: Project config.
        model: Trained sequence model.
        pipeline: Fitted feature pipeline.
        feature_cols: Feature columns used by the model.
        device: Torch device used for inference.

    Returns:
        Metrics and prediction rows, or None when official files are missing.
    """
    try:
        test_raw = load_raw_test(cfg.data)
        rul_labels = load_rul_labels(cfg.data)
    except FileNotFoundError:
        return None

    _test_id_cols = [c for c in ("engine_id", "cycle") if c in test_raw.columns]
    test_features = pipeline.transform(test_raw)
    test_df = pd.concat(
        [
            test_raw[_test_id_cols].reset_index(drop=True),
            test_features.reset_index(drop=True),
        ],
        axis=1,
    )
    test_windows = build_final_windows(
        test_df,
        feature_cols=feature_cols,
        window_size=cfg.sequence.window_size,
        target_col=None,
    )
    loader = build_sequence_loader(
        test_windows,
        batch_size=cfg.sequence.batch_size,
        shuffle=False,
    )
    y_pred = np.clip(
        predict_windows(model, loader, device, max_rul=cfg.data.max_rul), 0.0, None
    )
    y_true = align_labels_to_eligible_engines(
        test_windows.metadata,
        rul_labels,
    )
    metrics = official_test_metrics(y_true, y_pred)
    predictions = _prediction_frame(test_windows, y_true, y_pred)
    return metrics, predictions


def _model_payload(
    model: nn.Module,
    cfg: ProjectConfig,
    feature_cols: list[str],
    pipeline: Pipeline,
) -> dict[str, object]:
    """Build the serialized model checkpoint payload.

    Args:
        model: Trained sequence model.
        cfg: Project config.
        feature_cols: Feature columns used by the model.
        pipeline: Fitted feature pipeline.

    Returns:
        Torch-serializable model payload. The ``sequence_config`` carries the
        ``architecture`` so inference can rebuild the correct recurrent layer.
    """
    from turbofan.preprocessing.normalization import OperatingModeNormalizer

    normalizer = pipeline.named_steps["normalizer"]
    assert isinstance(normalizer, OperatingModeNormalizer)
    return {
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "sequence_config": cfg.sequence.model_dump(mode="json"),
        "normalizer_type": "operating_mode",
        "normalizer_payload": normalizer.to_payload(),
        "fd_subset": cfg.data.fd_subset,
        "random_seed": cfg.data.random_seed,
        "max_rul": cfg.data.max_rul,
    }


def main() -> None:
    """Train, evaluate, and persist a sequence model run for any RNN."""
    args = _parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    architecture = cfg.sequence.architecture

    device = resolve_device(cfg.sequence.device)

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

            pipeline = build_feature_pipeline(
                sensor_drop=cfg.features.sensor_cols_to_drop or None,
                n_modes=cfg.features.n_modes,
                random_state=cfg.data.random_seed,
                feature_families=(
                    sf := cfg.features.for_model(architecture)
                ).feature_families,
                windows=sf.windows,
                lag_steps=sf.lag_steps,
            )
            _id_cols = ["engine_id", "cycle", "rul"]
            train_features = pipeline.fit_transform(train_df)
            val_features = pipeline.transform(val_df)
            feature_cols = pipeline.named_steps["feature_engineer"].feature_cols_

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
            validation_windows = build_sliding_windows(
                val_normalized,
                feature_cols=feature_cols,
                window_size=cfg.sequence.window_size,
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
            model = build_sequence_model(
                architecture,
                input_size=len(feature_cols),
                hidden_size=cfg.sequence.hidden_size,
                num_layers=cfg.sequence.num_layers,
                dropout=cfg.sequence.dropout,
            )
            logger.info(
                "training %s for up to %d epochs", architecture, cfg.sequence.epochs
            )
            training_started = perf_counter()
            result = train_sequence_model(
                model=model,
                train_loader=train_loader,
                validation_windows_loader=validation_windows_loader,
                config=cfg.sequence,
                device=device,
                random_seed=cfg.data.random_seed,
                max_rul=cfg.data.max_rul,
            )
            training_duration_seconds = perf_counter() - training_started

            window_metrics, window_predictions = _evaluate_windows(
                result.model,
                validation_windows,
                device,
                cfg.sequence.batch_size,
                max_rul=cfg.data.max_rul,
            )

            tracking.configure_mlflow()
            mlflow.set_experiment(tracking.TRAINING_EXPERIMENT)
            with mlflow.start_run():
                run_dir = create_run_dir(
                    cfg.sequence.artifact_dir, f"sequence_{architecture}"
                )
                metrics_payload: dict[str, object] = {
                    "validation_windows": window_metrics,
                }
                run_metrics: dict[str, float] = {
                    "val_rmse": window_metrics["rmse"],
                    "val_mae": window_metrics["mae"],
                    "training_duration_seconds": training_duration_seconds,
                }

                official = _evaluate_official_test(
                    cfg,
                    result.model,
                    pipeline,
                    feature_cols,
                    device,
                )
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

                payload = _model_payload(
                    result.model, cfg, feature_cols, pipeline
                )
                save_json(metrics_payload, run_dir / "metrics.json")
                save_json(_config_to_dict(cfg), run_dir / "config.json")
                result.history.to_csv(
                    run_dir / "training_history.csv", index=False
                )
                save_predictions(
                    window_predictions,
                    run_dir / "validation_window_predictions.csv",
                )
                logger.info("saved %s run to %s", architecture, run_dir)

                tracking.log_params(
                    {
                        "architecture": architecture,
                        "window_size": cfg.sequence.window_size,
                        "hidden_size": cfg.sequence.hidden_size,
                        "learning_rate": cfg.sequence.learning_rate,
                        "weight_decay": cfg.sequence.weight_decay,
                        "num_layers": cfg.sequence.num_layers,
                        "dropout": cfg.sequence.dropout,
                        "batch_size": cfg.sequence.batch_size,
                        "epochs": cfg.sequence.epochs,
                        "patience": cfg.sequence.patience,
                        "feature_families": sf.feature_families,
                        "windows": sf.windows,
                        "lag_steps": sf.lag_steps,
                        "seed": cfg.data.random_seed,
                    }
                )
                tracking.log_metrics(run_metrics)
                tracking.log_history(result.history)
                tracking.set_tags(
                    {
                        "model_type": architecture,
                        "run_type": "production",
                        "best_epoch": result.best_epoch,
                        "run_dir": str(run_dir),
                    }
                )
                mlflow.log_artifact(str(tmp_run_log), artifact_path="logs")

                version = registry.log_and_register(
                    payload, model_type=architecture, subset=cfg.data.fd_subset
                )
                mlflow.log_artifact(
                    str(run_dir / "validation_window_predictions.csv"),
                    artifact_path="predictions",
                )
                if official is not None:
                    mlflow.log_artifact(
                        str(run_dir / "official_test_predictions.csv"),
                        artifact_path="predictions",
                    )
                logger.info(
                    "registered %s version %d",
                    registry.model_name(architecture, cfg.data.fd_subset),
                    version,
                )

                print(f"run_dir: {run_dir}")
                print(f"validation_windows rmse: {window_metrics['rmse']:.6f}")
                print(f"validation_windows mae: {window_metrics['mae']:.6f}")
                if official is not None:
                    print(f"official_test rmse: {official_metrics['rmse']:.6f}")
                    print(f"official_test mae: {official_metrics['mae']:.6f}")
                    print(
                        "official_test phm08_score: "
                        f"{official_metrics['phm08_score']:.6f}"
                    )
    finally:
        shutil.rmtree(tmp_log_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
