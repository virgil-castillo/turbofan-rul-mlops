"""Train a sequence RUL model (GRU or LSTM) selected by config.

The architecture comes from ``sequence.architecture`` in the project config; the
model is constructed through the sequence registry and registered under its
per-architecture registered-model name (``turbofan-<arch>-<subset>``).

Data preparation, model construction/training, and official-test evaluation are
delegated to :mod:`turbofan.models.sequence_pipeline`, shared with the
official-eval sweep and the feature-family screen so the three cannot drift
apart. The split and feature pipeline use ``cfg.data.random_seed`` as the data
seed and model initialisation/training reuse the same seed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch import nn

from turbofan import registry
from turbofan.config import schema
from turbofan.config.schema import ProjectConfig
from turbofan.models import artifacts, metrics, sequence_pipeline, sequence_training
from turbofan.models.sequence_training import SequenceLoader
from turbofan.sequences.windowing import WindowedSequences
from turbofan.utils import logging as turbofan_logging

logger = turbofan_logging.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional command-line arguments. Uses ``sys.argv[1:]`` when
            ``None``.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs/default.yaml",
        help="Path to YAML project config.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging verbosity (falls back to the LOG_LEVEL env var or INFO).",
    )
    return parser.parse_args(argv)


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
    loader: SequenceLoader,
    windows: WindowedSequences,
    device: torch.device,
    max_rul: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate labeled validation windows and build the prediction artifact.

    Args:
        model: Trained sequence model.
        loader: Sequential loader over ``windows``.
        windows: Labeled validation windows.
        device: Torch device used for inference.
        max_rul: Maximum RUL cap for prediction rescaling.

    Returns:
        Metrics and prediction artifact rows.
    """
    metrics, y_true, y_pred = sequence_pipeline.evaluate_window_metrics(
        model, loader, windows, device=device, max_rul=max_rul
    )
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
        official = sequence_pipeline.predict_sequence_official(
            cfg.data,
            pipeline=pipeline,
            feature_cols=feature_cols,
            model=model,
            device=device,
            window_size=cfg.sequence.window_size,
            batch_size=cfg.sequence.batch_size,
            max_rul=cfg.data.max_rul,
        )
    except FileNotFoundError:
        return None
    official_metrics = metrics.official_test_metrics(
        official.y_true, official.y_pred
    )
    predictions = _prediction_frame(
        official.windows, official.y_true, official.y_pred
    )
    return official_metrics, predictions


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
        "feature_pipeline": pipeline,
        "sequence_config": cfg.sequence.model_dump(mode="json"),
        "normalizer_type": "operating_mode",
        "normalizer_payload": normalizer.to_payload(),
        "fd_subset": cfg.data.fd_subset,
        "random_seed": cfg.data.random_seed,
        "max_rul": cfg.data.max_rul,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Train, evaluate, and persist a sequence model run for any RNN.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    turbofan_logging.setup_logging(args.log_level)
    cfg = schema.load_config(args.config)
    architecture = cfg.sequence.architecture

    device = sequence_training.resolve_device(cfg.sequence.device)

    tmp_log_dir = Path(tempfile.mkdtemp())
    tmp_run_log = tmp_log_dir / "run.log"
    try:
        with turbofan_logging.run_file_logging(tmp_run_log):
            logger.info("loading training data for %s", cfg.data.fd_subset)
            sf = cfg.features.for_model(architecture)
            prepared = sequence_pipeline.prepare_sequence_data(
                cfg.data,
                feature_families=sf.feature_families,
                windows=sf.windows,
                lag_steps=sf.lag_steps,
                sensor_drop=cfg.features.sensor_cols_to_drop or None,
                n_modes=cfg.features.n_modes,
                data_seed=cfg.data.random_seed,
                max_rul=cfg.data.max_rul,
                test_size=cfg.data.test_size,
                window_size=cfg.sequence.window_size,
                batch_size=cfg.sequence.batch_size,
            )
            feature_cols = prepared.feature_cols

            logger.info(
                "training %s for up to %d epochs", architecture, cfg.sequence.epochs
            )
            training_started = perf_counter()
            result = sequence_pipeline.train_prepared_sequence(
                prepared,
                cfg.sequence,
                device=device,
                model_seed=cfg.data.random_seed,
                max_rul=cfg.data.max_rul,
            )
            training_duration_seconds = perf_counter() - training_started

            window_metrics, window_predictions = _evaluate_windows(
                result.model,
                prepared.val_loader,
                prepared.val_windows,
                device,
                cfg.data.max_rul,
            )

            registry.tracking.configure_mlflow()
            mlflow.set_experiment(registry.tracking.TRAINING_EXPERIMENT)
            with mlflow.start_run():
                run_dir = artifacts.create_run_dir(
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
                    prepared.pipeline,
                    feature_cols,
                    device,
                )
                if official is not None:
                    official_metrics, official_predictions = official
                    metrics_payload["official_test"] = official_metrics
                    run_metrics["official_rmse"] = official_metrics["rmse"]
                    run_metrics["official_mae"] = official_metrics["mae"]
                    run_metrics["official_phm08"] = official_metrics["phm08_score"]
                    artifacts.save_predictions(
                        official_predictions,
                        run_dir / "official_test_predictions.csv",
                    )
                else:
                    logger.warning(
                        "official test evaluation skipped: "
                        "test or RUL files not found"
                    )

                payload = _model_payload(
                    result.model, cfg, feature_cols, prepared.pipeline
                )
                artifacts.save_json(metrics_payload, run_dir / "metrics.json")
                artifacts.save_json(_config_to_dict(cfg), run_dir / "config.json")
                result.history.to_csv(
                    run_dir / "training_history.csv", index=False
                )
                artifacts.save_predictions(
                    window_predictions,
                    run_dir / "validation_window_predictions.csv",
                )
                logger.info("saved %s run to %s", architecture, run_dir)

                registry.tracking.log_params(
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
                registry.tracking.log_metrics(run_metrics)
                registry.tracking.log_history(result.history)
                registry.tracking.set_tags(
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
