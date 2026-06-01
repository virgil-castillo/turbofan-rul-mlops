"""Train a GRU sequence RUL model."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from turbofan.config.schema import ProjectConfig, load_config
from turbofan.data.loader import load_raw_test, load_raw_train, load_rul_labels
from turbofan.features.pipeline import build_feature_pipeline
from turbofan.models.artifacts import create_run_dir, save_json, save_predictions
from turbofan.models.evaluate import add_rul_column
from turbofan.models.gru import GRURULRegressor
from turbofan.models.metrics import official_test_metrics, regression_metrics
from turbofan.models.sequence_training import (
    predict_windows,
    resolve_device,
    seed_everything,
    train_gru_model,
)
from turbofan.models.split import split_by_engine
from turbofan.models.test_evaluation import align_labels_to_eligible_engines
from turbofan.models.training_log import append_training_log, build_log_entry
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.windowing import (
    WindowedSequences,
    build_final_windows,
    build_sliding_windows,
)


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
    """Convert config to a JSON-friendly dict.

    Args:
        cfg: Project config.

    Returns:
        Dictionary with JSON-friendly values.
    """
    return cfg.model_dump(mode="json")


def _manifest_payload(run_dir: Path) -> dict[str, object]:
    """Build a schema-version-1 GRU model manifest payload.

    Args:
        run_dir: Created training run directory.

    Returns:
        JSON-serializable model manifest.
    """
    return {
        "schema_version": 1,
        "model_type": "gru",
        "artifact_id": f"sequence_gru/{run_dir.name}",
        "prediction_scope": "final_window",
        "model_path": "model.pt",
        "config_path": "config.json",
        "metrics_path": "metrics.json",
    }


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
    model: GRURULRegressor,
    windows: WindowedSequences,
    device: torch.device,
    batch_size: int,
    max_rul: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate labeled sequence windows.

    Args:
        model: Trained GRU model.
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
    model: GRURULRegressor,
    pipeline: Pipeline,
    feature_cols: list[str],
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame] | None:
    """Evaluate final-cycle official test labels when files exist.

    Args:
        cfg: Project config.
        model: Trained GRU model.
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
    model: GRURULRegressor,
    cfg: ProjectConfig,
    feature_cols: list[str],
    pipeline: Pipeline,
) -> dict[str, object]:
    """Build the serialized model checkpoint payload.

    Args:
        model: Trained GRU model.
        cfg: Project config.
        feature_cols: Feature columns used by the model.
        pipeline: Fitted feature pipeline.

    Returns:
        Torch-serializable model payload.
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
    """Train, evaluate, and persist a GRU sequence model run."""
    args = _parse_args()
    cfg = load_config(args.config)
    if cfg.sequence.architecture != "gru":
        raise ValueError("Sequence training CLI requires architecture='gru'.")

    device = resolve_device(cfg.sequence.device)

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
        feature_set=(gf := cfg.features.for_model("gru")).feature_set,
        windows=gf.windows,
        lag_steps=gf.lag_steps,
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
        device=device,
        random_seed=cfg.data.random_seed,
        max_rul=cfg.data.max_rul,
    )
    training_duration_seconds = perf_counter() - training_start

    window_metrics, window_predictions = _evaluate_windows(
        result.model,
        validation_windows,
        device,
        cfg.sequence.batch_size,
        max_rul=cfg.data.max_rul,
    )

    run_dir = create_run_dir(cfg.sequence.artifact_dir, "sequence_gru")
    metrics_payload: dict[str, object] = {
        "validation_windows": window_metrics,
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
        save_predictions(
            official_predictions,
            run_dir / "official_test_predictions.csv",
        )
    else:
        print("official test evaluation skipped: test or RUL files not found")

    torch.save(
        _model_payload(result.model, cfg, feature_cols, pipeline),
        run_dir / "model.pt",
    )
    save_json(metrics_payload, run_dir / "metrics.json")
    save_json(_config_to_dict(cfg), run_dir / "config.json")
    save_json(_manifest_payload(run_dir), run_dir / "model_manifest.json")
    result.history.to_csv(run_dir / "training_history.csv", index=False)
    save_predictions(
        window_predictions,
        run_dir / "validation_window_predictions.csv",
    )

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
        metrics=window_metrics,
        training_duration_seconds=training_duration_seconds,
        device=device.type,
        run_dir=str(run_dir),
        best_epoch=result.best_epoch,
    )
    append_training_log(log_entry)

    print(f"run_dir: {run_dir}")
    print(f"validation_windows rmse: {window_metrics['rmse']:.6f}")
    print(f"validation_windows mae: {window_metrics['mae']:.6f}")
    if official is not None:
        print(f"official_test rmse: {official_metrics['rmse']:.6f}")
        print(f"official_test mae: {official_metrics['mae']:.6f}")
        print(f"official_test phm08_score: {official_metrics['phm08_score']:.6f}")


if __name__ == "__main__":
    main()
