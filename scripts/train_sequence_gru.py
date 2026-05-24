"""Train a GRU sequence RUL model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from turbofan.config.schema import ProjectConfig, load_config
from turbofan.data.loader import load_raw_test, load_raw_train, load_rul_labels
from turbofan.models.artifacts import create_run_dir, save_json, save_predictions
from turbofan.models.evaluate import add_rul_column, align_official_test_labels
from turbofan.models.gru import GRURULRegressor
from turbofan.models.metrics import regression_metrics
from turbofan.models.sequence_training import (
    predict_windows,
    resolve_device,
    seed_everything,
    train_gru_model,
)
from turbofan.models.split import split_by_engine
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.normalize import SequenceNormalizer, default_feature_cols
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
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate labeled sequence windows.

    Args:
        model: Trained GRU model.
        windows: Labeled sequence windows.
        device: Torch device used for inference.
        batch_size: Prediction batch size.

    Returns:
        Metrics and prediction artifact rows.
    """
    loader = build_sequence_loader(windows, batch_size=batch_size, shuffle=False)
    y_pred = np.clip(predict_windows(model, loader, device), 0.0, None)
    y_true = windows.y.astype(np.float64)
    metrics = regression_metrics(y_true, y_pred)
    return metrics, _prediction_frame(windows, y_true, y_pred)


def _align_official_labels_to_eligible_engines(
    metadata: pd.DataFrame,
    rul_labels: pd.Series,
) -> pd.Series:
    """Align official labels to eligible sequence test engines.

    C-MAPSS official RUL labels are ordered by engine ID, while final sequence
    windows can skip engines shorter than ``window_size``. This selects labels
    for the eligible engine IDs before applying the standard count check.

    Args:
        metadata: Final-window metadata containing eligible ``engine_id`` rows.
        rul_labels: Official RUL labels in full test engine order.

    Returns:
        Float RUL Series aligned to ``metadata``.

    Raises:
        KeyError: If ``metadata`` lacks ``engine_id``.
        ValueError: If an eligible engine ID cannot be mapped to a label row.
    """
    engine_ids = metadata["engine_id"].to_numpy(dtype=np.int64)
    label_positions = engine_ids - 1
    if np.any(label_positions < 0) or np.any(label_positions >= len(rul_labels)):
        raise ValueError(
            "Official RUL labels must include a row for every eligible test engine."
        )

    eligible_labels = rul_labels.iloc[label_positions].reset_index(drop=True)
    return align_official_test_labels(metadata.reset_index(drop=True), eligible_labels)


def _evaluate_official_test(
    cfg: ProjectConfig,
    model: GRURULRegressor,
    normalizer: SequenceNormalizer,
    feature_cols: list[str],
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame] | None:
    """Evaluate final-cycle official test labels when files exist.

    Args:
        cfg: Project config.
        model: Trained GRU model.
        normalizer: Fitted sequence feature normalizer.
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

    test_df = normalizer.transform(test_raw)
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
    y_pred = np.clip(predict_windows(model, loader, device), 0.0, None)
    y_true = _align_official_labels_to_eligible_engines(
        test_windows.metadata,
        rul_labels,
    )
    metrics = regression_metrics(y_true, y_pred)
    predictions = _prediction_frame(test_windows, y_true, y_pred)
    return metrics, predictions


def _model_payload(
    model: GRURULRegressor,
    cfg: ProjectConfig,
    feature_cols: list[str],
    normalizer: SequenceNormalizer,
) -> dict[str, object]:
    """Build the serialized model checkpoint payload.

    Args:
        model: Trained GRU model.
        cfg: Project config.
        feature_cols: Feature columns used by the model.
        normalizer: Fitted sequence feature normalizer.

    Returns:
        Torch-serializable model payload.
    """
    return {
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "sequence_config": cfg.sequence.model_dump(mode="json"),
        "normalizer_means": {
            str(key): float(value)
            for key, value in normalizer.means_.to_dict().items()
        },
        "normalizer_stds": {
            str(key): float(value)
            for key, value in normalizer.stds_.to_dict().items()
        },
        "fd_subset": cfg.data.fd_subset,
        "random_seed": cfg.data.random_seed,
    }


def main() -> None:
    """Train, evaluate, and persist a GRU sequence model run."""
    args = _parse_args()
    cfg = load_config(args.config)
    if cfg.sequence.architecture != "gru":
        raise ValueError("Sequence training CLI requires architecture='gru'.")

    device = resolve_device(cfg.sequence.device)
    feature_cols = default_feature_cols()

    train_raw = load_raw_train(cfg.data)
    train_labeled = add_rul_column(train_raw, max_rul=cfg.data.max_rul)
    train_df, val_df = split_by_engine(
        train_labeled,
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
    )

    normalizer = SequenceNormalizer(feature_cols=feature_cols)
    train_normalized = normalizer.fit_transform(train_df)
    val_normalized = normalizer.transform(val_df)

    train_windows = build_sliding_windows(
        train_normalized,
        feature_cols=feature_cols,
        window_size=cfg.sequence.window_size,
    )
    validation_final_windows = build_final_windows(
        val_normalized,
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
    validation_final_loader = build_sequence_loader(
        validation_final_windows,
        batch_size=cfg.sequence.batch_size,
        shuffle=False,
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
    result = train_gru_model(
        model=model,
        train_loader=train_loader,
        validation_final_loader=validation_final_loader,
        validation_windows_loader=validation_windows_loader,
        config=cfg.sequence,
        device=device,
        random_seed=cfg.data.random_seed,
    )

    final_metrics, final_predictions = _evaluate_windows(
        result.model,
        validation_final_windows,
        device,
        cfg.sequence.batch_size,
    )
    window_metrics, window_predictions = _evaluate_windows(
        result.model,
        validation_windows,
        device,
        cfg.sequence.batch_size,
    )

    run_dir = create_run_dir(cfg.sequence.artifact_dir, "sequence_gru")
    metrics_payload: dict[str, object] = {
        "validation_final_window": final_metrics,
        "validation_windows": window_metrics,
    }

    official = _evaluate_official_test(
        cfg,
        result.model,
        normalizer,
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
        _model_payload(result.model, cfg, feature_cols, normalizer),
        run_dir / "model.pt",
    )
    save_json(metrics_payload, run_dir / "metrics.json")
    save_json(_config_to_dict(cfg), run_dir / "config.json")
    result.history.to_csv(run_dir / "training_history.csv", index=False)
    save_predictions(
        final_predictions,
        run_dir / "validation_final_window_predictions.csv",
    )
    save_predictions(
        window_predictions,
        run_dir / "validation_window_predictions.csv",
    )

    print(f"run_dir: {run_dir}")
    print(f"validation_final_window rmse: {final_metrics['rmse']:.6f}")
    print(f"validation_final_window mae: {final_metrics['mae']:.6f}")
    print(
        "validation_final_window phm08_score: "
        f"{final_metrics['phm08_score']:.6f}"
    )


if __name__ == "__main__":
    main()
