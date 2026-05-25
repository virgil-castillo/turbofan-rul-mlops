"""Sweep GRU sequence model hyperparameters on the validation split."""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
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
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.normalize import SequenceNormalizer, default_feature_cols
from turbofan.sequences.windowing import build_final_windows, build_sliding_windows

RESULT_COLUMNS = [
    "window_size",
    "hidden_size",
    "learning_rate",
    "best_epoch",
    "rmse",
    "mae",
    "phm08_score",
]


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
        "--window-sizes",
        type=int,
        nargs="+",
        default=[15, 20, 30, 45],
        help="Window sizes to sweep.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="GRU hidden sizes to sweep.",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-3, 5e-4, 1e-4],
        help="Adam learning rates to sweep.",
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
    window_sizes: list[int],
    hidden_sizes: list[int],
    learning_rates: list[float],
) -> None:
    """Validate GRU sweep grid values.

    Args:
        window_sizes: Candidate sequence window sizes.
        hidden_sizes: Candidate GRU hidden sizes.
        learning_rates: Candidate Adam learning rates.

    Raises:
        ValueError: If any grid is empty or contains non-positive values.
    """
    if not window_sizes or any(window_size <= 0 for window_size in window_sizes):
        raise ValueError("All window sizes must be positive.")
    if not hidden_sizes or any(hidden_size <= 0 for hidden_size in hidden_sizes):
        raise ValueError("All hidden sizes must be positive.")
    if not learning_rates or any(
        learning_rate <= 0.0 for learning_rate in learning_rates
    ):
        raise ValueError("All learning rates must be positive.")


def _append_incremental_row(
    row: dict[str, float | int],
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


def run_gru_sweep(
    config_path: Path,
    window_sizes: list[int],
    hidden_sizes: list[int],
    learning_rates: list[float],
    device: str,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Train and evaluate a GRU validation hyperparameter sweep.

    Args:
        config_path: Project config path.
        window_sizes: Sequence window sizes to evaluate.
        hidden_sizes: GRU hidden state widths to evaluate.
        learning_rates: Adam learning rates to evaluate.
        device: Requested torch device, either ``"cpu"`` or ``"cuda"``.
        output_path: Optional CSV path for incremental and final results.

    Returns:
        Results sorted by validation PHM08 score.

    Raises:
        ValueError: If sweep inputs are invalid.
    """
    _validate_inputs(window_sizes, hidden_sizes, learning_rates)
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'.")

    cfg = load_config(config_path)
    if cfg.sequence.architecture != "gru":
        raise ValueError("GRU sweep requires sequence architecture='gru'.")

    torch_device = resolve_device(cast(Literal["cpu", "cuda"], device))
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

    rows: list[dict[str, float | int]] = []
    specs = list(product(window_sizes, hidden_sizes, learning_rates))
    total_runs = len(specs)

    for run_idx, (window_size, hidden_size, learning_rate) in enumerate(specs, 1):
        spec_cfg = cfg.sequence.model_copy(
            update={
                "window_size": window_size,
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
            }
        )
        train_windows = build_sliding_windows(
            train_normalized,
            feature_cols=feature_cols,
            window_size=window_size,
        )
        validation_final_windows = build_final_windows(
            val_normalized,
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
            batch_size=spec_cfg.batch_size,
            shuffle=True,
        )
        validation_final_loader = build_sequence_loader(
            validation_final_windows,
            batch_size=spec_cfg.batch_size,
            shuffle=False,
        )
        validation_windows_loader = build_sequence_loader(
            validation_windows,
            batch_size=spec_cfg.batch_size,
            shuffle=False,
        )

        seed_everything(cfg.data.random_seed)
        model = GRURULRegressor(
            input_size=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=spec_cfg.num_layers,
            dropout=spec_cfg.dropout,
        )
        result = train_gru_model(
            model=model,
            train_loader=train_loader,
            validation_final_loader=validation_final_loader,
            validation_windows_loader=validation_windows_loader,
            config=spec_cfg,
            device=torch_device,
            random_seed=cfg.data.random_seed,
        )

        predictions = np.clip(
            predict_windows(result.model, validation_final_loader, torch_device),
            0.0,
            None,
        )
        metrics = regression_metrics(
            validation_final_windows.y.astype(np.float64),
            predictions,
        )
        row: dict[str, float | int] = {
            "window_size": window_size,
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "best_epoch": result.best_epoch,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "phm08_score": metrics["phm08_score"],
        }
        rows.append(row)
        if output_path is not None:
            _append_incremental_row(
                row,
                output_path,
                append=len(rows) > 1,
            )
        print(
            f"run {run_idx}/{total_runs}: "
            f"window_size={window_size} hidden_size={hidden_size} "
            f"learning_rate={learning_rate:g} "
            f"phm08_score={metrics['phm08_score']:.6f}"
        )

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return results.sort_values("phm08_score").reset_index(drop=True)


def main() -> None:
    """Run the GRU hyperparameter sweep CLI."""
    args = _parse_args()
    results = run_gru_sweep(
        config_path=args.config,
        window_sizes=args.window_sizes,
        hidden_sizes=args.hidden_sizes,
        learning_rates=args.learning_rates,
        device=args.device,
        output_path=args.output,
    )
    print(results.to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
