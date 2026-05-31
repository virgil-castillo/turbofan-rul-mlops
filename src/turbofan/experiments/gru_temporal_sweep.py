"""Stage 1 GRU temporal-context sweep: sequence_window_size × rolling features."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from turbofan.config.schema import load_config
from turbofan.data.loader import load_raw_train
from turbofan.experiments.feature_sweep import ExperimentSpec, _evaluate_gru_spec
from turbofan.features.engineering import FeatureSet
from turbofan.models.evaluate import add_rul_column
from turbofan.models.split import split_by_engine


@dataclass(frozen=True)
class TemporalSpec:
    """Specification for one temporal-context sweep run.

    Combines a feature engineering configuration with a specific sequence
    window size so the sweep can cross both axes independently.

    Args:
        feature_set: Feature family to evaluate.
        windows: Rolling window sizes. Non-empty only for rolling feature sets.
        lag_steps: Lag offsets. Always ``()`` in Stage 1.
        sequence_window_size: Number of cycles per sequence window for this run.
        hidden_size: GRU hidden state width.
        learning_rate: Adam optimizer learning rate.
    """

    feature_set: FeatureSet
    windows: tuple[int, ...]
    lag_steps: tuple[int, ...]
    sequence_window_size: int
    hidden_size: int = 64
    learning_rate: float = 0.001


def build_stage1_specs(
    rolling_feature_set: str,
    rolling_windows: list[int],
    sequence_window_sizes: list[int],
) -> list[TemporalSpec]:
    """Build the Stage 1 sweep grid: rolling + raw control × sequence windows.

    For each ``sequence_window_size``:

    * One **raw** spec (no rolling window axis).
    * One spec per ``rolling_window`` in ``rolling_windows``.

    Args:
        rolling_feature_set: Rolling feature family name (e.g. ``"rolling_mean"``).
        rolling_windows: Rolling window sizes to evaluate.
        sequence_window_sizes: GRU sequence window sizes to evaluate.

    Returns:
        List of :class:`TemporalSpec` objects covering the full grid plus raw
        controls.
    """
    specs: list[TemporalSpec] = []
    for seq_win in sequence_window_sizes:
        # Raw control — one per sequence window size
        specs.append(
            TemporalSpec(
                feature_set="raw",
                windows=(),
                lag_steps=(),
                sequence_window_size=seq_win,
            )
        )
        # Rolling feature × rolling window combinations
        validated_rolling_feature_set = cast(FeatureSet, rolling_feature_set)
        for roll_win in rolling_windows:
            specs.append(
                TemporalSpec(
                    feature_set=validated_rolling_feature_set,
                    windows=(roll_win,),
                    lag_steps=(),
                    sequence_window_size=seq_win,
                )
            )
    return specs


def _temporal_to_experiment(spec: TemporalSpec) -> ExperimentSpec:
    """Convert a :class:`TemporalSpec` to an :class:`ExperimentSpec`.

    Args:
        spec: Temporal sweep specification.

    Returns:
        Equivalent :class:`ExperimentSpec` for use with
        :func:`~turbofan.experiments.feature_sweep._evaluate_gru_spec`.
    """
    return ExperimentSpec(
        feature_set=spec.feature_set,
        windows=spec.windows,
        lag_steps=spec.lag_steps,
    )


def run_stage1_sweep(
    config_path: Path,
    rolling_feature_set: str,
    rolling_windows: list[int],
    sequence_window_sizes: list[int],
    device: str = "cpu",
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Train and evaluate the Stage 1 temporal-context GRU sweep.

    Crosses ``sequence_window_sizes`` against rolling-feature candidates plus a
    raw baseline, evaluating each combination on the validation split.

    Args:
        config_path: Path to the project YAML configuration.
        rolling_feature_set: Rolling feature family name (e.g. ``"rolling_mean"``).
        rolling_windows: Rolling window sizes to evaluate.
        sequence_window_sizes: GRU sequence window sizes to evaluate.
        device: Torch device string, either ``"cpu"`` or ``"cuda"``.
        output_path: Optional CSV path; writes sorted results when provided.

    Returns:
        DataFrame of results sorted ascending by ``phm08_score``.
    """
    from turbofan.models.sequence_training import resolve_device

    torch_device = resolve_device(cast(Literal["cpu", "cuda"], device))

    cfg = load_config(config_path)
    train_raw = load_raw_train(cfg.data)
    train_labeled = add_rul_column(train_raw, max_rul=cfg.data.max_rul)
    train_df, val_df = split_by_engine(
        train_labeled,
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
    )

    specs = build_stage1_specs(
        rolling_feature_set=rolling_feature_set,
        rolling_windows=rolling_windows,
        sequence_window_sizes=sequence_window_sizes,
    )

    rows: list[dict[str, object]] = []
    for i, spec in enumerate(specs, 1):
        # Override sequence config with this spec's window size and hyperparams
        overridden_cfg = cfg.model_copy(
            update={
                "sequence": cfg.sequence.model_copy(
                    update={
                        "window_size": spec.sequence_window_size,
                        "hidden_size": spec.hidden_size,
                        "learning_rate": spec.learning_rate,
                    }
                )
            }
        )
        experiment_spec = _temporal_to_experiment(spec)
        row = _evaluate_gru_spec(
            experiment_spec, train_df, val_df, overridden_cfg, torch_device
        )
        # Annotate with temporal sweep dimensions
        row["sequence_window_size"] = spec.sequence_window_size
        row["hidden_size"] = spec.hidden_size
        row["learning_rate"] = spec.learning_rate
        rows.append(row)
        print(
            f"run {i}/{len(specs)}: "
            f"feature_set={spec.feature_set} "
            f"windows={spec.windows} "
            f"sequence_window_size={spec.sequence_window_size} "
            f"phm08_score={row['phm08_score']:.6f}"
        )

    results = pd.DataFrame(rows).sort_values("phm08_score").reset_index(drop=True)
    if output_path is None:
        output_path = Path(
            f"results/stage1_temporal_sweep_{cfg.data.fd_subset.lower()}.csv"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Stage 1 temporal sweep.

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
        "--rolling-feature-set",
        default="rolling_mean",
        help="Rolling feature family to sweep (default: rolling_mean).",
    )
    parser.add_argument(
        "--rolling-windows",
        type=int,
        nargs="+",
        default=[10, 15, 20],
        help="Rolling window sizes to evaluate.",
    )
    parser.add_argument(
        "--sequence-window-sizes",
        type=int,
        nargs="+",
        default=[30, 45, 60],
        help="GRU sequence window sizes to evaluate.",
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
    """Run the Stage 1 GRU temporal-context sweep CLI."""
    args = _parse_args()
    results = run_stage1_sweep(
        config_path=args.config,
        rolling_feature_set=args.rolling_feature_set,
        rolling_windows=args.rolling_windows,
        sequence_window_sizes=args.sequence_window_sizes,
        device=args.device,
        output_path=args.output,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
