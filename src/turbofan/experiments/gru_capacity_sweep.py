"""Stage 2 GRU capacity sweep: hidden_size × learning_rate on top-K Stage 1 configs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
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
class CapacitySpec:
    """Specification for one Stage 2 capacity sweep run.

    Combines a Stage 1 winning configuration with a specific ``hidden_size``
    and ``learning_rate`` so the sweep can explore model capacity independently
    of the feature engineering choices fixed in Stage 1.

    Args:
        feature_set: Feature family to evaluate.
        windows: Rolling window sizes. Non-empty only for rolling feature sets.
        sequence_window_size: Number of cycles per sequence window (fixed from
            Stage 1).
        hidden_size: GRU hidden state width to evaluate in this run.
        learning_rate: Adam optimizer learning rate to evaluate in this run.
    """

    feature_set: FeatureSet
    windows: tuple[int, ...]
    sequence_window_size: int
    hidden_size: int
    learning_rate: float


def _parse_tuple_cell(cell: str) -> tuple[int, ...]:
    """Parse a stringified Python tuple from a CSV cell.

    Handles the representations produced by ``str(tuple(...))`` when persisting
    sweep results to CSV, e.g. ``"(15,)"`` → ``(15,)`` and ``"()"`` → ``()``.

    Args:
        cell: String representation of a Python tuple of integers.

    Returns:
        Parsed tuple of integers.

    Raises:
        ValueError: If ``cell`` cannot be interpreted as a tuple of integers.
    """
    stripped = cell.strip()
    # Remove outer parentheses
    if not (stripped.startswith("(") and stripped.endswith(")")):
        raise ValueError(f"Expected a tuple string, got: {cell!r}")
    inner = stripped[1:-1].strip()
    if not inner:
        return ()
    parts = [p.strip().rstrip(",") for p in inner.split(",") if p.strip().rstrip(",")]
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"Non-integer element in tuple string {cell!r}") from exc


def select_top_k_from_stage1(
    csv_path: Path,
    k: int,
    primary: str = "rmse",
    tiebreak: str = "mae",
) -> list[dict[str, object]]:
    """Select the top-K configurations from a Stage 1 sweep CSV.

    Rows are sorted ascending by ``primary`` (lower is better) then by
    ``tiebreak`` (also ascending), and the first ``k`` rows are returned as
    plain dictionaries suitable for passing to :func:`build_stage2_specs`.

    Args:
        csv_path: Path to the Stage 1 sweep results CSV.
        k: Number of top configurations to return.
        primary: Column name to sort by first (ascending). Defaults to
            ``"rmse"`` (the validation ranking metric).
        tiebreak: Column name to sort by second (ascending) when ``primary``
            values are equal. Defaults to ``"mae"``.

    Returns:
        List of at most ``k`` row dicts, sorted best-first.
    """
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values([primary, tiebreak]).reset_index(drop=True)
    raw: list[dict[str, object]] = cast(
        list[dict[str, object]], df_sorted.head(k).to_dict(orient="records")
    )
    return raw


def build_stage2_specs(
    bases: list[dict[str, object]],
    hidden_sizes: list[int],
    learning_rates: list[float],
) -> list[CapacitySpec]:
    """Build the Stage 2 sweep grid from top-K Stage 1 bases × capacity axes.

    For every combination of ``(base, hidden_size, learning_rate)`` one
    :class:`CapacitySpec` is produced, resulting in
    ``len(bases) × len(hidden_sizes) × len(learning_rates)`` specs.

    Args:
        bases: List of Stage 1 row dicts (as returned by
            :func:`select_top_k_from_stage1`).  Each dict must contain
            ``"feature_set"``, ``"windows"`` (stringified tuple), and
            ``"sequence_window_size"``.
        hidden_sizes: GRU hidden state widths to sweep.
        learning_rates: Adam optimizer learning rates to sweep.

    Returns:
        List of :class:`CapacitySpec` objects covering the full cross-product.
    """
    specs: list[CapacitySpec] = []
    for base, hidden_size, lr in product(bases, hidden_sizes, learning_rates):
        windows_raw = base["windows"]
        if isinstance(windows_raw, str):
            windows: tuple[int, ...] = _parse_tuple_cell(windows_raw)
        elif isinstance(windows_raw, (list, tuple)):
            windows = tuple(int(w) for w in windows_raw)
        elif isinstance(windows_raw, float) and windows_raw != windows_raw:
            # NaN produced by pandas when reading an empty-string cell (raw
            # feature set has no rolling windows).
            windows = ()
        elif isinstance(windows_raw, (int, float)):
            # Bare numeric value – a single window stored without parens.
            windows = (int(windows_raw),)
        else:
            raise TypeError(f"Unexpected type for 'windows': {type(windows_raw)}")
        seq_win = base["sequence_window_size"]
        if not isinstance(seq_win, (int, float)):
            raise TypeError(
                f"Unexpected type for 'sequence_window_size': {type(seq_win)}"
            )
        specs.append(
            CapacitySpec(
                feature_set=cast(FeatureSet, str(base["feature_set"])),
                windows=windows,
                sequence_window_size=int(seq_win),
                hidden_size=hidden_size,
                learning_rate=lr,
            )
        )
    return specs


def _capacity_to_experiment(spec: CapacitySpec) -> ExperimentSpec:
    """Convert a :class:`CapacitySpec` to an :class:`ExperimentSpec`.

    Args:
        spec: Capacity sweep specification.

    Returns:
        Equivalent :class:`ExperimentSpec` for use with
        :func:`~turbofan.experiments.feature_sweep._evaluate_gru_spec`.
    """
    return ExperimentSpec(
        feature_set=spec.feature_set,
        windows=spec.windows,
        lag_steps=(),
    )


def run_stage2_sweep(
    config_path: Path,
    stage1_csv: Path,
    top_k: int,
    hidden_sizes: list[int],
    learning_rates: list[float],
    device: str = "cpu",
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Train and evaluate the Stage 2 GRU capacity sweep.

    Reads the top-K configurations from a Stage 1 CSV, then crosses each with
    every combination of ``hidden_sizes`` × ``learning_rates``, training and
    evaluating a GRU model for each.

    Args:
        config_path: Path to the project YAML configuration.
        stage1_csv: Path to the Stage 1 sweep results CSV produced by
            :func:`~turbofan.experiments.gru_temporal_sweep.run_stage1_sweep`.
        top_k: Number of Stage 1 configurations to promote to Stage 2.
        hidden_sizes: GRU hidden state widths to sweep.
        learning_rates: Adam optimizer learning rates to sweep.
        device: Torch device string, either ``"cpu"`` or ``"cuda"``.
        output_path: Optional CSV path; writes sorted results when provided.
            Defaults to ``results/stage2_capacity_sweep_<fd_subset>.csv``.

    Returns:
        DataFrame of results sorted ascending by ``rmse``.
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

    bases = select_top_k_from_stage1(stage1_csv, k=top_k)
    specs = build_stage2_specs(
        bases, hidden_sizes=hidden_sizes, learning_rates=learning_rates
    )

    rows: list[dict[str, object]] = []
    for i, spec in enumerate(specs, 1):
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
        experiment_spec = _capacity_to_experiment(spec)
        row = _evaluate_gru_spec(
            experiment_spec, train_df, val_df, overridden_cfg, torch_device
        )
        # Annotate with capacity sweep dimensions
        row["sequence_window_size"] = spec.sequence_window_size
        row["hidden_size"] = spec.hidden_size
        row["learning_rate"] = spec.learning_rate
        rows.append(row)
        print(
            f"run {i}/{len(specs)}: "
            f"feature_set={spec.feature_set} "
            f"windows={spec.windows} "
            f"hidden_size={spec.hidden_size} "
            f"learning_rate={spec.learning_rate} "
            f"rmse={row['rmse']:.6f}"
        )

    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    if output_path is None:
        output_path = Path(
            f"results/stage2_capacity_sweep_{cfg.data.fd_subset.lower()}.csv"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Stage 2 capacity sweep.

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
        "--stage1-csv",
        type=Path,
        required=True,
        help="Path to Stage 1 sweep results CSV.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of Stage 1 configs to promote to Stage 2 (default: 3).",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="GRU hidden state widths to sweep.",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.001, 0.0003, 0.0001],
        help="Adam optimizer learning rates to sweep.",
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
    """Run the Stage 2 GRU capacity sweep CLI."""
    args = _parse_args()
    results = run_stage2_sweep(
        config_path=args.config,
        stage1_csv=args.stage1_csv,
        top_k=args.top_k,
        hidden_sizes=args.hidden_sizes,
        learning_rates=args.learning_rates,
        device=args.device,
        output_path=args.output,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
