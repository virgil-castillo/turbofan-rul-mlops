"""Seed-noise band driver for the feature-family screen.

The primary 240-run screen
(:mod:`turbofan.experiments.feature_family_screen`) used a single seed (42), so
it cannot quantify run-to-run variance.  This driver re-runs only the eight
winning cells (one per ``architecture`` x ``subset``, ranked by validation
RMSE) across several seeds and appends every result to a single resumable CSV,
``results/feature_family_seed_band.csv``.  The within-cell standard deviation of
``val_rmse`` across seeds is the empirical seed-noise band used to qualify the
screen's conclusions (e.g. whether the GRU-vs-LSTM gap exceeds the noise).

It reuses :func:`turbofan.experiments.feature_family_screen.run_cell` verbatim,
so every fixed hyperparameter and the split/normaliser seed (always 42) match
the primary screen exactly; only model initialisation and training consume the
swept seed.

Usage:
    python jobs/slurm/seed_band.py --seeds 42 43 44 45 46 --device auto
"""
from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from pathlib import Path

from turbofan.config.schema import FeatureFamilyName
from turbofan.experiments.feature_family_screen import (
    ScreenCell,
    append_row,
    run_cell,
)
from turbofan.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Best (architecture, subset) winner from the primary screen, ranked by
# validation RMSE. Tuple layout matches the ScreenCell fields that vary here:
# (architecture, subset, feature_config, feature_families, rolling_window,
#  lag_step, sequence_window).
BEST_CELLS: list[
    tuple[str, str, str, list[FeatureFamilyName], int | None, int | None, int]
] = [
    ("gru", "FD001", "raw+rolling_slope", ["raw", "rolling_slope"], 20, None, 60),
    ("gru", "FD002", "raw+rolling_slope", ["raw", "rolling_slope"], 20, None, 60),
    ("gru", "FD003", "raw", ["raw"], None, None, 60),
    ("gru", "FD004", "raw+rolling_mean", ["raw", "rolling_mean"], 20, None, 30),
    ("lstm", "FD001", "raw+rolling_slope", ["raw", "rolling_slope"], 20, None, 60),
    ("lstm", "FD002", "raw+rolling_slope", ["raw", "rolling_slope"], 20, None, 30),
    ("lstm", "FD003", "raw+rolling_delta", ["raw", "rolling_delta"], 20, None, 60),
    ("lstm", "FD004", "raw+rolling_mean", ["raw", "rolling_mean"], 20, None, 30),
]

# Resume identity: architecture + subset + the five ScreenCell discriminators +
# seed, all as the strings they round-trip to in the CSV.
BandKey = tuple[str, str, str, str, str, str, str]


def _band_key(
    architecture: str,
    subset: str,
    feature_config: str,
    rolling_window: int | None,
    lag_step: int | None,
    sequence_window: int,
    seed: int,
) -> BandKey:
    """Return the resume key for one band cell.

    Unlike :func:`turbofan.experiments.feature_family_screen.cell_key`, this key
    includes ``architecture`` and ``subset`` because the band writes every cell
    to one combined CSV rather than per-(architecture, subset) files.

    Args:
        architecture: Sequence model architecture (``"gru"`` or ``"lstm"``).
        subset: C-MAPSS subset identifier (e.g. ``"FD001"``).
        feature_config: Stable feature-config label.
        rolling_window: Rolling window size, or ``None`` when inapplicable.
        lag_step: Lag step, or ``None`` when inapplicable.
        sequence_window: Sequence window size in cycles.
        seed: Model init / training seed.

    Returns:
        A 7-tuple of strings; ``None`` factors render as ``""``.
    """
    rw = "" if rolling_window is None else str(rolling_window)
    ls = "" if lag_step is None else str(lag_step)
    return (
        architecture,
        subset,
        feature_config,
        rw,
        ls,
        str(sequence_window),
        str(seed),
    )


def _completed(path: Path) -> set[BandKey]:
    """Read the band CSV and return the set of already-completed cell keys.

    Args:
        path: Path to the band CSV. A missing file yields an empty set.

    Returns:
        Set of :data:`BandKey` tuples for every fully-written row.
    """
    if not path.exists():
        return set()
    done: set[BandKey] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            values = (
                row.get("architecture"),
                row.get("subset"),
                row.get("feature_config"),
                row.get("rolling_window"),
                row.get("lag_step"),
                row.get("sequence_window"),
                row.get("seed"),
            )
            if None in values:  # short / half-written trailing row
                continue
            done.add(tuple(str(v) for v in values))  # type: ignore[arg-type]
    return done


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the seed-band driver.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Seeds to replicate each winning cell across (seed 42 reproduces "
        "the primary-screen value).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for the band CSV.",
    )
    parser.add_argument(
        "--out-name",
        default="feature_family_seed_band.csv",
        help="Band CSV file name within --results-dir.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs/subsets"),
        help="Directory containing per-subset YAML configs.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=os.environ.get("DEVICE", "auto"),
        help="Compute device. 'auto' picks CUDA when present, else CPU.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging verbosity.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the seed-band replication with CSV resume.

    Args:
        argv: Optional CLI arguments; uses ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code (0 on success).
    """
    args = _build_parser().parse_args(argv)
    setup_logging(args.log_level)

    out_path: Path = args.results_dir / args.out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(out_path)

    plan = [(cell, seed) for cell in BEST_CELLS for seed in args.seeds]
    total = len(plan)
    logger.info(
        "seed band: %d winning cells x %d seeds = %d runs -> %s (device=%s)",
        len(BEST_CELLS),
        len(args.seeds),
        total,
        out_path,
        args.device,
    )

    for idx, (spec, seed) in enumerate(plan, start=1):
        arch, subset, fcfg, families, rw, ls, sw = spec
        key = _band_key(arch, subset, fcfg, rw, ls, sw, seed)
        desc = f"{arch} {subset} {fcfg} rw={rw} lag={ls} sw={sw} seed={seed}"
        if key in done:
            logger.info("skipping completed %d/%d: %s", idx, total, desc)
            continue

        logger.info("running %d/%d: %s", idx, total, desc)
        cell = ScreenCell(
            architecture=arch,
            subset=subset,
            feature_config=fcfg,
            feature_families=families,
            rolling_window=rw,
            lag_step=ls,
            sequence_window=sw,
            seed=seed,
        )
        row = run_cell(cell, configs_dir=args.configs_dir, device=args.device)
        append_row(out_path, row)
        done.add(key)
        logger.info(
            "completed %d/%d: %s -> val_rmse=%.4f val_mae=%.4f (%.1fs)",
            idx,
            total,
            desc,
            row["val_rmse"],
            row["val_mae"],
            row["training_duration_seconds"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
