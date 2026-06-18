"""CLI entry point for the feature-family sweep (``turbofan-feature-screen``).

Parses grid-sweep parameters and delegates to
:func:`turbofan.experiments.feature_family_screen.run_screen`.  All arguments
have spec-mandated defaults so the command can be invoked with no flags for a
full 256-run primary screen, or with ``--seeds 43 44 45 46`` (plus a resume
skip-set) for the noise-band runs.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from turbofan.experiments.feature_family_screen import run_screen
from turbofan.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the feature-family screen CLI.

    Returns:
        Configured argument parser with all sweep parameters and their defaults.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=["FD001", "FD002", "FD003", "FD004"],
        default=["FD001", "FD002", "FD003", "FD004"],
        help="C-MAPSS subsets to include in the sweep.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=["gru", "lstm"],
        default=["gru", "lstm"],
        help="Sequence model architectures to sweep.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for model init and training.",
    )
    parser.add_argument(
        "--rolling-windows",
        type=int,
        nargs="+",
        default=[5, 20],
        help="Rolling window sizes for rolling-* feature configs.",
    )
    parser.add_argument(
        "--lag-steps",
        type=int,
        nargs="+",
        default=[1, 5],
        help="Lag step sizes for the lag feature config.",
    )
    parser.add_argument(
        "--sequence-windows",
        type=int,
        nargs="+",
        default=[30, 60],
        help="Sequence window sizes (cycles per window).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/results"),
        help="Root directory for generated result CSV files.",
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
        help=(
            "Compute device. 'auto' (default) picks CUDA when available and "
            "falls back to CPU, so the same job runs on GPU and CPU nodes; "
            "'cuda' fails if no GPU is present. Falls back to the DEVICE env var."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging verbosity (falls back to the LOG_LEVEL env var or INFO).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the feature-family sweep with the given parameters.

    Args:
        argv: Optional command-line arguments. Uses ``sys.argv[1:]`` when
            ``None``.

    Returns:
        Process exit code (always 0 on success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    logger.info(
        "starting feature-family screen: architectures=%s subsets=%s seeds=%s "
        "device=%s",
        args.architectures,
        args.subsets,
        args.seeds,
        args.device,
    )
    run_screen(
        architectures=args.architectures,
        subsets=args.subsets,
        sequence_windows=args.sequence_windows,
        rolling_windows=args.rolling_windows,
        lag_steps=args.lag_steps,
        seeds=args.seeds,
        results_dir=args.results_dir,
        configs_dir=args.configs_dir,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
