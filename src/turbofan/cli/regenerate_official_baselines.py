"""Generate the multi-seed official-evaluation CSVs for selected models.

This is the reproducible replacement for the ad hoc process that produced the
committed benchmark snapshots under ``results/baselines/``. For every production
configuration shipped in ``configs/subsets`` it trains the model and evaluates
it on the official C-MAPSS test set, then writes:

- ``outputs/results/latest_official_eval_per_run.csv``: one row per model,
  subset, seed.
- ``outputs/results/latest_official_eval_summary.csv``: aggregate metrics per
  model and subset.

The per-run CSV is written incrementally and the run is resumable: an existing
row for a model/subset/seed is skipped, so an interrupted sweep can be restarted
without recomputation. The summary is rebuilt from the full per-run CSV at the
end. No MLflow runs are created and the model registry is never touched.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from turbofan.benchmarks import official_jobs, official_results
from turbofan.utils import logging as turbofan_logging

logger = turbofan_logging.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs/results"
_DEFAULT_CONFIGS_DIR = _REPO_ROOT / "configs/subsets"
_PER_RUN_FILENAME = "latest_official_eval_per_run.csv"
_SUMMARY_FILENAME = "latest_official_eval_summary.csv"
_DEFAULT_SEQUENCE_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)


def main(argv: Sequence[str] | None = None) -> int:
    """Train and evaluate the production configs and write the eval CSVs.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    turbofan_logging.setup_logging(args.log_level)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = output_dir / _PER_RUN_FILENAME
    summary_path = output_dir / _SUMMARY_FILENAME

    try:
        jobs = official_jobs.build_jobs(
            configs_dir=args.configs_dir,
            models=tuple(args.models),
            sequence_seeds=tuple(args.seeds),
        )
        done = official_results.completed_keys(per_run_path)
        remaining = [
            job for job in jobs if official_jobs.job_key(job) not in done
        ]
        logger.info(
            "official-eval sweep: %d job(s), %d already done, %d to run "
            "(device=%s) -> %s",
            len(jobs),
            len(jobs) - len(remaining),
            len(remaining),
            args.device,
            per_run_path,
        )
        for index, job in enumerate(remaining, start=1):
            logger.info(
                "running %d/%d: %s %s seed=%d",
                index,
                len(remaining),
                job.model,
                job.subset,
                job.seed,
            )
            started = perf_counter()
            record = official_jobs.run_job(job, device=args.device)
            official_results.append_record(per_run_path, record)
            logger.info(
                "completed %d/%d: %s %s seed=%d -> "
                "official_rmse=%.4f val_rmse=%.4f (%.1fs)",
                index,
                len(remaining),
                job.model,
                job.subset,
                job.seed,
                record.official_rmse,
                record.val_rmse,
                perf_counter() - started,
            )
        records = official_results.read_records(per_run_path)
        summary = official_results.build_summary_frame(records)
        summary.to_csv(summary_path, index=False)
    except Exception as exc:  # noqa: BLE001 - CLI boundary surfaces failures
        logger.error(str(exc))
        logger.debug("Traceback for the error above:", exc_info=True)
        return 1

    logger.info("wrote %d per-run rows to %s", len(records), per_run_path)
    logger.info("wrote %d summary rows to %s", len(summary), summary_path)
    print(f"per-run:  {per_run_path}")
    print(f"summary:  {summary_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the sweep CLI.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=(
            "Directory to write the official-eval CSVs "
            "(defaults to outputs/results/)."
        ),
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=_DEFAULT_CONFIGS_DIR,
        help="Directory containing per-subset YAML configs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("ridge", "gru", "lstm"),
        default=["ridge", "gru", "lstm"],
        help="Model families to evaluate (defaults to all three).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(_DEFAULT_SEQUENCE_SEEDS),
        help="Seeds swept for sequence models (Ridge always uses seed 42).",
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
        help="Logging verbosity (falls back to the LOG_LEVEL env var or INFO).",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
