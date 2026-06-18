"""Generate the multi-seed official-evaluation CSVs for selected models.

This is the committed, version-controlled replacement for the untracked process
that previously produced ``results/latest_official_eval_*.csv``. For every
production configuration shipped in ``configs/subsets`` it trains the model and
evaluates it on the official C-MAPSS test set, then writes:

- ``results/latest_official_eval_per_run.csv``: one row per model, subset, seed.
- ``results/latest_official_eval_summary.csv``: aggregate metrics per model and
  subset.

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

from turbofan.evaluation.official_jobs import (
    Job as _Job,
)
from turbofan.evaluation.official_jobs import (
    RunRecord,
    build_jobs,
    config_path_for,
    job_key,
    join_lag_steps,
    join_windows,
    run_job,
)
from turbofan.evaluation.official_results import (
    append_record,
    build_summary_frame,
    completed_keys,
    group_sort_key,
    read_records,
    record_from_row,
    record_key,
    record_sort_key,
    record_to_row,
    repr_float,
    sample_sd,
)
from turbofan.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = Path("results")
_DEFAULT_CONFIGS_DIR = Path("configs/subsets")
_PER_RUN_FILENAME = "latest_official_eval_per_run.csv"
_SUMMARY_FILENAME = "latest_official_eval_summary.csv"
_DEFAULT_SEQUENCE_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

# Backward-compatible helper names retained for tests and external scripts that
# imported the old single-file implementation.
_append_record = append_record
_completed_keys = completed_keys
_config_path = config_path_for
_group_sort_key = group_sort_key
_job_key = job_key
_join_lag_steps = join_lag_steps
_join_windows = join_windows
_read_records = read_records
_record_from_row = record_from_row
_record_key = record_key
_record_sort_key = record_sort_key
_record_to_row = record_to_row
_repr_float = repr_float
_sample_sd = sample_sd

__all__ = [
    "RunRecord",
    "_Job",
    "build_jobs",
    "build_summary_frame",
    "main",
    "run_job",
]


def main(argv: Sequence[str] | None = None) -> int:
    """Train and evaluate the production configs and write the eval CSVs.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = output_dir / _PER_RUN_FILENAME
    summary_path = output_dir / _SUMMARY_FILENAME

    try:
        jobs = build_jobs(
            configs_dir=args.configs_dir,
            models=tuple(args.models),
            sequence_seeds=tuple(args.seeds),
        )
        done = completed_keys(per_run_path)
        remaining = [job for job in jobs if job_key(job) not in done]
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
            record = run_job(job, device=args.device)
            append_record(per_run_path, record)
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
        records = read_records(per_run_path)
        summary = build_summary_frame(records)
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
        help="Directory to write the official-eval CSVs (defaults to results/).",
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
