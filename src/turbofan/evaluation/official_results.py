"""CSV persistence and summary aggregation for official evaluation results."""
from __future__ import annotations

import csv
import statistics
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

import pandas as pd

from turbofan.evaluation.official_jobs import MODEL_ORDER, RunRecord

PER_RUN_COLUMNS: list[str] = [field.name for field in fields(RunRecord)]
SUMMARY_COLUMNS: list[str] = [
    "model",
    "subset",
    "n_runs",
    "feature_config",
    "rolling_window",
    "lag_step",
    "sequence_window",
    "val_rmse_mean",
    "val_rmse_sd",
    "official_rmse_mean",
    "official_rmse_sd",
    "official_phm08_mean",
    "official_phm08_sd",
]


def build_summary_frame(records: Sequence[RunRecord]) -> pd.DataFrame:
    """Aggregate per-run records into the per-model/subset summary frame.

    Standard-deviation columns use the sample standard deviation and are blank
    for single-run groups.

    Args:
        records: Per-run official-eval records.

    Returns:
        DataFrame ordered to match :data:`SUMMARY_COLUMNS`.
    """
    groups: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        groups.setdefault((record.model, record.subset), []).append(record)

    rows: list[dict[str, object]] = []
    for (model, subset), group in groups.items():
        head = group[0]
        rows.append(
            {
                "model": model,
                "subset": subset,
                "n_runs": len(group),
                "feature_config": head.feature_config,
                "rolling_window": head.rolling_window,
                "lag_step": head.lag_step,
                "sequence_window": head.sequence_window,
                "val_rmse_mean": repr_float(
                    statistics.mean(r.val_rmse for r in group)
                ),
                "val_rmse_sd": sample_sd([r.val_rmse for r in group]),
                "official_rmse_mean": repr_float(
                    statistics.mean(r.official_rmse for r in group)
                ),
                "official_rmse_sd": sample_sd([r.official_rmse for r in group]),
                "official_phm08_mean": repr_float(
                    statistics.mean(r.official_phm08 for r in group)
                ),
                "official_phm08_sd": sample_sd([r.official_phm08 for r in group]),
            }
        )
    rows.sort(key=group_sort_key)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def sample_sd(values: list[float]) -> str:
    """Compute the sample standard deviation, blank for single-value groups.

    Args:
        values: Metric values for one aggregation group.

    Returns:
        The sample standard deviation as a full-precision string, or an empty
        string when fewer than two values are present.
    """
    if len(values) < 2:
        return ""
    return repr_float(statistics.stdev(values))


def repr_float(value: float) -> str:
    """Render a float as its shortest exactly-round-tripping decimal string.

    Args:
        value: Metric value to render.

    Returns:
        The ``repr`` of the float.
    """
    return repr(float(value))


def append_record(path: Path, record: RunRecord) -> None:
    """Append one record to the per-run CSV, writing the header if needed.

    Args:
        path: Per-run CSV path.
        record: The record to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_RUN_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(record_to_row(record))
        handle.flush()


def record_to_row(record: RunRecord) -> dict[str, str]:
    """Render a record as a CSV row mapping with full-precision metrics.

    Args:
        record: The record to render.

    Returns:
        Mapping from column name to string cell value.
    """
    return {
        "model": record.model,
        "subset": record.subset,
        "seed": str(record.seed),
        "feature_config": record.feature_config,
        "rolling_window": record.rolling_window,
        "lag_step": record.lag_step,
        "sequence_window": record.sequence_window,
        "hidden_size": record.hidden_size,
        "learning_rate": record.learning_rate,
        "val_rmse": repr_float(record.val_rmse),
        "val_mae": repr_float(record.val_mae),
        "official_rmse": repr_float(record.official_rmse),
        "official_mae": repr_float(record.official_mae),
        "official_phm08": repr_float(record.official_phm08),
    }


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    """Read the per-run CSV and return completed model/subset/seed keys.

    Args:
        path: Per-run CSV path. A missing file yields an empty set.

    Returns:
        Set of resume identity keys for fully-written rows.
    """
    return {record_key(record) for record in read_records(path)}


def read_records(path: Path) -> list[RunRecord]:
    """Read written per-run records from the CSV, sorted for stable output.

    Args:
        path: Per-run CSV path. A missing file yields an empty list.

    Returns:
        Records ordered by model, subset, then seed.

    Raises:
        ValueError: If a row carries a non-numeric metric or seed.
    """
    if not path.exists():
        return []
    records: list[RunRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if any(row.get(column) in (None, "") for column in ("model", "seed")):
                continue
            records.append(record_from_row(row))
    records.sort(key=record_sort_key)
    return records


def record_from_row(row: dict[str, str]) -> RunRecord:
    """Parse a CSV row mapping back into a run record.

    Args:
        row: CSV row mapping.

    Returns:
        The parsed record.
    """
    return RunRecord(
        model=row["model"],
        subset=row["subset"],
        seed=int(row["seed"]),
        feature_config=row.get("feature_config", ""),
        rolling_window=row.get("rolling_window", "") or "",
        lag_step=row.get("lag_step", "") or "",
        sequence_window=row.get("sequence_window", "") or "",
        hidden_size=row.get("hidden_size", "") or "",
        learning_rate=row.get("learning_rate", "") or "",
        val_rmse=float(row["val_rmse"]),
        val_mae=float(row["val_mae"]),
        official_rmse=float(row["official_rmse"]),
        official_mae=float(row["official_mae"]),
        official_phm08=float(row["official_phm08"]),
    )


def record_key(record: RunRecord) -> tuple[str, str, str]:
    """Return the resume identity key for a written record.

    Args:
        record: A per-run record.

    Returns:
        Tuple of model, subset, and seed as strings.
    """
    return (record.model, record.subset, str(record.seed))


def record_sort_key(record: RunRecord) -> tuple[int, str, int]:
    """Sort key ordering per-run rows by model, subset, then seed.

    Args:
        record: A per-run record.

    Returns:
        Tuple of model order, subset, and seed.
    """
    return (MODEL_ORDER.get(record.model, 99), record.subset, record.seed)


def group_sort_key(row: dict[str, object]) -> tuple[int, str]:
    """Sort key ordering summary rows by model order then subset.

    Args:
        row: Summary row mapping.

    Returns:
        Tuple of model order and subset.
    """
    return (MODEL_ORDER.get(str(row["model"]), 99), str(row["subset"]))
