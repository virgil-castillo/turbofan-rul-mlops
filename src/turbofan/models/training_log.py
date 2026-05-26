"""Training log entry construction and JSONL persistence."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

_REQUIRED_METRICS = frozenset({"rmse", "mae", "phm08_score"})


def build_log_entry(
    model_type: str,
    dataset: str,
    random_seed: int,
    hyperparameters: dict[str, object],
    metrics: dict[str, float],
    training_duration_seconds: float,
    device: str,
    run_dir: str | None = None,
    best_epoch: int | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a model-agnostic training log entry.

    Args:
        model_type: Model family or implementation name.
        dataset: Dataset identifier used for the run.
        random_seed: Random seed used for training.
        hyperparameters: Hyperparameters used for the run.
        metrics: Evaluation metrics for the trained model. Must include
            ``rmse``, ``mae``, and ``phm08_score``.
        training_duration_seconds: Training wall-clock duration in seconds.
        device: Device used for training.
        run_dir: Optional directory containing run artifacts.
        best_epoch: Optional epoch selected as best during training.
        extra: Optional additional metadata to attach to the entry.

    Returns:
        Training log entry dictionary.

    Raises:
        ValueError: If any required metric is missing.
    """
    missing_metrics = _REQUIRED_METRICS.difference(metrics)
    if missing_metrics:
        missing = ", ".join(sorted(missing_metrics))
        raise ValueError(f"Missing required metric(s): {missing}")

    return {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "model_type": model_type,
        "dataset": dataset,
        "random_seed": random_seed,
        "run_dir": run_dir,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "training_duration_seconds": training_duration_seconds,
        "device": device,
        "extra": extra if extra is not None else {},
    }


def append_training_log(
    entry: dict[str, object],
    log_path: Path = Path("results/training_log.jsonl"),
) -> None:
    """Append a training log entry to a JSON Lines file.

    Args:
        entry: Training log entry to serialize.
        log_path: Destination JSON Lines log path.

    Returns:
        None.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, default=str))
        file.write("\n")
