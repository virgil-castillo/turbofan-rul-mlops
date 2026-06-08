"""Generate the multi-seed official-evaluation CSVs for the selected models.

This is the committed, version-controlled replacement for the untracked process
that previously produced ``results/latest_official_eval_*.csv``. For every
production configuration shipped in ``configs/subsets`` it trains the model and
evaluates it on the official C-MAPSS test set, then writes:

- ``results/latest_official_eval_per_run.csv`` — one row per (model, subset,
  seed).
- ``results/latest_official_eval_summary.csv`` — per (model, subset) aggregate
  (mean and sample standard deviation across seeds).

Seed design (mirrors the sequence feature-family seed band):

- **Ridge** is evaluated at a single seed (42). Ridge regression is
  deterministic; only the engine train/val split and the KMeans operating-mode
  normalizer consume the seed, so a seed band would measure split noise rather
  than model noise. It is reported as a single value (``n_runs=1``, blank sd).
- **GRU/LSTM** are evaluated across five seeds (42-46). The engine split and the
  feature pipeline's ``random_state`` are held at 42 for every seed; only model
  initialization and training consume the swept seed. Seed 42 therefore
  reproduces the registered ``v1`` model exactly, and the across-seed spread is
  the model-initialization noise reported as ``mean ± sd`` in the README.

Evaluation matches the committed training CLIs exactly. Ridge predictions are
clipped to ``[0, max_rul]`` (the piecewise-linear RUL ceiling); sequence
predictions are clipped to be non-negative and rescaled by ``max_rul`` without
an upper cap, exactly as ``train_baseline.py`` and
``train_sequence.py::_evaluate_official_test`` do, so the regenerated numbers
agree with the MLflow ``v1`` metrics at seed 42.

The per-run CSV is written incrementally and the run is resumable: an existing
row for a (model, subset, seed) is skipped, so an interrupted sweep can be
restarted without recomputation. The summary is rebuilt from the full per-run
CSV at the end. No MLflow runs are created and the model registry is never
touched.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch import nn

from turbofan.config.schema import ProjectConfig, load_config
from turbofan.data.loader import load_raw_test, load_raw_train, load_rul_labels
from turbofan.features.pipeline import build_feature_pipeline
from turbofan.models.baseline import build_baseline_pipeline
from turbofan.models.evaluate import (
    add_rul_column,
    align_official_test_labels,
    select_last_cycle_per_engine,
    split_features_target,
)
from turbofan.models.metrics import official_test_metrics, regression_metrics
from turbofan.models.sequence_models import build_sequence_model
from turbofan.models.sequence_training import (
    predict_windows,
    resolve_device,
    seed_everything,
    train_sequence_model,
)
from turbofan.models.split import split_by_engine
from turbofan.models.test_evaluation import align_labels_to_eligible_engines
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.windowing import build_final_windows, build_sliding_windows
from turbofan.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = Path("results")
_DEFAULT_CONFIGS_DIR = Path("configs/subsets")
_PER_RUN_FILENAME = "latest_official_eval_per_run.csv"
_SUMMARY_FILENAME = "latest_official_eval_summary.csv"

_SUBSETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")
_SEQUENCE_MODELS: tuple[str, ...] = ("gru", "lstm")
_RIDGE_SEEDS: tuple[int, ...] = (42,)
_DEFAULT_SEQUENCE_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

#: The engine split and feature-pipeline random_state are held at this seed for
#: every sequence run, so only model init/training varies across the seed band
#: and seed 42 reproduces the registered v1 model exactly.
_SPLIT_SEED = 42

#: Display order for the ``model`` column (baseline first, then sequence models).
_MODEL_ORDER: dict[str, int] = {"ridge": 0, "gru": 1, "lstm": 2}

_ID_COLS: list[str] = ["engine_id", "cycle", "rul"]


@dataclass(frozen=True)
class RunRecord:
    """Official-eval result for one (model, subset, seed) training run.

    Args:
        model: Model family (``"ridge"``, ``"gru"``, or ``"lstm"``).
        subset: C-MAPSS subset identifier, e.g. ``"FD001"``.
        seed: Model-init/training seed (the split seed is fixed at 42).
        feature_config: Feature families joined with ``"+"``.
        rolling_window: Rolling-window size(s) joined with ``"|"``, or empty.
        lag_step: Lag step(s) joined with ``"|"``, or empty.
        sequence_window: Sequence window size for sequence models, else empty.
        hidden_size: Recurrent hidden size for sequence models, else empty.
        learning_rate: Learning rate for sequence models, else empty.
        val_rmse: Validation RMSE.
        val_mae: Validation MAE.
        official_rmse: Official-test RMSE.
        official_mae: Official-test MAE.
        official_phm08: Official-test PHM08 score.
    """

    model: str
    subset: str
    seed: int
    feature_config: str
    rolling_window: str
    lag_step: str
    sequence_window: str
    hidden_size: str
    learning_rate: str
    val_rmse: float
    val_mae: float
    official_rmse: float
    official_mae: float
    official_phm08: float


_PER_RUN_COLUMNS: list[str] = [field.name for field in fields(RunRecord)]
_SUMMARY_COLUMNS: list[str] = [
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


@dataclass(frozen=True)
class _Job:
    """One unit of work: a config to train and evaluate at one seed.

    Args:
        model: Model family (``"ridge"``, ``"gru"``, or ``"lstm"``).
        subset: C-MAPSS subset identifier.
        config_path: Path to the YAML config for this (model, subset).
        seed: Model-init/training seed.
    """

    model: str
    subset: str
    config_path: Path
    seed: int


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
        done = _completed_keys(per_run_path)
        remaining = [job for job in jobs if _job_key(job) not in done]
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
            _append_record(per_run_path, record)
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
        records = _read_records(per_run_path)
        summary = build_summary_frame(records)
        summary.to_csv(summary_path, index=False)
    except Exception as exc:  # noqa: BLE001 - surface any failure as exit 1
        logger.error(str(exc))
        return 1

    logger.info("wrote %d per-run rows to %s", len(records), per_run_path)
    logger.info("wrote %d summary rows to %s", len(summary), summary_path)
    print(f"per-run:  {per_run_path}")
    print(f"summary:  {summary_path}")
    return 0


def build_jobs(
    *,
    configs_dir: Path,
    models: Sequence[str],
    sequence_seeds: Sequence[int],
) -> list[_Job]:
    """Enumerate the train/eval jobs for the requested models and subsets.

    Ridge jobs use the per-subset baseline config at a single seed; sequence
    jobs use the per-architecture config across ``sequence_seeds``.

    Args:
        configs_dir: Directory holding the per-subset YAML configs.
        models: Model families to include (subset of ridge/gru/lstm).
        sequence_seeds: Seeds to sweep for sequence models.

    Returns:
        Jobs ordered by model, subset, then seed.

    Raises:
        FileNotFoundError: If an expected config file is missing.
    """
    jobs: list[_Job] = []
    for model in sorted(models, key=lambda name: _MODEL_ORDER.get(name, 99)):
        for subset in _SUBSETS:
            config_path = _config_path(configs_dir, model, subset)
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found: {config_path}")
            seeds = sequence_seeds if model in _SEQUENCE_MODELS else _RIDGE_SEEDS
            for seed in seeds:
                jobs.append(
                    _Job(
                        model=model,
                        subset=subset,
                        config_path=config_path,
                        seed=seed,
                    )
                )
    return jobs


def run_job(job: _Job, *, device: str) -> RunRecord:
    """Train and officially evaluate one job.

    Args:
        job: The (model, subset, config, seed) unit of work.
        device: Requested compute device (``"cpu"``, ``"cuda"``, or ``"auto"``).

    Returns:
        The populated :class:`RunRecord`.
    """
    cfg = load_config(job.config_path)
    if job.model == "ridge":
        return _evaluate_ridge(cfg, job)
    return _evaluate_sequence(cfg, job, device=device)


def _evaluate_ridge(cfg: ProjectConfig, job: _Job) -> RunRecord:
    """Train the Ridge baseline and evaluate it on the official test set.

    Mirrors ``train_baseline.py``: predictions are clipped to ``[0, max_rul]``
    on both validation and official test.

    Args:
        cfg: Loaded project config for this subset.
        job: The unit of work (carries the seed).

    Returns:
        The populated :class:`RunRecord`.
    """
    max_rul = cfg.data.max_rul
    train_raw = load_raw_train(cfg.data)
    train_labeled = add_rul_column(train_raw, max_rul=max_rul)
    train_df, val_df = split_by_engine(
        train_labeled, test_size=cfg.data.test_size, random_seed=job.seed
    )
    x_train, y_train = split_features_target(train_df)
    x_val, y_val = split_features_target(val_df)

    rf = cfg.features.for_model("ridge")
    estimator = build_baseline_pipeline(
        model_name=cfg.model.name,
        alpha=cfg.model.alpha,
        feature_families=rf.feature_families,
        windows=rf.windows,
        lag_steps=rf.lag_steps,
        sensor_drop=cfg.features.sensor_cols_to_drop or None,
        n_modes=cfg.features.n_modes,
        random_state=job.seed,
    )
    estimator.fit(x_train, y_train)

    val_pred = _clip(estimator.predict(x_val), max_rul)
    val_metrics = regression_metrics(y_val, val_pred)
    official = _evaluate_ridge_official(cfg, estimator, max_rul=max_rul)
    return _build_record(
        job,
        feature_families=rf.feature_families,
        windows=rf.windows,
        lag_steps=rf.lag_steps,
        sequence_window="",
        hidden_size="",
        learning_rate="",
        val_metrics=val_metrics,
        official=official,
    )


def _evaluate_ridge_official(
    cfg: ProjectConfig,
    estimator: Pipeline,
    *,
    max_rul: int,
) -> dict[str, float]:
    """Evaluate a fitted Ridge estimator on the official test set.

    Args:
        cfg: Loaded project config (for data paths and ``max_rul``).
        estimator: Fitted Ridge pipeline.
        max_rul: Maximum-RUL ceiling for clipping predictions.

    Returns:
        Official-test metric dict (``rmse``, ``mae``, ``phm08_score``).
    """
    test_raw = load_raw_test(cfg.data)
    rul_labels = load_rul_labels(cfg.data)
    last_rows = select_last_cycle_per_engine(test_raw)
    y_true = align_official_test_labels(last_rows, rul_labels)
    all_pred = _clip(estimator.predict(test_raw), max_rul)
    pred_rows = test_raw[["engine_id", "cycle"]].copy()
    pred_rows["prediction"] = all_pred
    last_pred = select_last_cycle_per_engine(pred_rows)
    y_pred = last_pred["prediction"].to_numpy(dtype=np.float64)
    return official_test_metrics(y_true, y_pred)


def _evaluate_sequence(cfg: ProjectConfig, job: _Job, *, device: str) -> RunRecord:
    """Train a sequence model and evaluate it on the official test set.

    Mirrors ``train_sequence.py``: the engine split and feature pipeline use the
    fixed split seed (42) while ``job.seed`` governs model init/training, and
    official predictions are clipped to be non-negative without an upper cap.

    Args:
        cfg: Loaded project config for this (subset, architecture).
        job: The unit of work (carries the architecture via cfg and the seed).
        device: Requested compute device.

    Returns:
        The populated :class:`RunRecord`.
    """
    architecture = cfg.sequence.architecture
    dev = resolve_device(device)  # type: ignore[arg-type]
    max_rul = cfg.data.max_rul
    sf = cfg.features.for_model(architecture)

    train_raw = load_raw_train(cfg.data)
    train_labeled = add_rul_column(train_raw, max_rul=max_rul)
    train_df, val_df = split_by_engine(
        train_labeled, test_size=cfg.data.test_size, random_seed=_SPLIT_SEED
    )

    pipeline = build_feature_pipeline(
        sensor_drop=cfg.features.sensor_cols_to_drop or None,
        n_modes=cfg.features.n_modes,
        random_state=_SPLIT_SEED,
        feature_families=sf.feature_families,
        windows=sf.windows,
        lag_steps=sf.lag_steps,
    )
    train_features = pipeline.fit_transform(train_df)
    val_features = pipeline.transform(val_df)
    feature_cols = pipeline.named_steps["feature_engineer"].feature_cols_

    train_windows = build_sliding_windows(
        _join_ids(train_df, train_features),
        feature_cols=feature_cols,
        window_size=cfg.sequence.window_size,
    )
    val_windows = build_sliding_windows(
        _join_ids(val_df, val_features),
        feature_cols=feature_cols,
        window_size=cfg.sequence.window_size,
    )
    train_loader = build_sequence_loader(
        train_windows, batch_size=cfg.sequence.batch_size, shuffle=True
    )
    val_loader = build_sequence_loader(
        val_windows, batch_size=cfg.sequence.batch_size, shuffle=False
    )

    seed_everything(job.seed)
    model = build_sequence_model(
        architecture,
        input_size=len(feature_cols),
        hidden_size=cfg.sequence.hidden_size,
        num_layers=cfg.sequence.num_layers,
        dropout=cfg.sequence.dropout,
    )
    result = train_sequence_model(
        model=model,
        train_loader=train_loader,
        validation_windows_loader=val_loader,
        config=cfg.sequence,
        device=dev,
        random_seed=job.seed,
        max_rul=max_rul,
    )

    val_pred = np.clip(
        predict_windows(result.model, val_loader, dev, max_rul=max_rul), 0.0, None
    )
    val_metrics = regression_metrics(val_windows.y.astype(np.float64), val_pred)
    official = _evaluate_sequence_official(
        cfg,
        result.model,
        pipeline,
        feature_cols,
        dev,
        max_rul=max_rul,
    )
    return _build_record(
        job,
        feature_families=sf.feature_families,
        windows=sf.windows,
        lag_steps=sf.lag_steps,
        sequence_window=str(cfg.sequence.window_size),
        hidden_size=str(cfg.sequence.hidden_size),
        learning_rate=str(cfg.sequence.learning_rate),
        val_metrics=val_metrics,
        official=official,
    )


def _evaluate_sequence_official(
    cfg: ProjectConfig,
    model: nn.Module,
    pipeline: Pipeline,
    feature_cols: list[str],
    device: torch.device,
    *,
    max_rul: int,
) -> dict[str, float]:
    """Evaluate a trained sequence model on the official test set.

    Args:
        cfg: Loaded project config (for data paths and sequence settings).
        model: Trained sequence model.
        pipeline: Fitted feature pipeline used during training.
        feature_cols: Feature columns produced by the pipeline.
        device: Torch device for inference.
        max_rul: Maximum-RUL cap used to rescale raw model output.

    Returns:
        Official-test metric dict (``rmse``, ``mae``, ``phm08_score``).
    """
    test_raw = load_raw_test(cfg.data)
    rul_labels = load_rul_labels(cfg.data)
    id_cols = [c for c in ("engine_id", "cycle") if c in test_raw.columns]
    test_features = pipeline.transform(test_raw)
    test_df = pd.concat(
        [
            test_raw[id_cols].reset_index(drop=True),
            test_features.reset_index(drop=True),
        ],
        axis=1,
    )
    test_windows = build_final_windows(
        test_df,
        feature_cols=feature_cols,
        window_size=cfg.sequence.window_size,
        target_col=None,
    )
    loader = build_sequence_loader(
        test_windows, batch_size=cfg.sequence.batch_size, shuffle=False
    )
    y_pred = np.clip(
        predict_windows(model, loader, device, max_rul=max_rul), 0.0, None
    )
    y_true = align_labels_to_eligible_engines(test_windows.metadata, rul_labels)
    return official_test_metrics(y_true, y_pred)


def _build_record(
    job: _Job,
    *,
    feature_families: Sequence[str],
    windows: Sequence[int] | None,
    lag_steps: Sequence[int] | None,
    sequence_window: str,
    hidden_size: str,
    learning_rate: str,
    val_metrics: dict[str, float],
    official: dict[str, float],
) -> RunRecord:
    """Assemble a :class:`RunRecord` from a job, its config, and its metrics.

    Args:
        job: The unit of work.
        feature_families: Feature families used by the model.
        windows: Rolling-window sizes, or ``None``.
        lag_steps: Lag steps, or ``None``.
        sequence_window: Sequence window column value (empty for Ridge).
        hidden_size: Hidden-size column value (empty for Ridge).
        learning_rate: Learning-rate column value (empty for Ridge).
        val_metrics: Validation metric dict.
        official: Official-test metric dict.

    Returns:
        The populated :class:`RunRecord`.
    """
    families = list(feature_families)
    return RunRecord(
        model=job.model,
        subset=job.subset,
        seed=job.seed,
        feature_config="+".join(families),
        rolling_window=_join_windows(families, windows),
        lag_step=_join_lag_steps(families, lag_steps),
        sequence_window=sequence_window,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        val_rmse=val_metrics["rmse"],
        val_mae=val_metrics["mae"],
        official_rmse=official["rmse"],
        official_mae=official["mae"],
        official_phm08=official["phm08_score"],
    )


def build_summary_frame(records: Sequence[RunRecord]) -> pd.DataFrame:
    """Aggregate per-run records into the per-(model, subset) summary frame.

    Standard-deviation columns use the sample standard deviation and are blank
    for single-run groups.

    Args:
        records: Per-run official-eval records.

    Returns:
        DataFrame ordered to match :data:`_SUMMARY_COLUMNS`.
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
                "val_rmse_mean": _repr_float(
                    statistics.mean(r.val_rmse for r in group)
                ),
                "val_rmse_sd": _sample_sd([r.val_rmse for r in group]),
                "official_rmse_mean": _repr_float(
                    statistics.mean(r.official_rmse for r in group)
                ),
                "official_rmse_sd": _sample_sd([r.official_rmse for r in group]),
                "official_phm08_mean": _repr_float(
                    statistics.mean(r.official_phm08 for r in group)
                ),
                "official_phm08_sd": _sample_sd(
                    [r.official_phm08 for r in group]
                ),
            }
        )
    rows.sort(key=_group_sort_key)
    return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)


def _clip(values: object, max_rul: int) -> npt.NDArray[np.float64]:
    """Clip predictions to ``[0, max_rul]`` as float64.

    Args:
        values: Raw predictions.
        max_rul: Maximum-RUL ceiling.

    Returns:
        Float64 predictions clipped to ``[0, max_rul]``.
    """
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, float(max_rul))


def _join_ids(rows: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Concatenate identifier columns with engineered features.

    Args:
        rows: Source rows carrying ``engine_id``/``cycle``/``rul``.
        features: Engineered feature frame aligned to ``rows``.

    Returns:
        Combined frame with identifiers followed by features.
    """
    return pd.concat(
        [
            rows[_ID_COLS].reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )


def _sample_sd(values: list[float]) -> str:
    """Compute the sample standard deviation, blank for single-value groups.

    Args:
        values: Metric values for one aggregation group.

    Returns:
        The sample (``ddof=1``) standard deviation as a full-precision string,
        or an empty string when fewer than two values are present.
    """
    if len(values) < 2:
        return ""
    return _repr_float(statistics.stdev(values))


def _repr_float(value: float) -> str:
    """Render a float as its shortest exactly-round-tripping decimal string.

    Args:
        value: Metric value to render.

    Returns:
        The ``repr`` of the float.
    """
    return repr(float(value))


def _join_windows(
    feature_families: Sequence[str],
    windows: Sequence[int] | None,
) -> str:
    """Render the rolling-window column from feature families and windows.

    Args:
        feature_families: Feature family identifiers used by the model.
        windows: Rolling-window sizes, or ``None``.

    Returns:
        The window size(s) joined with ``"|"``, or empty when no rolling
        family is present or no windows are configured.
    """
    if not any(family.startswith("rolling_") for family in feature_families):
        return ""
    return "|".join(str(window) for window in (windows or []))


def _join_lag_steps(
    feature_families: Sequence[str],
    lag_steps: Sequence[int] | None,
) -> str:
    """Render the lag-step column from feature families and lag steps.

    Args:
        feature_families: Feature family identifiers used by the model.
        lag_steps: Lag steps, or ``None``.

    Returns:
        The lag step(s) joined with ``"|"``, or empty when the ``lag`` family
        is not used.
    """
    if "lag" not in feature_families:
        return ""
    return "|".join(str(step) for step in (lag_steps or []))


def _config_path(configs_dir: Path, model: str, subset: str) -> Path:
    """Return the config path for a (model, subset).

    Args:
        configs_dir: Directory holding the per-subset YAML configs.
        model: Model family (``"ridge"``, ``"gru"``, or ``"lstm"``).
        subset: C-MAPSS subset identifier.

    Returns:
        Path to ``fd00X.yaml`` (Ridge) or ``fd00X_{arch}.yaml`` (sequence).
    """
    lower = subset.lower()
    if model == "ridge":
        return configs_dir / f"{lower}.yaml"
    return configs_dir / f"{lower}_{model}.yaml"


def _job_key(job: _Job) -> tuple[str, str, str]:
    """Return the resume identity key for a job.

    Args:
        job: The unit of work.

    Returns:
        Tuple of model, subset, and seed (all as strings).
    """
    return (job.model, job.subset, str(job.seed))


def _record_key(record: RunRecord) -> tuple[str, str, str]:
    """Return the resume identity key for a written record.

    Args:
        record: A per-run record.

    Returns:
        Tuple of model, subset, and seed (all as strings).
    """
    return (record.model, record.subset, str(record.seed))


def _append_record(path: Path, record: RunRecord) -> None:
    """Append one record to the per-run CSV, writing the header if needed.

    Args:
        path: Per-run CSV path.
        record: The record to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_PER_RUN_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(_record_to_row(record))
        handle.flush()


def _record_to_row(record: RunRecord) -> dict[str, str]:
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
        "val_rmse": _repr_float(record.val_rmse),
        "val_mae": _repr_float(record.val_mae),
        "official_rmse": _repr_float(record.official_rmse),
        "official_mae": _repr_float(record.official_mae),
        "official_phm08": _repr_float(record.official_phm08),
    }


def _completed_keys(path: Path) -> set[tuple[str, str, str]]:
    """Read the per-run CSV and return the set of completed (model, subset, seed).

    Args:
        path: Per-run CSV path. A missing file yields an empty set.

    Returns:
        Set of resume identity keys for fully-written rows.
    """
    return {_record_key(record) for record in _read_records(path)}


def _read_records(path: Path) -> list[RunRecord]:
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
            records.append(_record_from_row(row))
    records.sort(key=lambda record: _record_sort_key(record))
    return records


def _record_from_row(row: dict[str, str]) -> RunRecord:
    """Parse a CSV row mapping back into a :class:`RunRecord`.

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


def _record_sort_key(record: RunRecord) -> tuple[int, str, int]:
    """Sort key ordering per-run rows by model, subset, then seed.

    Args:
        record: A per-run record.

    Returns:
        Tuple of model order, subset, and seed.
    """
    return (_MODEL_ORDER.get(record.model, 99), record.subset, record.seed)


def _group_sort_key(row: dict[str, object]) -> tuple[int, str]:
    """Sort key ordering summary rows by model order then subset.

    Args:
        row: Summary row mapping.

    Returns:
        Tuple of model order and subset.
    """
    return (_MODEL_ORDER.get(str(row["model"]), 99), str(row["subset"]))


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
