"""Training jobs for regenerating official C-MAPSS evaluation results."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from turbofan.config import schema
from turbofan.config.schema import (
    DeviceRequest,
    FDSubset,
    ModelName,
    ProjectConfig,
    SequenceArchitecture,
)
from turbofan.models import baseline, evaluate, metrics
from turbofan.training import sequence_pipeline, sequence_training, split

SUBSETS: tuple[FDSubset, ...] = ("FD001", "FD002", "FD003", "FD004")
SEQUENCE_MODELS: tuple[SequenceArchitecture, ...] = ("gru", "lstm")
RIDGE_SEEDS: tuple[int, ...] = (42,)

# The engine split and feature-pipeline random_state are held at this seed for
# every sequence run, so only model init/training varies across the seed band
# and seed 42 reproduces the registered v1 model exactly.
SPLIT_SEED = 42

# Display order for the model column: baseline first, then sequence models.
MODEL_ORDER: dict[ModelName, int] = {"ridge": 0, "gru": 1, "lstm": 2}


@dataclass(frozen=True)
class RunRecord:
    """Official-eval result for one training run.

    Args:
        model: Model family (``"ridge"``, ``"gru"``, or ``"lstm"``).
        subset: C-MAPSS subset identifier, e.g. ``"FD001"``.
        seed: Model-init/training seed.
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

    model: ModelName
    subset: FDSubset
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


@dataclass(frozen=True)
class Job:
    """One official-eval unit of work.

    Args:
        model: Model family (``"ridge"``, ``"gru"``, or ``"lstm"``).
        subset: C-MAPSS subset identifier.
        config_path: Path to the YAML config for this model/subset pair.
        seed: Model-init/training seed.
    """

    model: ModelName
    subset: FDSubset
    config_path: Path
    seed: int


def build_jobs(
    *,
    configs_dir: Path,
    models: Sequence[ModelName],
    sequence_seeds: Sequence[int],
) -> list[Job]:
    """Enumerate the train/eval jobs for the requested models and subsets.

    Ridge jobs use the per-subset baseline config at a single seed; sequence
    jobs use the per-architecture config across ``sequence_seeds``.

    Args:
        configs_dir: Directory holding the per-subset YAML configs.
        models: Model families to include.
        sequence_seeds: Seeds to sweep for sequence models.

    Returns:
        Jobs ordered by model, subset, then seed.

    Raises:
        FileNotFoundError: If an expected config file is missing.
    """
    jobs: list[Job] = []
    for model in sorted(models, key=lambda name: MODEL_ORDER.get(name, 99)):
        for subset in SUBSETS:
            config_path = config_path_for(configs_dir, model, subset)
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found: {config_path}")
            seeds = sequence_seeds if model in SEQUENCE_MODELS else RIDGE_SEEDS
            for seed in seeds:
                jobs.append(
                    Job(
                        model=model,
                        subset=subset,
                        config_path=config_path,
                        seed=seed,
                    )
                )
    return jobs


def run_job(job: Job, *, device: DeviceRequest) -> RunRecord:
    """Train and officially evaluate one job.

    Args:
        job: The model/subset/config/seed unit of work.
        device: Requested compute device (``"cpu"``, ``"cuda"``, or ``"auto"``).

    Returns:
        The populated run record.
    """
    cfg = schema.load_config(job.config_path)
    if job.model == "ridge":
        return _evaluate_ridge(cfg, job)
    return _evaluate_sequence(cfg, job, device=device)


def join_windows(
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


def join_lag_steps(
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


def config_path_for(configs_dir: Path, model: ModelName, subset: FDSubset) -> Path:
    """Return the config path for a model/subset pair.

    Args:
        configs_dir: Directory holding the per-subset YAML configs.
        model: Model family (``"ridge"``, ``"gru"``, or ``"lstm"``).
        subset: C-MAPSS subset identifier.

    Returns:
        Path to ``fd00X.yaml`` for Ridge or ``fd00X_{arch}.yaml`` for a
        sequence model.
    """
    lower = subset.lower()
    if model == "ridge":
        return configs_dir / f"{lower}.yaml"
    return configs_dir / f"{lower}_{model}.yaml"


def job_key(job: Job) -> tuple[str, str, str]:
    """Return the resume identity key for a job.

    Args:
        job: The unit of work.

    Returns:
        Tuple of model, subset, and seed as strings.
    """
    return (job.model, job.subset, str(job.seed))


def _evaluate_ridge(cfg: ProjectConfig, job: Job) -> RunRecord:
    """Train the Ridge baseline and evaluate it on the official test set."""
    max_rul = cfg.data.max_rul
    frames = split.load_and_split(
        cfg.data,
        max_rul=max_rul,
        test_size=cfg.data.test_size,
        split_seed=job.seed,
    )
    x_train, y_train = evaluate.split_features_target(frames.train)
    x_val, y_val = evaluate.split_features_target(frames.val)

    rf = cfg.features.for_model("ridge")
    estimator = baseline.build_ridge_estimator(cfg, seed=job.seed)
    estimator.fit(x_train, y_train)

    val_pred = evaluate.predict_with_clipping(
        estimator, x_val, max_rul=max_rul, label="validation"
    )
    val_metrics = metrics.regression_metrics(y_val, val_pred)
    official = evaluate.predict_ridge_official(
        cfg.data, estimator=estimator, max_rul=max_rul
    )
    official_metrics = metrics.official_test_metrics(official.y_true, official.y_pred)
    return _build_record(
        job,
        feature_families=rf.feature_families,
        windows=rf.windows,
        lag_steps=rf.lag_steps,
        sequence_window="",
        hidden_size="",
        learning_rate="",
        val_metrics=val_metrics,
        official=official_metrics,
    )


def _evaluate_sequence(
    cfg: ProjectConfig, job: Job, *, device: DeviceRequest
) -> RunRecord:
    """Train a sequence model and evaluate it on the official test set."""
    dev = sequence_training.resolve_device(device)
    max_rul = cfg.data.max_rul
    sf = cfg.features.for_model(cfg.sequence.architecture)

    prepared = sequence_pipeline.prepare_sequence_data(
        cfg.data,
        feature_families=sf.feature_families,
        windows=sf.windows,
        lag_steps=sf.lag_steps,
        sensor_drop=cfg.features.sensor_cols_to_drop or None,
        n_modes=cfg.features.n_modes,
        data_seed=SPLIT_SEED,
        max_rul=max_rul,
        test_size=cfg.data.test_size,
        window_size=cfg.sequence.window_size,
        batch_size=cfg.sequence.batch_size,
    )
    result = sequence_pipeline.train_prepared_sequence(
        prepared,
        cfg.sequence,
        device=dev,
        model_seed=job.seed,
        max_rul=max_rul,
    )

    val_metrics, _, _ = sequence_pipeline.evaluate_window_metrics(
        result.model,
        prepared.val_loader,
        prepared.val_windows,
        device=dev,
        max_rul=max_rul,
    )
    official = sequence_pipeline.predict_sequence_official(
        cfg.data,
        pipeline=prepared.pipeline,
        feature_cols=prepared.feature_cols,
        model=result.model,
        device=dev,
        window_size=cfg.sequence.window_size,
        batch_size=cfg.sequence.batch_size,
        max_rul=max_rul,
    )
    official_metrics = metrics.official_test_metrics(official.y_true, official.y_pred)
    return _build_record(
        job,
        feature_families=sf.feature_families,
        windows=sf.windows,
        lag_steps=sf.lag_steps,
        sequence_window=str(cfg.sequence.window_size),
        hidden_size=str(cfg.sequence.hidden_size),
        learning_rate=str(cfg.sequence.learning_rate),
        val_metrics=val_metrics,
        official=official_metrics,
    )


def _build_record(
    job: Job,
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
    """Assemble a run record from a job, config metadata, and metrics."""
    families = list(feature_families)
    return RunRecord(
        model=job.model,
        subset=job.subset,
        seed=job.seed,
        feature_config="+".join(families),
        rolling_window=join_windows(families, windows),
        lag_step=join_lag_steps(families, lag_steps),
        sequence_window=sequence_window,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        val_rmse=val_metrics["rmse"],
        val_mae=val_metrics["mae"],
        official_rmse=official["rmse"],
        official_mae=official["mae"],
        official_phm08=official["phm08_score"],
    )
