# MLflow Experiment Tracking — Design

**Date:** 2026-06-01
**Stage:** MLOps Infrastructure (step 2 of 4: experiment tracking)
**Status:** Approved design, pending implementation plan

## Goal

Unify Ridge + GRU run logging under MLflow, replacing the GRU-only
`results/training_log.jsonl` audit log. MLflow becomes the single run-logging
system for both production training and hyperparameter sweeps. Local file store
only — no tracking server. On-disk artifacts (`model.*`, `metrics.json`,
`config.json`, `model_manifest.json`, prediction CSVs, sweep result CSVs) remain
the source of truth and are unchanged.

## Decisions (settled during brainstorming)

| Decision | Choice |
|---|---|
| Scope | Local file store at `./mlruns`; log params + metrics (+ curves); disk stays artifact source of truth |
| Fate of `training_log.jsonl` | Replace entirely — stop writing it across all paths |
| Code paths affected | All five: 2 production train CLIs + 3 experiment sweeps |
| Logging style | MLflow-native (Approach B): log params, per-epoch curves (GRU), and final metrics inline within a run context |
| Sweep run grouping | Nested parent/child — one parent run per sweep, one nested child run per trial |
| Experiment organization | Two experiments: `turbofan-training` (production) and `turbofan-sweeps` |
| Old JSONL file | Frozen on disk (235 historical rows kept); writer code removed; file no longer written |
| `mlflow` dependency | Core `dependencies` in `pyproject.toml` (train CLIs import the tracking module) |

### Realization note on "curves"

GRU training already returns the full `result.history` DataFrame *after* training
completes. Curves are produced by **replaying that history into MLflow as stepped
metrics** post-hoc — the `train_gru_model` loop is **not** modified. This keeps
Approach B low-churn.

## Architecture

### New module: `src/turbofan/tracking.py`

A thin, testable wrapper over MLflow containing no training logic. Public API
(all fully type-annotated, Google-style docstrings):

- `TRAINING_EXPERIMENT: str = "turbofan-training"`
- `SWEEP_EXPERIMENT: str = "turbofan-sweeps"`
- `configure_mlflow(tracking_uri: str | None = None) -> None`
  Point MLflow at the local file store (default `./mlruns`; honors the
  `MLFLOW_TRACKING_URI` env var when set). Idempotent.
- `log_params(params: Mapping[str, object]) -> None`
  Stringify and log a flat param mapping.
- `log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None`
  Log final or stepped metrics. Validates that `rmse` and `mae` are present
  (validation logic migrated from the removed `build_log_entry`).
- `set_tags(tags: Mapping[str, object]) -> None`
  Stringify and set run tags.
- `log_history(history: pd.DataFrame) -> None`
  Replay per-epoch `train_loss`/`val_loss` as stepped MLflow metrics.

### Metric / tag naming conventions

- Validation metrics: `val_rmse`, `val_mae`.
- Official-test metrics (when files present): `official_rmse`, `official_mae`,
  `official_phm08`.
- Per-epoch curve metrics: `train_loss`, `val_loss` (with `step=epoch`).
- Tags: `model_type` (`ridge`/`gru`), `run_type` (`production`/`sweep`),
  `run_dir` (production only), `best_epoch` (GRU), plus sweep feature tags
  (`feature_set`, `windows`, `lag_steps`) where applicable.

### Per-path integration

Each call site wraps its existing eval/save section in
`with mlflow.start_run(...)`. No control-flow merging between CLIs and sweeps —
they already share the `build_log_entry`/`append_training_log` seam, which is the
only thing being swapped.

| Path | Experiment | Run shape | Logs |
|---|---|---|---|
| `cli/train_baseline.py` (Ridge) | training | single | params (alpha, feature_set, windows, lag_steps, seed), `val_*` + `official_*`, tags (`model_type=ridge`, `run_type=production`, `run_dir`). No curve. |
| `cli/train_sequence_gru.py` | training | single | params (sequence hyperparams + feature cfg + seed), curve from `result.history`, `val_*` + `official_*`, tags (`model_type=gru`, `best_epoch`, `run_dir`). |
| `experiments/sequence_gru_sweep.py` | sweeps | parent + nested child per trial | per-trial params, curve, `val_*`, sweep tags |
| `experiments/feature_sweep.py` (Ridge) | sweeps | parent + nested child per trial | per-trial params, `val_*`, feature tags. No curve. |
| `experiments/feature_gru_sweep.py` | sweeps | parent + nested child per trial | per-trial params, curve, `val_*`, feature tags |

### Removal of the JSONL system

- Delete `src/turbofan/models/training_log.py`. The `rmse`/`mae` validation it
  performed moves into `tracking.log_metrics`.
- Update the five import sites / call sites to use `tracking` instead of
  `append_training_log` + `build_log_entry`.
- `results/training_log.jsonl`: **left frozen on disk** (not `git rm`-ed). No
  code writes to it after this change.

### Dependency / config / gitignore

- Add `mlflow` to core `dependencies` in `pyproject.toml`. CI already installs
  core deps via `pip install -e ".[dev]"`, so the CI contract picks it up.
- Add `mlruns/` to `.gitignore` (local run store, not committed).
- mypy: add an override for `mlflow.*` (`ignore_missing_imports = true`) if it
  ships no stubs, matching the existing pattern for `sklearn.*`/`scipy.*`.

## Testing

- Replace `tests/models/test_training_log.py` with `tests/test_tracking.py`:
  point the tracking URI at `tmp_path`, exercise each helper, and assert logged
  params/metrics/tags and nested-run structure via `mlflow.search_runs`.
- Update the CLI/sweep tests that asserted `append_training_log` behavior
  (`test_train_sequence_gru_cli.py`, `test_feature_sweep.py`,
  `test_sweep_feature_gru.py`, `test_sweep_sequence_gru.py`, `conftest.py`) to
  point MLflow at `tmp_path` and assert runs were created instead.
- All tests must remain data-independent (synthetic fixtures only); CI needs no
  C-MAPSS download.

## Documentation

- Update forward-looking docs (README, `docs/roadmap.md`) that describe the
  run-logging mechanism to reference MLflow.
- **Leave historical sweep reports unchanged** — they cite past
  `training_log.jsonl`-era data; per project convention, historical numbers are
  not rewritten.

## Out of scope (deferred)

- Live in-loop epoch logging (instrumenting `train_gru_model`) — post-hoc history
  replay is sufficient for v1.
- Logging artifacts into MLflow (models/predictions) — disk remains the source of
  truth.
- Tracking server / remote backend — local file store only.
- Steps 3 (structured logging) and 4 (model registry) of the MLOps stage.

## Conventions honored

Google-style docstrings; full type annotations; `mypy --strict` clean; ruff
`E,F,W,I,UP,ANN` at line length 88; `--no-ff` merge; reports cite local data only.
