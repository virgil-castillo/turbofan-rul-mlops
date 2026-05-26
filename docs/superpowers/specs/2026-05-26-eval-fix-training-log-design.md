# Evaluation Bug Fix & Training Logger: Design Spec

**Date:** 2026-05-26
**Depends on:** Sub-project 5 (Sequence Models), GRU Sweep Validation Report
**Status:** Draft

---

## Goal

Two changes:

1. Fix the degenerate evaluation bug where the GRU sweep and training loop
   evaluate against all-zero targets (final windows of training engines
   where RUL is always 0 by construction).
2. Add a scalable, model-agnostic training logger that appends one JSONL
   entry per trained model to a git-tracked file.

---

## Part 1: Evaluation Bug Fix

### Problem

`build_final_windows` on training-split engines produces targets that are
all zero because every training engine runs to failure
(`rul = max_cycle - max_cycle = 0`). The sweep evaluates metrics against
these all-zero targets, and the training loop uses them for early stopping.
This makes RMSE measure "distance from zero" rather than prediction quality,
and early stopping actively selects model states that predict zero.

See `docs/gru_sweep_validation_report.md` for the full root-cause analysis.

### Fix

#### `src/turbofan/models/sequence_training.py`

Change the early-stopping criterion from `validation_final_loader` to
`validation_windows_loader`. The function signature already accepts both
loaders.

**Before (line 94-96):**
```python
final_metrics = _evaluate_loader(model, validation_final_loader, device)
...
current_metric = final_metrics["rmse"]
```

**After:**
```python
window_metrics = _evaluate_loader(model, validation_windows_loader, device)
...
current_metric = window_metrics["rmse"]
```

Final-window metrics are still computed and recorded in the training
history for reference (they become meaningful when evaluating against the
official test set where engines are cut off before failure). They just no
longer drive model selection.

The history DataFrame columns remain the same. No signature changes.

#### `scripts/sweep_sequence_gru.py`

Change evaluation from `validation_final_windows.y` to
`validation_windows.y`, and predict on `validation_windows_loader` instead
of `validation_final_loader`.

**Before (lines 241-248):**
```python
predictions = np.clip(
    predict_windows(result.model, validation_final_loader, torch_device),
    0.0, None,
)
metrics = regression_metrics(
    validation_final_windows.y.astype(np.float64),
    predictions,
)
```

**After:**
```python
predictions = np.clip(
    predict_windows(result.model, validation_windows_loader, torch_device),
    0.0, None,
)
metrics = regression_metrics(
    validation_windows.y.astype(np.float64),
    predictions,
)
```

#### `scripts/train_sequence_gru.py`

The standalone training script evaluates both final-window and sliding-
window metrics correctly and stores both. However, its reported "headline"
metrics printed to stdout (lines 369-374) use final-window metrics. Change
these to report sliding-window metrics as the primary output, with final-
window as secondary.

#### Files NOT changed

- Data pipeline (`loader.py`, `labels.py`)
- Normalizer (`normalize.py`)
- Windowing (`windowing.py`)
- Model architecture (`gru.py`)
- Metrics (`metrics.py`)
- Dataset (`dataset.py`)
- Config schema (`schema.py`)

#### Tests

Update existing tests for `train_gru_model` and the sweep function to
verify that early stopping and evaluation use sliding-window metrics.
Specifically:

- `tests/models/test_sequence_training.py`: Verify `train_gru_model`
  selects the best epoch based on validation-window RMSE, not final-window
  RMSE.
- Sweep tests: Verify reported metrics come from sliding windows.

---

## Part 2: Training Logger

### Motivation

The project currently saves per-run artifacts (config, metrics,
predictions, model checkpoints) in timestamped directories, but there is
no centralized, append-only record of all training runs across model types.
Comparing experiments requires navigating individual run directories.

The logger must scale to future model types: GRU, Ridge, CNN, RCNN,
XGBoost, and any other architecture without schema changes.

### Log File

- **Path:** `results/training_log.jsonl`
- **Format:** JSONL (one JSON object per line)
- **Tracking:** Git-tracked, append-only

### Schema: `TrainingLogEntry`

Each line is a JSON object with this structure:

```json
{
  "timestamp": "2026-05-26T14:30:00",
  "model_type": "gru",
  "dataset": "FD001",
  "random_seed": 42,
  "run_dir": "artifacts/models/sequence_gru/20260526-143000",
  "hyperparameters": {
    "window_size": 30,
    "hidden_size": 64,
    "learning_rate": 0.001,
    "num_layers": 1,
    "dropout": 0.0,
    "batch_size": 64,
    "epochs": 50,
    "patience": 8
  },
  "metrics": {
    "rmse": 14.2,
    "mae": 10.8,
    "phm08_score": 450.3
  },
  "best_epoch": 23,
  "training_duration_seconds": 142.5,
  "device": "cuda",
  "extra": {}
}
```

#### Field Definitions

| Field | Type | Required | Description |
|---|---|---|---|
| `timestamp` | str (ISO 8601) | Yes | UTC timestamp when the entry is created |
| `model_type` | str | Yes | Model architecture identifier: `"gru"`, `"ridge"`, `"cnn"`, `"rcnn"`, `"xgboost"`, etc. |
| `dataset` | str | Yes | Dataset identifier, e.g. `"FD001"`, `"FD002"` |
| `random_seed` | int | Yes | Random seed used for reproducibility |
| `run_dir` | str or null | Yes | Path to the full artifact directory, or null for sweep-only runs without artifact persistence |
| `hyperparameters` | dict | Yes | Free-form dict of model-specific hyperparameters. No fixed keys. GRU uses `window_size`, `hidden_size`, etc. Ridge uses `alpha`, `feature_set`. Future models add their own keys. |
| `metrics` | dict | Yes | Evaluation metrics. Always contains `rmse`, `mae`, `phm08_score`. May contain additional model-specific metrics. |
| `best_epoch` | int or null | Yes | Best training epoch (1-indexed) for iterative models, null for non-iterative models (Ridge, XGBoost) |
| `training_duration_seconds` | float | Yes | Wall-clock training time in seconds |
| `device` | str | Yes | Compute device: `"cpu"`, `"cuda"`, `"n/a"` for sklearn models |
| `extra` | dict | Yes | Catch-all for model-specific metadata. E.g., `n_features` for baseline, normalizer stats for GRU. Empty dict `{}` when not needed. |

### Module: `src/turbofan/models/training_log.py`

Two public functions:

#### `build_log_entry`

```python
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
```

Constructs a log entry dict with `timestamp` auto-populated as UTC ISO
8601. Validates that `metrics` contains the three required keys (`rmse`,
`mae`, `phm08_score`). Returns the dict without writing it.

#### `append_training_log`

```python
def append_training_log(
    entry: dict[str, object],
    log_path: Path = Path("results/training_log.jsonl"),
) -> None:
```

Serializes the entry as a single JSON line and appends it to the log file.
Creates the file and parent directories if they don't exist. Uses
`default=str` for JSON serialization to handle Path objects and other
non-standard types.

### Integration Points

#### `scripts/train_sequence_gru.py`

After saving all artifacts, call `build_log_entry` + `append_training_log`
with:
- `model_type="gru"`
- `hyperparameters` from `cfg.sequence` fields
- `metrics` from sliding-window validation (post-fix)
- `run_dir` from the created run directory
- `best_epoch` from `result.best_epoch`
- `device` from resolved device
- Wrap training in `time.time()` calls for duration

#### `scripts/sweep_sequence_gru.py`

After each sweep configuration completes, call `build_log_entry` +
`append_training_log` with:
- `model_type="gru"`
- `hyperparameters` with `window_size`, `hidden_size`, `learning_rate`
  plus held-fixed values from config
- `metrics` from sliding-window validation (post-fix)
- `run_dir=None` (sweep doesn't persist per-run artifacts)
- `best_epoch` from `result.best_epoch`
- Wrap each run in `time.time()` calls for duration

#### `scripts/train_baseline.py` (worktree)

After saving artifacts, call `build_log_entry` + `append_training_log`
with:
- `model_type="ridge"`
- `hyperparameters`: `alpha`, `feature_set`, `windows`
- `best_epoch=None` (non-iterative)
- `device="n/a"`

### Tests: `tests/models/test_training_log.py`

1. **`test_build_log_entry_required_fields`** — Verify all required fields
   are present, timestamp is valid ISO 8601, defaults are applied.

2. **`test_build_log_entry_validates_metrics`** — Missing `rmse`, `mae`,
   or `phm08_score` raises `ValueError`.

3. **`test_append_training_log_creates_file`** — Appending to a
   nonexistent file creates it.

4. **`test_append_training_log_appends_multiple`** — Multiple calls
   produce one valid JSON object per line.

5. **`test_append_training_log_serializes_paths`** — Path objects in
   `extra` or `run_dir` are serialized as strings.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/turbofan/models/sequence_training.py` | Modify | Fix early-stopping to use sliding-window RMSE |
| `scripts/sweep_sequence_gru.py` | Modify | Fix evaluation target + add training log integration |
| `scripts/train_sequence_gru.py` | Modify | Fix headline metrics + add training log integration |
| `src/turbofan/models/training_log.py` | Create | `build_log_entry` and `append_training_log` |
| `tests/models/test_training_log.py` | Create | Unit tests for training log module |
| `tests/models/test_sequence_training.py` | Modify | Update early-stopping assertions |

---

## Out of Scope

- Target normalization (dividing RUL by `max_rul`)
- Removing operational settings from FD001 feature set
- Gradient clipping
- Log rotation or cleanup tooling
- Log visualization/querying CLI
- Remote experiment tracking (MLflow, W&B)
- Modifying the baseline training script in the worktree (documented as
  an integration point but not implemented in this change)
