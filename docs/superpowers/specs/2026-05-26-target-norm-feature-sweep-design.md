# Target Normalization & GRU Feature Engineering Sweep

**Date:** 2026-05-26
**Status:** Draft

---

## 1. Overview

Two changes to the GRU training pipeline:

1. **Target normalization** -- divide RUL labels by `max_rul` before loss
   computation so the model trains on [0, 1] targets instead of raw [0, 125].
   Rescale predictions before evaluation metrics.

2. **Feature engineering sweep script** -- a new
   `scripts/sweep_feature_gru.py` that trains the GRU with fixed
   hyperparameters across multiple feature sets to find the best input
   representation. Feature sets: raw, raw+rolling, correlation-filtered,
   and correlation-filtered+rolling.

---

## 2. Target Normalization

### 2.1 Motivation

The GRU currently trains with MSE loss on raw RUL targets in [0, 125].
This causes two problems:

- **Gradient magnitude:** Squared errors can reach 15,625 (125^2), making
  gradients large and training sensitive to learning rate.
- **Sample weighting bias:** High-RUL samples dominate the loss because
  their errors are squared at a larger scale. A 10-unit error at RUL=120
  contributes the same loss as a 10-unit error at RUL=5, but the squared
  magnitudes of typical errors are much larger for high-RUL windows.

Normalizing targets to [0, 1] equalizes gradient scale and is standard
practice in RUL prediction literature.

### 2.2 Design

**`train_gru_model` changes:**

Add a `max_rul: int` parameter. Inside the training loop:

- `_train_one_epoch`: accepts `max_rul` and divides each batch's target
  tensor by `max_rul` on the fly before loss computation. The
  `WindowedSequences.y` array is not modified. The model learns to
  predict in [0, 1].
- `_evaluate_loader`: multiply model outputs by `max_rul` before
  computing `regression_metrics`. Early stopping and history metrics
  remain in real RUL units so they are directly comparable to literature
  and prior runs.

**`predict_windows` changes:**

Add a `max_rul: int` parameter. Multiply raw model outputs by `max_rul`
before returning. Callers receive predictions in real RUL units.

**`SequenceConfig` changes:**

None. `max_rul` already lives on `DataConfig`. The training function
receives it as an explicit parameter from the caller (sweep script or
training script).

**What does NOT change:**

- `WindowedSequences.y` -- stays in raw [0, 125] scale.
- `SequenceNormalizer` -- feature normalization unchanged.
- `regression_metrics` -- always receives real-scale values.
- `GRURULRegressor` -- architecture unchanged (unbounded scalar output).
- `build_sliding_windows` / `build_final_windows` -- data pipeline
  unchanged.

### 2.3 Data Flow After Change

```
WindowedSequences.y          [0, 125]   (unchanged)
  |
_train_one_epoch             / max_rul -> [0, 1] for MSE loss only
  |
_evaluate_loader             model output * max_rul -> [0, 125] for metrics
  |
predict_windows              model output * max_rul -> [0, 125] returned
  |
caller: regression_metrics   receives [0, 125] predictions (unchanged)
```

### 2.4 Test Plan

- Unit test: `_train_one_epoch` with `max_rul=125` produces smaller loss
  values than with `max_rul=1` (identity, current behavior) on the same
  data.
- Unit test: `predict_windows` with `max_rul=125` returns values in the
  expected range (roughly [0, 125] after training).
- Unit test: `_evaluate_loader` returns metrics in real RUL units
  regardless of `max_rul`.
- Integration: `train_gru_model` with `max_rul=125` converges (best_epoch
  > 1) and returns metrics in the same unit range as the unnormalized
  variant.

---

## 3. Feature Engineering Sweep

### 3.1 Feature Sets

The sweep evaluates four feature families. GRU hyperparameters are fixed
to the `configs/default.yaml` defaults (`window_size=30`,
`hidden_size=64`, `learning_rate=0.001`, `patience=8`, `epochs=50`).

| ID | Name | Description | Feature columns |
|---|---|---|---|
| `raw` | Raw sensors + ops | Current 24-feature default | `op_1-3`, `s_1-21` |
| `raw_plus_rolling` | Raw + rolling stats | Raw features plus rolling mean/std/min/max per sensor | `op_1-3`, `s_1-21`, + 4 rolling cols per sensor |
| `top_corr` | Correlation-filtered | Sensors with \|corr(sensor, RUL)\| above a threshold, plus `op_1-3` | `op_1-3` + filtered `s_*` subset |
| `top_corr_rolling` | Filtered + rolling | Correlation-filtered sensors plus their rolling stats | `op_1-3` + filtered `s_*` + 4 rolling cols per filtered sensor |

### 3.2 Correlation-Based Feature Selection

A new function `select_correlated_sensors` in
`src/turbofan/sequences/feature_selection.py`:

```
select_correlated_sensors(
    df: pd.DataFrame,
    target_col: str = "rul",
    threshold: float = 0.5,
) -> list[str]
```

Computes the absolute Pearson correlation between each `s_*` column and
`target_col` across all rows of the input DataFrame. Returns sensor
column names where `|r| >= threshold`, sorted by descending `|r|`.

This runs on the **training split only** to avoid leaking validation
statistics into feature selection.

The sweep varies the threshold across `[0.3, 0.5, 0.7]` by default
(CLI-configurable), producing separate runs for `top_corr` and
`top_corr_rolling` at each threshold level.

### 3.3 Rolling Feature Integration

The sweep reuses `RollingFeatureExtractor` from
`src/turbofan/features/rolling.py`. For feature sets that include rolling
features:

1. Apply `RollingFeatureExtractor` (fitted on training split) to both
   training and validation DataFrames before normalization and windowing.
2. The rolling window size is a CLI argument (default: 10, matching the
   baseline winner). This is the rolling aggregation window, not the GRU
   sequence window.
3. Rolling features add 4 columns per sensor per rolling window
   (`s_N_rmean_W`, `s_N_rstd_W`, `s_N_rmin_W`, `s_N_rmax_W`).
4. The resulting feature columns (raw + rolling, or filtered + rolling)
   are passed to `SequenceNormalizer` and `build_sliding_windows`.

### 3.4 Sweep Script: `scripts/sweep_feature_gru.py`

Mirrors the structure of `sweep_sequence_gru.py`:

**CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/default.yaml` | Project config path |
| `--feature-sets` | `raw raw_plus_rolling top_corr top_corr_rolling` | Feature families to sweep |
| `--corr-thresholds` | `0.3 0.5 0.7` | Thresholds for correlation-based feature sets |
| `--rolling-window` | `10` | Rolling aggregation window size |
| `--device` | `cpu` | Torch device |
| `--output` | `None` | CSV path for results |

**Sweep grid:**

For `raw` and `raw_plus_rolling`: one run each (no threshold axis).
For `top_corr` and `top_corr_rolling`: one run per threshold value.

Total runs with defaults: `1 + 1 + 3 + 3 = 8`.

**Per-run flow:**

1. Determine feature columns for this run's feature set (and threshold
   if applicable).
2. If rolling: apply `RollingFeatureExtractor` to train and val
   DataFrames, update feature column list.
3. Fit `SequenceNormalizer` on training split with this run's feature
   columns.
4. Build sliding windows for train and validation.
5. Build final windows for validation (used for history logging, not
   early stopping).
6. Train with `train_gru_model` using fixed GRU hyperparameters from
   config and `max_rul` from config (target normalization applied).
7. Evaluate on validation sliding windows; compute and log metrics.
8. Append to training log via `append_training_log`.

**Output columns:**

`feature_set`, `corr_threshold` (NaN for non-correlation sets),
`n_features`, `best_epoch`, `rmse`, `mae`, `phm08_score`.

### 3.5 Data Flow

```
load_raw_train(FD001)
  |
add_rul_column(max_rul=125)
  |
split_by_engine(0.2, seed=42)       # 80 train / 20 val
  |
  +-- [if top_corr or top_corr_rolling]:
  |     select_correlated_sensors(train, threshold=T)
  |       -> filtered sensor list
  |
  +-- [if raw_plus_rolling or top_corr_rolling]:
  |     RollingFeatureExtractor(window=10).fit(train)
  |       -> transform(train), transform(val)
  |       -> expanded feature column list
  |
  +-- Determine final feature_cols for this run
  |
SequenceNormalizer(feature_cols).fit(train)
  |
build_sliding_windows(train, feature_cols, ws=30)
build_sliding_windows(val, feature_cols, ws=30)
build_final_windows(val, feature_cols, ws=30)
  |
train_gru_model(..., max_rul=125)    # target normalization applied
  |
predict_windows(..., max_rul=125) on val sliding windows
  |
regression_metrics(y_true, predictions)  # real RUL units
```

### 3.6 Test Plan

**`select_correlated_sensors`:**

- Returns only sensors above the threshold.
- Returns empty list when threshold is 1.0 (no sensor has perfect
  correlation).
- Sorts by descending |r|.
- Does not return non-sensor columns.
- Handles the case where no sensors pass the threshold (raises
  `ValueError`).

**`sweep_feature_gru.py` (unit-testable `run_feature_sweep` function):**

- Validates feature set names.
- Validates correlation thresholds (must be in (0, 1)).
- Validates rolling window (must be positive).
- Produces correct number of runs for the grid (1+1+len(thresholds)+len(thresholds)).

**Integration (manual, via CLI):**

- Runs to completion on FD001 with default arguments.
- Produces a CSV with expected columns.
- Metrics are in literature-plausible ranges (RMSE 12-25, PHM08 > 100).
- Training log entries are appended.

---

## 4. Files Changed

### New files:

- `src/turbofan/sequences/feature_selection.py` --
  `select_correlated_sensors` function.
- `tests/sequences/test_feature_selection.py` -- unit tests.
- `scripts/sweep_feature_gru.py` -- feature engineering sweep CLI.

### Modified files:

- `src/turbofan/models/sequence_training.py` -- `train_gru_model`,
  `predict_windows`, `_train_one_epoch`, `_evaluate_loader` gain
  `max_rul` parameter.
- `tests/models/test_gru.py` -- updated tests for `max_rul` parameter.
- `scripts/sweep_sequence_gru.py` -- pass `max_rul` to `train_gru_model`
  and `predict_windows`; rescale no longer needed at call site since
  `predict_windows` handles it.

---

## 5. Out of Scope

- Gradient clipping (separate concern, optional improvement).
- Dropping `op_1-3` for FD001 (the `raw` feature set keeps them; the
  correlation filter will naturally drop them if they don't correlate
  with RUL).
- Re-running the hyperparameter sweep (use config defaults for now).
- Multi-dataset support (FD002-FD004). The sweep is FD001-focused but
  the code is dataset-agnostic via the config.
