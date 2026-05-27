# Test Evaluation in GRU Sweep Scripts

**Date:** 2026-05-27
**Status:** Draft

## Goal

Extend `scripts/sweep_sequence_gru.py` and `scripts/sweep_feature_gru.py` to
evaluate each sweep run against the official C-MAPSS held-out test set, reporting
test metrics alongside validation metrics. Test metrics are informational only;
sweep ranking remains sorted by validation `phm08_score`.

## Context

Both sweep scripts currently train on a train split and evaluate on a validation
split derived from `split_by_engine`. The held-out test set (`test_FD00X.txt` +
`RUL_FD00X.txt`) is never touched. The single-run training script
`train_sequence_gru.py` already evaluates on test data via private helpers
(`_evaluate_official_test`, `_align_official_labels_to_eligible_engines`), but
this logic is not reusable.

## Design

### 1. Shared test evaluation module

Extract test evaluation into a public module at
`src/turbofan/models/test_evaluation.py` with two functions:

#### `align_labels_to_eligible_engines(metadata, rul_labels) -> pd.Series`

Maps official RUL labels (one per test engine, ordered by engine ID) to the
subset of engines that have enough cycles to form a window. Uses the
`engine_id - 1` index mapping from `_align_official_labels_to_eligible_engines`
in `train_sequence_gru.py`, then delegates to `align_official_test_labels` for
the final count check.

**Parameters:**
- `metadata: pd.DataFrame` -- final-window metadata with `engine_id` column
- `rul_labels: pd.Series` -- official RUL labels in full test engine order

**Returns:** Float RUL Series aligned to `metadata`.

**Raises:** `ValueError` if an eligible engine ID cannot be mapped to a label.

#### `evaluate_official_test(...) -> dict[str, float] | None`

End-to-end test evaluation: loads test data, normalizes with a pre-fitted
normalizer, builds final windows, predicts, aligns labels, computes metrics.

**Parameters:**
- `data_config: DataConfig` -- data config for file paths and `max_rul`
- `model: GRURULRegressor` -- trained model
- `normalizer: SequenceNormalizer` -- fitted normalizer (train statistics)
- `feature_cols: list[str]` -- feature columns matching the model's input
- `device: torch.device` -- inference device
- `window_size: int` -- sequence window size
- `batch_size: int` -- inference batch size

**Returns:** `{"test_rmse": float, "test_mae": float, "test_phm08_score": float}`
or `None` when test files are missing.

**Behavior:**
1. Load test data via `load_raw_test(data_config)` and labels via
   `load_rul_labels(data_config)`. Return `None` on `FileNotFoundError`.
2. Normalize test data with `normalizer.transform(test_raw)`.
3. Build final windows via `build_final_windows(..., target_col=None)`.
4. Predict via `predict_windows`, clip to >= 0.
5. Align labels via `align_labels_to_eligible_engines`.
6. Compute and return `regression_metrics`.

### 2. Feature sweep: test-time feature engineering

In `sweep_feature_gru.py`, each run uses a different feature set and may apply
rolling feature extraction. The test data must go through the **same** pipeline:

- When `use_rolling=True`: apply the run's `RollingFeatureExtractor` to test
  data before normalization.
- The shared helper receives the already-fitted `normalizer` and `feature_cols`,
  so it does not need to know about rolling features.
- The sweep script transforms test data with the extractor before calling the
  shared helper. This means the sweep script will call a slightly different
  pattern: load test raw, optionally apply rolling features, then call a variant
  of the shared helper that accepts a pre-transformed DataFrame instead of
  loading it internally.

To support this cleanly, add a second entry point:

#### `evaluate_test_from_df(test_df, rul_labels, model, normalizer, feature_cols, device, window_size, batch_size, max_rul) -> dict[str, float]`

Accepts an already-loaded and feature-engineered test DataFrame plus labels.
Normalizes, windows, predicts, aligns, returns metrics. No file I/O.

The feature sweep script will:
1. Load test raw + labels (once, outside the run loop, with graceful `None` on
   missing files).
2. Per run: optionally apply rolling features, then call
   `evaluate_test_from_df`.

The sequence sweep script and `train_sequence_gru.py` will use the simpler
`evaluate_official_test` that handles loading internally.

### 3. Sweep script changes

#### `sweep_sequence_gru.py`

- Add `test_rmse`, `test_mae`, `test_phm08_score` to `RESULT_COLUMNS`.
- After training each run, call `evaluate_official_test(...)`.
- If result is `None`, fill test columns with `float("nan")`.
- Add test metrics to the training log entry's `extra` dict.
- Print test phm08_score alongside validation score.
- Sorting unchanged: `sort_values("phm08_score")`.

#### `sweep_feature_gru.py`

- Same new columns as above.
- Load test raw + labels once before the run loop.
- Per run: apply rolling features if applicable, call
  `evaluate_test_from_df(...)`.
- Same `extra` dict, print, and sort behavior.

### 4. Refactor `train_sequence_gru.py`

Replace private `_evaluate_official_test` and
`_align_official_labels_to_eligible_engines` with calls to the shared module.
The `_prediction_frame` helper stays private since it's specific to the
single-run output format.

### 5. Training log integration

Test metrics go into the `extra` dict of each log entry:
```python
extra={"test_rmse": ..., "test_mae": ..., "test_phm08_score": ...}
```

The existing `metrics` field stays validation-only for backward compatibility.
For `sweep_feature_gru.py`, the existing `extra` keys (`feature_set`,
`corr_threshold`, `n_features`, `rolling_window`) are merged with test metrics.

### 6. Result columns

| Script | Existing columns | New columns |
|--------|-----------------|-------------|
| `sweep_sequence_gru.py` | window_size, hidden_size, learning_rate, best_epoch, rmse, mae, phm08_score | test_rmse, test_mae, test_phm08_score |
| `sweep_feature_gru.py` | feature_set, corr_threshold, n_features, best_epoch, rmse, mae, phm08_score | test_rmse, test_mae, test_phm08_score |

## Decisions

- **Sort by validation, not test.** Test metrics are informational to avoid
  leaking test signal into model selection.
- **Always evaluate test.** No opt-in flag. Graceful skip if files missing.
- **Shared module over inline duplication.** Prevents drift between scripts.
- **Two entry points.** `evaluate_official_test` for simple cases (loads files
  internally); `evaluate_test_from_df` for cases needing custom preprocessing
  (feature sweep rolling features).

## Files to create/modify

| File | Action |
|------|--------|
| `src/turbofan/models/test_evaluation.py` | Create -- shared test eval helpers |
| `scripts/sweep_sequence_gru.py` | Modify -- add test eval per run |
| `scripts/sweep_feature_gru.py` | Modify -- add test eval per run |
| `scripts/train_sequence_gru.py` | Modify -- use shared module |
| `tests/test_test_evaluation.py` | Create -- unit tests for shared module |
