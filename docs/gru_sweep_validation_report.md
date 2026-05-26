# GRU Hyperparameter Sweep: Validation Report

**Date:** 2026-05-26
**Sweep job:** SLURM 4294027
**Dataset:** C-MAPSS FD001
**Status:** INVALID -- results must not be presented

---

## Executive Summary

All 36 GRU sweep results are invalid. RMSE values range from 0.0 to 5.5,
while literature GRU/LSTM models on FD001 report RMSE 12--20 and PHM08
scores in the hundreds to thousands. Eight configurations report RMSE = 0.0
exactly. The root cause is a critical evaluation bug: the sweep computes
metrics against validation targets that are **all zeros by construction**.

The model architecture, feature composition, and data loading are correct.
No target leakage, no engine-ID contamination. The bug is isolated to how
the validation metric is defined.

---

## 1. Results vs. Literature

| Source | RMSE | PHM08 Score |
|---|---|---|
| Literature GRU/LSTM (FD001) | 12 -- 20 | 200 -- 2,000 |
| Tabular Ridge baseline (this repo) | 23.5 -- 24.5 | 160K -- 511K |
| **GRU sweep best (this run)** | **0.0** | **0.0** |
| **GRU sweep worst (this run)** | **5.6** | **12.0** |

The GRU results are 1--2 orders of magnitude too low for RMSE and 3--5
orders of magnitude too low for PHM08. Eight of 36 configurations report
exact zeros across all three metrics.

---

## 2. Validation Checklist

### 2.1 Feature Composition -- PASS

The model receives 24 features per timestep:
`op_1, op_2, op_3, s_1, s_2, ..., s_21`.

- `engine_id` is **not** in the feature set.
- `cycle` is **not** in the feature set.
- `rul` (the target) is **not** in the feature set.
- No target leakage was found.

Verified by inspecting `default_feature_cols()` in
[normalize.py](../src/turbofan/sequences/normalize.py) and confirming the
windowing code in [windowing.py](../src/turbofan/sequences/windowing.py)
extracts only `feature_list` columns into `X`.

### 2.2 Feature Normalization -- PASS

`SequenceNormalizer` applies z-score normalization to the 24 feature
columns only. The `rul` column is not in `feature_cols` and is not
transformed. Means and standard deviations are fit on the training split
and applied to both training and validation data. Zero-std columns are
replaced with 1.0 to avoid division by zero.

### 2.3 RUL Label Construction -- PASS

`compute_rul_labels` computes `rul = min(125, max_cycle - current_cycle)`.
The piecewise-linear capping at 125 matches FD001 literature convention.
Values range from 0 (at failure) to 125 (early life).

### 2.4 Train/Validation Split -- PASS

`split_by_engine` splits at the engine level (20% = 20 engines for
validation) with a fixed seed. No data leakage between engines.

### 2.5 Model Architecture -- PASS

`GRURULRegressor`: single-layer GRU with a `Linear(hidden_size, 1)` head.
Output squeezed to shape `(batch_size,)`. Standard design for sequence
regression.

### 2.6 Training Loop -- PASS (with caveat)

`train_gru_model` trains with Adam + MSE loss on sliding windows (RUL
range 0--125). The training loss is computed correctly. However, **early
stopping** selects the epoch with the lowest RMSE on
`validation_final_loader` -- which is the degenerate all-zeros target
(see Section 3). This means the training loop actively selects the model
state that predicts closest to zero.

### 2.7 Evaluation Metric Targets -- **FAIL (ROOT CAUSE)**

See Section 3.

---

## 3. Root Cause: Degenerate Evaluation Targets

### The Bug

The sweep evaluates each configuration on `validation_final_windows`:

```python
# sweep_sequence_gru.py, lines 241-248
predictions = np.clip(
    predict_windows(result.model, validation_final_loader, torch_device),
    0.0, None,
)
metrics = regression_metrics(
    validation_final_windows.y.astype(np.float64),  # <-- ALL ZEROS
    predictions,
)
```

`build_final_windows` constructs one window per engine, ending at the
engine's last cycle. In C-MAPSS **training data**, every engine runs to
failure, so the last cycle always has `rul = max_cycle - max_cycle = 0`.

**Empirical confirmation:**

```
Window size 15 -> final_windows.y unique: [0.], ALL ZEROS: True
Window size 30 -> final_windows.y unique: [0.], ALL ZEROS: True
Window size 45 -> final_windows.y unique: [0.], ALL ZEROS: True
```

All 20 validation engines have `rul = 0` at their final cycle.

### Why This Produces the Observed Results

With `y_true = [0, 0, ..., 0]`:

- **RMSE** = `sqrt(mean(y_pred^2))` -- measures distance from zero, not
  prediction quality.
- **MAE** = `mean(|y_pred|)` -- same problem.
- **PHM08** = `sum(exp(|y_pred| / k) - 1)` -- exponentially penalizes
  non-zero predictions.

Models that happen to predict near-zero (e.g., undertrained models at
LR=0.0001 with small hidden size) score "perfectly." Models that actually
learned meaningful RUL patterns score worse because they predict non-zero
values for the last window.

### The 8 Zero-Metric Configurations

All 8 have `learning_rate=0.0001` with `hidden_size` 32 or 64 and
`best_epoch=1`. These models barely learned from the data. Their
near-random predictions happened to cluster near zero after `clip(0)`,
giving RMSE = 0.0 against all-zero targets. These are the worst-performing
models being ranked as the best.

### Why Early Stopping Compounds the Problem

`train_gru_model` uses the same `validation_final_loader` for its early
stopping criterion:

```python
# sequence_training.py, lines 94-96
final_metrics = _evaluate_loader(model, validation_final_loader, device)
current_metric = final_metrics["rmse"]
```

This means the training loop **selects the model checkpoint that best
predicts zero**. Any epoch where the model starts learning meaningful RUL
patterns (predicting non-zero values) is penalized and discarded.

---

## 4. Secondary Issues

These do not explain the anomalous results but should be addressed for a
valid rerun.

### 4.1 No Target Normalization

The model predicts raw RUL in [0, 125] while features are z-score
normalized to approximately N(0, 1). MSE loss on raw targets is dominated
by high-RUL samples (error^2 can reach 15,625). Standard practice is to
normalize the target to [0, 1] by dividing by `max_rul`, then rescale
predictions at evaluation time.

### 4.2 Operational Settings in FD001

`op_1, op_2, op_3` are included in the 24-feature set. FD001 has a single
operating condition, so these columns are near-constant. Including them
adds 3 noise dimensions. For FD002/FD004 (six operating conditions), they
would be informative.

### 4.3 No Gradient Clipping

GRU training with raw-scale targets in [0, 125] and MSE loss can produce
large gradients, especially early in training. No `torch.nn.utils.clip_grad_norm_`
is applied. This is a stability concern but not a correctness bug.

---

## 5. Recommended Fixes

### Fix 1: Evaluate on Sliding Windows (Required)

Replace the final-window evaluation with sliding-window evaluation, which
spans the full RUL range [0, 125]:

```python
# In sweep_sequence_gru.py, change evaluation to use validation_windows:
metrics = regression_metrics(
    validation_windows.y.astype(np.float64),
    predictions_on_sliding_windows,
)
```

Alternatively, evaluate on the official test set (`test_FD001.txt` +
`RUL_FD001.txt`), where engines are cut off before failure and RUL > 0.

### Fix 2: Fix Early Stopping Target (Required)

In `train_gru_model`, change the early-stopping criterion from
`validation_final_loader` to `validation_windows_loader`:

```python
# In sequence_training.py, line 94-96:
window_metrics = _evaluate_loader(model, validation_windows_loader, device)
current_metric = window_metrics["rmse"]
```

### Fix 3: Normalize Targets (Recommended)

Divide RUL by `max_rul` before training. Rescale predictions by
`max_rul` before computing metrics.

### Fix 4: Drop Near-Constant Features for FD001 (Optional)

Remove `op_1, op_2, op_3` from the feature set for FD001, or use the
existing `sensor_std_threshold` mechanism to drop low-variance columns.

### Fix 5: Add Gradient Clipping (Optional)

Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
after `loss.backward()` in `_train_one_epoch`.

---

## 6. Impact Assessment

| Aspect | Status |
|---|---|
| All 36 sweep results | Invalid, must be discarded |
| Best configuration selection | Inverted (worst models ranked first) |
| Trained model checkpoints | Actively harmed by degenerate early stopping |
| Data pipeline (`loader`, `labels`, `normalizer`) | Correct, no changes needed |
| Model architecture (`GRURULRegressor`) | Correct, no changes needed |
| Training loop (`_train_one_epoch`) | Correct for loss computation |
| Sweep script structure | Correct modulo evaluation target |

The fix is localized to the evaluation target in the sweep script and the
early-stopping target in the training loop. After applying Fixes 1 and 2,
the sweep should be rerun from scratch.

---

## 7. Appendix: Data Flow Trace

```
load_raw_train(FD001)           # 20,631 rows, 100 engines
  |
add_rul_column(max_rul=125)     # rul in [0, 125], 0 at failure
  |
split_by_engine(0.2, seed=42)   # 80 train / 20 val engines
  |
SequenceNormalizer.fit(train)   # z-score on 24 feature cols only
  |                               rul column NOT touched
  |
transform(val)                  # apply train stats to val features
  |
build_final_windows(val, ws)    # 1 window per engine, last ws rows
  |                               target = rul at final row = 0 ALWAYS
  |
build_sliding_windows(val, ws)  # many windows per engine
  |                               target = rul at each position, [0, 125]
  |
GRURULRegressor(24, hs, 1, 0)  # GRU encoder + Linear(hs, 1) head
  |
train_gru_model(...)            # MSE on sliding windows (correct)
  |                               early stop on final windows (BROKEN)
  |
regression_metrics(y=0, pred)   # RMSE = sqrt(mean(pred^2)), not RUL error
```
