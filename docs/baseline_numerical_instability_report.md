# Baseline Numerical Instability Report

## Summary

The baseline Ridge model was producing validation predictions around
`1e13-1e14`, while Remaining Useful Life (RUL) labels are expected to stay
around `0-125`.

The issue was caused by unstable feature scaling immediately before Ridge. Some
engineered rolling features were divided by near-zero standard deviations during
normalization, producing extremely large but finite validation feature values.

## Evidence

A diagnostic run on a representative subset showed:

| Item | Value |
| --- | ---: |
| Validation transformed feature absmax | `5.45e13` |
| Raw Ridge prediction absmax | `9.37e13` |
| Minimum normalization std | `6.59e-15` |

The largest exploding columns were rolling engineered features such as:

- `s_17_rstd_3`
- `s_2_rmean_3`
- `s_11_rstd_3`
- `s_13_rstd_3`
- `s_8_rstd_3`

## Root Cause

`OperationalNormalizer` grouped rows by exact operational condition and computed
per-group z-score statistics.

The existing guard only handled exact zero standard deviations:

```python
replace(0.0, 1.0)
```

That missed near-zero values like `1e-15`. When validation values were divided by
those tiny standard deviations, normal differences were amplified into
`1e10-1e14`.

A secondary issue was that Ridge received engineered features directly, without
final imputation or scaling after rolling feature creation and identifier
dropping.

## Impact

The instability caused:

- Catastrophically large validation features before Ridge
- Raw predictions far outside the valid RUL range
- Unreliable validation metrics
- PHM08 overflow risk for large residuals
- Sensitivity to train/validation distribution mismatch

## Fixes Applied

The baseline pipeline now uses:

```text
features
drop_identifiers
low_variance_filter
imputer
scaler
model
```

Changes made:

- Added near-zero standard deviation protection in `OperationalNormalizer` using
  `std_floor=1e-3`
- Added post-feature low-variance filtering before Ridge
- Added median imputation
- Added `StandardScaler` after feature engineering and identifier dropping
- Preserved pandas feature names into Ridge
- Increased default Ridge alpha from `1.0` to `100.0`
- Added prediction clipping to `[0, cfg.data.max_rul]`
- Logged raw prediction min/max before clipping
- Made PHM08 scoring overflow-safe by clipping exponent inputs
- Added `scripts/sweep_baseline_alpha.py` for alpha comparison

## After Fix

The same repro improved to:

| Item | Before | After |
| --- | ---: | ---: |
| Validation transformed feature absmax | `5.45e13` | `1.38e3` |
| Raw Ridge prediction absmax | `9.37e13` | `1.09e3` |

Training and evaluation now clip predictions into the configured RUL range,
currently:

```text
[0, 125]
```

## Regression Tests Added

Coverage now includes:

- Transformed train/validation features contain no `NaN` or `inf`
- Validation transformed features are not catastrophically large
- Raw predictions are finite on the toy instability case
- Clipped predictions stay inside `[0, rul_cap]`
- Ridge receives DataFrame feature names
- Near-zero normalization standard deviations do not amplify validation values
- PHM08 score does not overflow on large residuals

## Verification

Commands run:

```bash
ruff check src/ tests/ scripts/
mypy src/turbofan
pytest -p no:cacheprovider --basetemp=.pytest-basetemp-codex
```

Result:

```text
251 passed
```
