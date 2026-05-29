# Feature Sweep Spec — `feat/unified-feature-sweep`

Retroactive spec and alignment check for the unified feature sweep work. Written after implementation to record what was decided, what was built, and what is still open.

---

## Problem Statement

Before this branch, feature engineering sweeps for Ridge and GRU were handled by two separate scripts with incompatible designs:

- `baseline_feature_comparison.py` — used the canonical `build_feature_pipeline()` and the shared `FeatureSet` vocabulary
- `feature_gru_sweep.py` — had its own bespoke inline feature engineering that bypassed `build_feature_pipeline()` entirely, with a different vocabulary (`raw_plus_rolling`, `top_corr`, `top_corr_rolling`) that predated the unified pipeline (PR #3)

This meant:
- You couldn't compare Ridge and GRU on the same feature engineering inputs
- The GRU sweep was running a different (older) preprocessing contract than `train_sequence_gru.py`
- Running all 4 C-MAPSS subsets required separate invocations with no shared results structure

---

## Design Decisions

### One script, `--model ridge|gru`

A single `turbofan-sweep-features` CLI that routes to the Ridge or GRU training path based on `--model`. Both paths use `build_feature_pipeline()` — the same contract as the train CLIs. Results include a `model` column so outputs from separate runs can be stacked and compared.

Config-driven sweep YAML was explicitly rejected — sweep dimensions (`feature_sets`, `windows`, `lag_steps`) are CLI arguments, not a second config file. The existing subset configs own dataset-specific parameters (`sensor_cols_to_drop`, `n_modes`, `max_rul`); the CLI owns what axes are being swept.

### Feature vocabulary — canonical FeatureSet only

The bespoke `top_corr` / `top_corr_rolling` feature sets from the old GRU sweep were dropped. They relied on inline `select_correlated_sensors()` logic that doesn't exist in `build_feature_pipeline()`, and sensor selection is now handled at EDA time via `sensor_cols_to_drop` in config. Adding it back would require lifting it into the unified pipeline, which is a separate decision.

`raw_plus_lag` was added as a companion to `raw_plus_rolling_mean`: raw normalized sensors concatenated with normalized lag-diff features. It expands the lag family symmetrically with the rolling family — both now have a "raw +" variant. Like `lag`, it yields one spec per lag step. The `_LAG_FEATURE_SETS` frozenset groups `lag` and `raw_plus_lag` for spec expansion, mirroring `_ROLLING_FEATURE_SETS`.

### Lag feature semantics — corrected

The existing `lag` feature set was discovered to compute `x[t-N]` (raw historical value) rather than the intended `(x[t] - x[t-N]) / rolling_mean(x, N)` (normalized lag-difference). This was corrected as part of this branch.

**Correct formula:** `(x[t] - x[t-N]) / rolling_mean(x, window=N, min_periods=1)`

- The numerator is the change in sensor value over N cycles
- The denominator is the local level estimate over the same N cycles, making the feature scale-invariant across sensors
- When the rolling mean is near zero (sensor near-zero), the denominator is replaced with 1.0 to avoid instability — the raw diff is returned instead

**Why this normalizes for sensor noise:** dividing by the rolling mean converts the absolute change into a relative change. A 5-unit change on a sensor averaging 10 is very different from a 5-unit change on a sensor averaging 500.

**Spectral interpretation:** `y[t] = x[t] - x[t-N]` has transfer function `H(z) = 1 - z^(-N)` and magnitude response `2|sin(ωN/2)|`. Small N emphasizes fast/high-frequency changes; large N shifts the passband toward slow degradation trends. Sweeping `lag_steps` therefore provides a crude multi-band decomposition of the sensor signal.

**Lag for GRU:** lag-diff features carry limited additional information for the GRU because the sequence window already contains consecutive cycles — the model can learn the delta directly from context. They are included in the sweep vocabulary for completeness but are not expected to outperform rolling features on GRU.

---

## What Was Built

### `src/turbofan/experiments/feature_sweep.py`

New experiment script. Entry point: `turbofan-sweep-features`.

**Core API:**

```python
run_feature_sweep(
    config_path: Path,
    model: Literal["ridge", "gru"],
    feature_sets: list[str],   # canonical FeatureSet names
    windows: list[int],        # one spec per window for rolling sets
    lag_steps: list[int],      # one spec per lag_step for lag set
    n_jobs: int = 1,           # parallelism for Ridge only
    device: str = "cpu",       # GRU only
    output_path: Path | None = None,
) -> pd.DataFrame
```

**`ExperimentSpec` expansion:**
- `raw` → 1 spec (no windows or lag)
- rolling sets (`rolling_mean`, `rolling_stats`, `raw_plus_rolling_mean`, `raw_plus_rolling_stats`) → 1 spec per window
- lag sets (`lag`, `raw_plus_lag`) → 1 spec per lag_step

**Output columns:**
- Ridge: `model`, `feature_set`, `windows`, `lag_steps`, `n_features`, `alpha`, `rmse`, `mae`, `phm08_score`
- GRU: `model`, `feature_set`, `windows`, `lag_steps`, `n_features`, `best_epoch`, `rmse`, `mae`, `phm08_score`

Results are sorted ascending by `phm08_score`. Optional `--output` writes a CSV.

### `src/turbofan/features/engineering.py` — `_lag_features` corrected

`_lag_features` now computes `(x[t] - x[t-N]) / rolling_mean(x, N)` instead of `x[t-N]`. Column names are unchanged (`{col}_lag_{step}`). Backfill still applies: cycle 1 has no prior history, so diff = 0 and the feature = 0.

### Tests added

- `tests/models/test_feature_sweep.py` — 16 tests: unit tests for `_build_experiment_specs` and `_validate_inputs`, integration tests for `run_feature_sweep` (Ridge + GRU), CLI tests for both models
- `tests/features/test_engineering.py` — 5 new tests for normalized lag-diff: exact values for lag-1 and lag-2, constant sensor → zero, no NaN output, engine boundary isolation; 2 additional tests for `raw_plus_lag` columns and `feature_cols_` attribute

---

## Open Items

### GRU training log not called

Every other GRU training path calls `append_training_log` after training (`train_sequence_gru.py`, `sequence_gru_sweep.py`, `feature_gru_sweep.py`). The new `_evaluate_gru_spec` does not. This is a convention violation, not a functional bug — sweep runs won't appear in `artifacts/training_log.csv`.

**Fix:** call `build_log_entry` + `append_training_log` in `_evaluate_gru_spec`, matching the pattern in `sequence_gru_sweep.py`.

### No default output path

Running without `--output` prints results to stdout only. Results are lost unless the user redirects. The comparable scripts behave the same way, so it is consistent, but at scale across 4 datasets it is inconvenient.

**Proposed default:** `results/feature_sweep_{model}_{fd_subset}.csv` — predictable location, no argument required for standard use.

### All-datasets loop not scripted

The sweep CLI is ready. Running it across all 4 subsets still requires a manual loop:

```powershell
foreach ($fd in "fd001","fd002","fd003","fd004") {
    turbofan-sweep-features --config configs/subsets/$fd.yaml --model ridge ...
    turbofan-sweep-features --config configs/subsets/$fd.yaml --model gru ...
}
```

A `train_all_subsets.py` experiment script that loops over configs and collects results into a single cross-dataset summary CSV was discussed but not built. It can be added after the open items above are resolved.

### `feature_gru_sweep.py` not removed

The old bespoke GRU feature sweep is still present. It can be removed once `turbofan-sweep-features --model gru` is confirmed as the replacement. The `top_corr` / `top_corr_rolling` feature sets it provided are not replicated in the new script — if correlation-based sensor selection is wanted again, it needs to be added to the unified pipeline.

---

## Usage Reference

```powershell
# Ridge sweep — FD001, rolling features across 3 windows
turbofan-sweep-features \
  --config configs/subsets/fd001.yaml \
  --model ridge \
  --feature-sets raw rolling_mean rolling_stats raw_plus_rolling_mean \
  --windows 5 10 20 \
  --lag-steps 1 3 5 \
  --n-jobs 4 \
  --output results/fd001_ridge_feature_sweep.csv

# GRU sweep — FD001, rolling + lag features
turbofan-sweep-features \
  --config configs/subsets/fd001.yaml \
  --model gru \
  --feature-sets raw rolling_mean raw_plus_rolling_mean lag raw_plus_lag \
  --windows 5 10 20 \
  --lag-steps 3 5 \
  --device cpu \
  --output results/fd001_gru_feature_sweep.csv
```

Defaults (if no `--feature-sets`, `--windows`, `--lag-steps` given):
- Feature sets: `raw rolling_mean lag`
- Windows: `5 10 20`
- Lag steps: `2 4 8`
- Config: `configs/default.yaml` (FD001)
