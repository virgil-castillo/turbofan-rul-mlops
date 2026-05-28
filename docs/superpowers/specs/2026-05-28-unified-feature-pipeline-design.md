# Unified Feature Pipeline Design

**Date:** 2026-05-28
**Status:** Draft

## Problem

Ridge and GRU are fed fundamentally different data today:

| | Ridge | GRU |
|---|---|---|
| Sensor dropping | Yes (`sensor_cols_to_drop`) | No — all 21 sensors |
| Features | Rolling stats on kept sensors | Raw values, all sensors |
| Op cols in model input | No (filtered by selector) | Yes (`op_1-3` included) |
| Normalization | `OperatingModeNormalizer` (auto feature_cols) | `OperatingModeNormalizer` (explicit, includes op cols) |
| Post-normalization scaling | `StandardScaler` | None |
| Feature set | Hard-coded rolling | Hard-coded raw |

Experiments across models are not comparable. The EDA-derived sensor drop decisions in `configs/subsets/` are ignored by the GRU entirely. There is no way to run feature engineering experiments without touching code.

## Goal

1. A single shared preprocessing contract: both Ridge and GRU receive the same clean, normalized, sensor-only feature matrix.
2. A config-driven feature layer on top: `feature_set` in `FeatureConfig` selects which engineered features both models receive. Swapping feature sets produces comparable experiments without code changes.

More exotic feature engineering (FFT, degradation slope, composite health indicators) is explicitly deferred — this refactor establishes the framework they will slot into later.

## Preprocessing Contract

**Shared input to both models:** non-dropped sensor columns only.

- `s_1`–`s_21` minus `sensor_cols_to_drop` from config
- No operating setting columns (`op_1`, `op_2`, `op_3`)
- No `engine_id`, `cycle`, `rul`
- Values z-scored per operating mode (`n_modes=1` → global z-score; `n_modes>1` → per-KMeans-cluster z-score)
- No `StandardScaler` on top — normalization is already handled by `OperatingModeNormalizer`

Feature engineering (rolling, lag) is applied **after** normalization. Rolling and lag operate on already-normalized sensor values, which are in a consistent scale across modes.

## Config Changes

**`src/turbofan/config/schema.py` — `FeatureConfig`:**

Add:
```yaml
feature_set: raw          # raw | rolling_mean | rolling_stats | raw_plus_rolling_mean | raw_plus_rolling_stats | lag
windows: [10]             # rolling window sizes (moved from ModelConfig)
lag_steps: [1]            # lag offsets in cycles (used when feature_set includes lag)
```

`windows` moves from `ModelConfig` to `FeatureConfig` — it is a feature engineering parameter, not a model parameter. `ModelConfig.windows` is removed.

`feature_set` defaults to `raw`.

## Feature Sets

| `feature_set` | Output columns per kept sensor |
|---|---|
| `raw` | sensor value |
| `rolling_mean` | rolling mean per window |
| `rolling_stats` | rolling mean + std + min + max per window |
| `raw_plus_rolling_mean` | sensor value + rolling mean per window |
| `raw_plus_rolling_stats` | sensor value + rolling mean + std + min + max per window |
| `lag` | lagged sensor values at each step in `lag_steps` |

Rolling uses `min_periods=1` to avoid NaN at the start of each engine's time series. Lag features for the first `max(lag_steps)` cycles of each engine are backfilled within the engine group (no engine boundary crossing).

## Design

### 1. `build_feature_pipeline` — shared preprocessor

**File:** `src/turbofan/features/pipeline.py`

Four steps in sequence:

```
SensorDropper → OperatingModeNormalizer → SensorColumnSelector → FeatureEngineer
```

**Step 1 — `SensorDropper`**
Unchanged. Drops `sensor_cols_to_drop`. Op cols pass through for KMeans.

**Step 2 — `OperatingModeNormalizer`**
Receives explicit `feature_cols` = `[f"s_{i}" for i in range(1, 22) if f"s_{i}" not in (sensor_drop or [])]`. Op cols available in the DataFrame for KMeans but not normalized.

**Step 3 — `SensorColumnSelector` (new)**
Small sklearn-compatible transformer in `pipeline.py`.
- `fit`: records `[c for c in X.columns if c.startswith("s_")]` as `feature_cols_`
- `transform`: returns `X[self.feature_cols_]`

Output is a clean DataFrame containing only kept, normalized sensor columns.

**Step 4 — `FeatureEngineer` (new)**
Sklearn-compatible transformer in `src/turbofan/features/engineering.py`.
- Constructor params: `feature_set`, `windows`, `lag_steps`
- `fit`: records the input sensor columns as `sensor_cols_`, computes output column names as `feature_cols_`
- `transform`: applies the configured transformation, returns the engineered DataFrame

For `feature_set: raw`, this is a pass-through. For all others, it computes the requested features and returns only those columns (or raw + engineered for `raw_plus_*` variants). Rolling is computed per engine group (no crossing engine boundaries). Lag is computed per engine group with backfill.

**Signature of `build_feature_pipeline`:**
```python
def build_feature_pipeline(
    op_cols: list[str] | None = None,
    sensor_drop: list[str] | None = None,
    n_modes: int = 1,
    random_state: int = 42,
    feature_set: str = "raw",
    windows: list[int] | None = None,
    lag_steps: list[int] | None = None,
) -> Pipeline:
```

`windows` parameter removed from `build_baseline_pipeline` (moved here).

### 2. Ridge — `build_baseline_pipeline`

**File:** `src/turbofan/models/baseline.py`

Reduced to two stages:

```
build_feature_pipeline(...) → Ridge
```

**Steps removed:**
- `RollingFeatureExtractor`
- `_ModelFeatureSelector`
- `_LowVarianceFeatureDropper`
- `FunctionTransformer(_drop_identifier_columns)`
- `SimpleImputer`
- `StandardScaler`

**Dead code deleted from `baseline.py`:**
- `BaselineFeatureSet` type alias
- `ROLLING_MARKERS`, `DEFAULT_BASELINE_WINDOWS`
- `_is_rolling_feature`, `_is_raw_sensor_feature`
- `_LowVarianceFeatureDropper`
- `_ModelFeatureSelector`
- `_drop_identifier_columns`

**Signature changes:**
- Remove `windows` and `feature_set` parameters (both now come from `FeatureConfig` and flow through `build_feature_pipeline`)

**`train_baseline.py`:** Pass `feature_set`, `windows`, `lag_steps` from `cfg.features` to `build_baseline_pipeline` → `build_feature_pipeline`.

### 3. GRU — `train_sequence_gru.py`

Replace manual `OperatingModeNormalizer` instantiation with `build_feature_pipeline`:

```python
pipeline = build_feature_pipeline(
    op_cols=op_cols,
    sensor_drop=cfg.features.sensor_cols_to_drop or None,
    n_modes=cfg.features.n_modes,
    random_state=cfg.data.random_seed,
    feature_set=cfg.features.feature_set,
    windows=cfg.features.windows,
    lag_steps=cfg.features.lag_steps,
)
train_normalized = pipeline.fit_transform(train_df)
val_normalized = pipeline.transform(val_df)

feature_cols = pipeline.named_steps["feature_engineer"].feature_cols_
```

`feature_cols` replaces the hardcoded `default_feature_cols()` call. `input_size = len(feature_cols)`.

The artifact stores `feature_cols` (post-engineering column list) and the normalizer payload from `pipeline.named_steps["normalizer"].to_payload()`. Keys `normalizer_type` and `normalizer_payload` are preserved.

`_evaluate_official_test` uses `pipeline.transform(test_raw)` instead of `normalizer.transform(test_raw)`.

### 4. GRU sweep — `sequence_gru_sweep.py`

Same change as the training CLI: replace `default_feature_cols()` and manual normalizer with `build_feature_pipeline`. `feature_set`, `windows`, `lag_steps` flow from config.

### 5. Dead code and deprecations

**`src/turbofan/sequences/normalize.py`:**
- `default_feature_cols()` — delete. No longer used once GRU and sweep adopt `build_feature_pipeline`.
- `SequenceNormalizer` — basic global z-score normalizer, fully superseded by `OperatingModeNormalizer`. Still imported by `src/turbofan/models/test_evaluation.py` as the type for `evaluate_test_from_df`'s `normalizer` parameter. Update that function to accept `OperatingModeNormalizer`, then delete `SequenceNormalizer`. If `evaluate_test_from_df` is dead code after the GRU CLI refactor, delete the function too.

**`src/turbofan/features/rolling.py`:**
- `RollingFeatureExtractor` — rolling logic moves into `FeatureEngineer`. Delete this file.

### 6. Inference compatibility

**`src/turbofan/inference/predictors.py`:** No changes required.

- **GRU:** `GRUPredictor` reconstructs `OperatingModeNormalizer` from `normalizer_payload`, calls `normalizer.transform(raw_frame)`, then passes to `build_final_windows(..., feature_cols=self._feature_cols)`. Windowing already selects only `feature_cols`, so op cols left in the DataFrame are naturally ignored. After this refactor, `feature_cols` in the artifact will be the post-engineering column list — inference handles this correctly with no code changes.
- **Ridge:** `RidgePredictor` loads the full sklearn pipeline via joblib. `SensorColumnSelector` and `FeatureEngineer` are embedded in the pipeline and carried forward automatically.

**Existing GRU artifacts:** Incompatible. Checkpoints trained before this refactor store `feature_cols` with op cols and all 21 sensors. The existing payload validation will reject them with a retrain error.

### 7. Testing

**`tests/features/test_pipeline.py`:**
- Remove rolling-related tests
- Add: output contains only `s_*` columns for `feature_set: raw`
- Add: op cols and `engine_id` absent from output
- Add: `SensorColumnSelector` fit/transform behavior
- Add: `FeatureEngineer` output shapes and column names for each `feature_set`
- Add: rolling respects engine boundaries (no NaN leakage across engines)
- Add: lag backfills within engine group, does not cross engine boundary

**`tests/features/test_sensor_dropper.py`:** No changes expected.

**`tests/models/test_baseline.py`:**
- Remove tests referencing `windows`, `feature_set` (old Ridge-specific), rolling columns
- Update to two-step pipeline

**`tests/models/test_train_baseline_cli.py`:**
- Remove `windows`/`feature_set` Ridge-specific argument references

**`tests/sequences/`:**
- `test_normalize.py`: remove `default_feature_cols` tests
- Update fixtures that call `default_feature_cols()`
- `test_train_sequence_gru_cli.py`: update monkeypatches of `default_feature_cols`
- `test_sweep_sequence_gru.py`: same

**`tests/models/test_test_evaluation.py`:**
- Update `SequenceNormalizer` usages to `OperatingModeNormalizer`

## What this is not

- Not FFT, degradation slope, trend indicators, or composite health indicators — those drop into `FeatureEngineer` as new `feature_set` values in a future spec
- Not a change to evaluation, serving, or Docker infrastructure
- Not a new model
- Not experiment sweep scripts for feature sets — those are a follow-on once the framework is in place
