# Feature Engineering: Sub-project 3 Design Spec

**Date:** 2026-05-23
**Sub-project:** 3 of 6
**Depends on:** Sub-project 1 (Foundation)
**Status:** Approved

---

## Goal

Build a reusable, sklearn-compatible feature engineering pipeline in `src/turbofan/features/` that transforms raw C-MAPSS sensor data into model-ready features. The pipeline drops constant sensors, computes multi-window rolling statistics, and normalizes by operational condition. It supports all four FD subsets (FD001-FD004) uniformly and prevents data leakage via sklearn's fit/transform semantics.

## Architecture

Three custom sklearn transformers (`SensorDropper`, `RollingFeatureExtractor`, `OperationalNormalizer`) are chained into an `sklearn.pipeline.Pipeline` by a factory function. Each transformer follows `BaseEstimator`/`TransformerMixin` conventions: `fit()` learns statistics from training data, `transform()` applies them to any data. This ensures normalization parameters, dropped columns, and condition-specific stats are never computed on test data. The pipeline accepts and returns pandas DataFrames throughout.

## Tech Stack

Python 3.12, pandas, numpy, scikit-learn, pytest

---

## File Structure

```
turbofan-rul-mlops/
├── src/turbofan/
│   └── features/
│       ├── __init__.py
│       ├── sensor_dropper.py         # SensorDropper transformer
│       ├── rolling.py                # RollingFeatureExtractor transformer
│       ├── normalizer.py             # OperationalNormalizer transformer
│       └── pipeline.py               # build_feature_pipeline() factory
├── tests/
│   └── features/
│       ├── test_sensor_dropper.py
│       ├── test_rolling.py
│       ├── test_normalizer.py
│       └── test_pipeline.py
└── pyproject.toml                    # add scikit-learn to dependencies
```

---

## Component Designs

### 1. `SensorDropper` — constant sensor removal

```python
class SensorDropper(BaseEstimator, TransformerMixin):
    """Drop sensor columns with zero variance in the training set.

    Discovers sensor columns by the ``s_*`` naming convention. Columns
    with zero standard deviation across the entire training set carry
    no information and are removed. Non-sensor columns (engine_id,
    cycle, op_*) are always preserved.

    Args:
        keep: Optional list of sensor column names to force-keep even
            if they appear constant in training data.
    """
```

- **`fit(X: pd.DataFrame) -> Self`** — Identifies sensor columns (`s_*` prefix) with zero standard deviation. Stores the list in `self.columns_to_drop_: list[str]`. Respects the `keep` parameter to force-keep specific sensors.
- **`transform(X: pd.DataFrame) -> pd.DataFrame`** — Returns a copy of X with `columns_to_drop_` removed.
- **Cross-subset behavior** — Different FD subsets may have different constant sensors. The transformer discovers them from training data, not from a hardcoded list.

### 2. `RollingFeatureExtractor` — multi-window rolling statistics

```python
class RollingFeatureExtractor(BaseEstimator, TransformerMixin):
    """Compute rolling statistics per engine for each sensor column.

    For each sensor and each window size, computes four rolling
    aggregations grouped by engine_id: mean, std, min, max.
    Uses min_periods=1 so early cycles get values instead of NaN.

    Args:
        windows: List of rolling window sizes in cycles.
            Default [5, 10, 20].
    """
```

- **`fit(X: pd.DataFrame) -> Self`** — Discovers sensor columns (`s_*` prefix) and stores them in `self.sensor_cols_: list[str]`. No statistics to learn — rolling features are computed per-engine, so fit is effectively a no-op beyond column discovery.
- **`transform(X: pd.DataFrame) -> pd.DataFrame`** — For each sensor and each window size, computes four rolling aggregations grouped by `engine_id`:
  - `{sensor}_rmean_{window}` — rolling mean
  - `{sensor}_rstd_{window}` — rolling standard deviation
  - `{sensor}_rmin_{window}` — rolling minimum
  - `{sensor}_rmax_{window}` — rolling maximum

  Uses `min_periods=1` so early cycles still get values. The original sensor columns are preserved alongside the new rolling columns.
- **Output width** — For N sensors and W window sizes, adds N x W x 4 new columns. With ~16 non-constant sensors and 3 windows, that's ~192 new features.

### 3. `OperationalNormalizer` — condition-aware z-score normalization

```python
class OperationalNormalizer(BaseEstimator, TransformerMixin):
    """Z-score normalize sensor readings by operational condition.

    Groups data by unique combinations of operational setting columns,
    then applies per-group z-score normalization. Falls back to global
    statistics for operating conditions seen at test time but not in
    training.

    Args:
        op_cols: Operational setting column names.
            Default ["op_1", "op_2", "op_3"].
    """
```

- **`fit(X: pd.DataFrame) -> Self`** — Groups training data by unique combinations of `op_cols`. Computes per-group mean and std for every numeric column except engine_id, cycle, and op_cols. Stores in `self.group_stats_: pd.DataFrame`. Also stores global mean/std in `self.global_stats_: pd.Series` as fallback for unseen conditions.
- **`transform(X: pd.DataFrame) -> pd.DataFrame`** — Applies z-score normalization: `(value - group_mean) / group_std` for each row based on its operating condition. Falls back to `global_stats_` for unseen conditions. Replaces zero-std groups with 1.0 to avoid division by zero. Non-numeric columns and op_cols are passed through unchanged.
- **FD001 behavior** — FD001 has effectively one operating condition, so this collapses to a global z-score normalization. The pipeline remains uniform across all subsets.

### 4. `build_feature_pipeline()` — pipeline factory

```python
def build_feature_pipeline(
    windows: list[int] | None = None,
    op_cols: list[str] | None = None,
) -> Pipeline:
    """Build an unfitted feature engineering pipeline.

    Returns an sklearn Pipeline with three named steps:
    sensor_dropper, rolling_features, normalizer.

    Args:
        windows: Rolling window sizes. Default [5, 10, 20].
        op_cols: Operational setting columns.
            Default ["op_1", "op_2", "op_3"].

    Returns:
        Unfitted sklearn.pipeline.Pipeline.
    """
```

Returns an unfitted `sklearn.pipeline.Pipeline` with three named steps: `"sensor_dropper"`, `"rolling_features"`, `"normalizer"`. Caller does `pipeline.fit_transform(train_df)` and `pipeline.transform(test_df)`.

### 5. `pyproject.toml` changes

Add `scikit-learn` to the main `[project] dependencies` list (not dev-only, since it's needed at runtime):

```toml
dependencies = [
    "numpy",
    "pandas>=2.0",
    "pydantic>=2.0",
    "pyyaml",
    "scikit-learn",
]
```

---

## Testing Strategy

All tests use synthetic DataFrames — no dependency on real downloaded data.

**`tests/features/test_sensor_dropper.py`**
- fit on data with one constant sensor, verify it's dropped on transform
- verify non-constant sensors and non-sensor columns survive
- verify fit on train and transform on test works correctly (leakage prevention)
- verify `keep` parameter prevents dropping a constant sensor
- verify empty drop list when all sensors vary

**`tests/features/test_rolling.py`**
- verify correct number of new columns (N sensors x W windows x 4 stats)
- verify rolling mean values on known data (e.g., constant series -> rolling mean equals constant)
- verify groupby engine_id (features don't bleed across engines)
- verify min_periods=1 fills early cycles with values (no NaNs)
- verify column naming convention (`{sensor}_rmean_{window}`)

**`tests/features/test_normalizer.py`**
- verify z-score produces ~0 mean / ~1 std after fit_transform
- verify multi-condition normalization (two op conditions with different sensor ranges produce condition-specific normalization)
- verify unseen condition at test time falls back to global stats
- verify zero-std handling (no division-by-zero errors, no NaNs)
- verify non-sensor columns pass through unchanged

**`tests/features/test_pipeline.py`**
- end-to-end: fit_transform on train, transform on test, verify shapes and no NaNs
- verify pipeline serialization with joblib round-trip (pickle/unpickle)
- verify pipeline works with FD001-style data (single op condition)
- verify pipeline works with multi-condition data (simulated FD002-style)

---

## What This Sub-project Does NOT Include

- Train/test splitting — deferred to Sub-projects 4 and 5
- Sequence windowing for LSTM/transformer input — Sub-project 5
- Any model training or evaluation — Sub-projects 4 and 5
- Feature selection (e.g., mutual information, L1 regularization) — deferred
- Feature store or caching layer — not needed now

---

## Sub-project Sequence

| # | Sub-project | Depends on |
|---|-------------|------------|
| 1 | Foundation | -- |
| 2 | EDA | 1 |
| 3 | **Feature Engineering** <- you are here | 1 |
| 4 | Baseline Models | 1, 3 |
| 5 | Sequence Models | 1, 3 |
| 6 | Inference & Deployment | 1, 4 or 5 |
