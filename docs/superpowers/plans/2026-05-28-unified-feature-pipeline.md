# Unified Feature Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the split Ridge/GRU preprocessing paths with a single shared sklearn Pipeline so both models receive the same clean, normalized, sensor-only feature matrix chosen by a config-driven `feature_set`.

**Architecture:** A 4-step sklearn Pipeline (`SensorDropper → OperatingModeNormalizer → SensorColumnSelector → FeatureEngineer`) is constructed by a rewritten `build_feature_pipeline`. Baseline Ridge is reduced to `build_feature_pipeline → Ridge`; GRU's manual normalizer is replaced by the same factory. `FeatureConfig` gains `feature_set`, `windows`, and `lag_steps`; `ModelConfig` loses them.

**Tech Stack:** Python 3.11+, scikit-learn, pandas, pydantic v2, pytest, mypy (strict), ruff.

---

## File Map

**New files:**
- `src/turbofan/features/engineering.py` — `FeatureEngineer` sklearn transformer
- `tests/features/test_engineering.py` — tests for `FeatureEngineer`

**Modified files:**
- `src/turbofan/config/schema.py` — add `feature_set`/`windows`/`lag_steps` to `FeatureConfig`; remove `feature_set`/`windows` from `ModelConfig`
- `src/turbofan/features/pipeline.py` — add `SensorColumnSelector`; rewrite `build_feature_pipeline` (4-step, new signature)
- `src/turbofan/models/baseline.py` — simplify to 2-step pipeline; delete all dead helpers
- `src/turbofan/cli/train_baseline.py` — read `feature_set`/`windows`/`lag_steps` from `cfg.features`
- `src/turbofan/cli/train_sequence_gru.py` — replace manual `OperatingModeNormalizer` with `build_feature_pipeline`
- `src/turbofan/experiments/sequence_gru_sweep.py` — same substitution as GRU CLI
- `src/turbofan/sequences/normalize.py` — delete `default_feature_cols`; keep `SequenceNormalizer` until Task 6
- `src/turbofan/models/test_evaluation.py` — change `evaluate_test_from_df` normalizer type to `OperatingModeNormalizer`
- `tests/features/test_pipeline.py` — rewrite for new 4-step pipeline
- `tests/models/test_baseline.py` — rewrite for 2-step pipeline
- `tests/models/test_train_baseline_cli.py` — update YAML configs; update named-step assertions
- `tests/sequences/test_normalize.py` — remove `default_feature_cols` tests
- `tests/models/test_train_sequence_gru_cli.py` — swap `default_feature_cols`/`OperatingModeNormalizer` monkeypatches for `build_feature_pipeline`
- `tests/models/test_sweep_sequence_gru.py` — same
- `tests/models/test_test_evaluation.py` — replace `SequenceNormalizer` with `OperatingModeNormalizer`

**Deleted files:**
- `src/turbofan/features/rolling.py`
- `tests/features/test_rolling.py`

---

## Task 1: FeatureEngineer transformer

The `FeatureEngineer` is a new standalone sklearn transformer. It accepts a DataFrame of normalized sensor columns (plus `engine_id` for groupby) and returns only the engineered feature columns. This task is fully isolated — no existing code changes.

**Files:**
- Create: `src/turbofan/features/engineering.py`
- Create: `tests/features/test_engineering.py`

- [ ] **Step 1: Write failing tests**

Create `tests/features/test_engineering.py`:

```python
"""Tests for turbofan.features.engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.features.engineering import FeatureEngineer


def _sensor_df(n_engines: int = 2, n_cycles: int = 10) -> pd.DataFrame:
    """Sensor-only DataFrame with engine_id for groupby testing."""
    rng = np.random.default_rng(0)
    rows = []
    for eid in range(1, n_engines + 1):
        for cyc in range(1, n_cycles + 1):
            rows.append({
                "engine_id": eid,
                "s_1": float(cyc) + rng.normal(0, 0.1),
                "s_2": 50.0 + rng.normal(0, 0.5),
                "s_3": float(eid) * 10 + rng.normal(0, 0.2),
            })
    return pd.DataFrame(rows)


def test_raw_returns_sensor_columns_only() -> None:
    """feature_set=raw returns only s_* columns with no engine_id."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="raw")
    result = eng.fit_transform(df)
    assert list(result.columns) == ["s_1", "s_2", "s_3"]
    assert "engine_id" not in result.columns


def test_raw_feature_cols_attribute() -> None:
    """feature_cols_ matches the output column list for raw."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="raw")
    eng.fit(df)
    assert eng.feature_cols_ == ["s_1", "s_2", "s_3"]


def test_rolling_mean_columns() -> None:
    """rolling_mean produces {sensor}_rmean_{window} columns."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_mean", windows=[5, 10])
    result = eng.fit_transform(df)
    expected = [
        "s_1_rmean_5", "s_2_rmean_5", "s_3_rmean_5",
        "s_1_rmean_10", "s_2_rmean_10", "s_3_rmean_10",
    ]
    assert list(result.columns) == expected


def test_rolling_stats_columns() -> None:
    """rolling_stats produces mean/std/min/max columns per sensor per window."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_stats", windows=[5])
    result = eng.fit_transform(df)
    for stat in ["rmean", "rstd", "rmin", "rmax"]:
        for sensor in ["s_1", "s_2", "s_3"]:
            assert f"{sensor}_{stat}_5" in result.columns
    assert "engine_id" not in result.columns


def test_raw_plus_rolling_mean_columns() -> None:
    """raw_plus_rolling_mean produces raw + rolling_mean columns."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="raw_plus_rolling_mean", windows=[5])
    result = eng.fit_transform(df)
    assert "s_1" in result.columns
    assert "s_1_rmean_5" in result.columns
    assert "engine_id" not in result.columns


def test_raw_plus_rolling_stats_columns() -> None:
    """raw_plus_rolling_stats produces raw + all rolling stat columns."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="raw_plus_rolling_stats", windows=[5])
    result = eng.fit_transform(df)
    assert "s_1" in result.columns
    assert "s_1_rmean_5" in result.columns
    assert "s_1_rstd_5" in result.columns
    assert "engine_id" not in result.columns


def test_lag_columns() -> None:
    """lag produces {sensor}_lag_{step} columns."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1, 2])
    result = eng.fit_transform(df)
    for step in [1, 2]:
        for sensor in ["s_1", "s_2", "s_3"]:
            assert f"{sensor}_lag_{step}" in result.columns
    assert "engine_id" not in result.columns


def test_rolling_no_nan_min_periods() -> None:
    """Rolling features have no NaN due to min_periods=1."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_stats", windows=[10])
    result = eng.fit_transform(df)
    assert not result.isna().any().any()


def test_rolling_respects_engine_boundaries() -> None:
    """Rolling mean for engine 2 cycle 1 is unaffected by engine 1 values."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 2, 2],
        "s_1": [100.0, 200.0, 5.0, 10.0],
    })
    eng = FeatureEngineer(feature_set="rolling_mean", windows=[5])
    result = eng.fit_transform(df)
    engine2_first_rmean = result.loc[df["engine_id"] == 2, "s_1_rmean_5"].iloc[0]
    assert engine2_first_rmean == pytest.approx(5.0)


def test_lag_backfills_within_engine() -> None:
    """Lag is backfilled for early cycles within each engine (no cross-engine boundary)."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2, 2],
        "s_1": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
    })
    eng = FeatureEngineer(feature_set="lag", lag_steps=[1])
    result = eng.fit_transform(df)
    # Engine 1 cycle 1 has no prior cycle: backfill from cycle 2 value
    e1_lag = result.loc[df["engine_id"] == 1, "s_1_lag_1"].tolist()
    assert e1_lag[0] == pytest.approx(e1_lag[1])  # backfilled to cycle 2's lag value
    # Engine 2 cycle 1 should not be affected by engine 1
    e2_lag = result.loc[df["engine_id"] == 2, "s_1_lag_1"].tolist()
    assert e2_lag[0] == pytest.approx(100.0)  # backfilled from engine 2's own data


def test_unsupported_feature_set_raises() -> None:
    """Unknown feature_set raises ValueError on fit."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="fft")
    with pytest.raises(ValueError, match="Unsupported feature_set"):
        eng.fit(df)


def test_feature_cols_attribute_rolling_stats() -> None:
    """feature_cols_ is set correctly on fit for rolling_stats."""
    df = _sensor_df()
    eng = FeatureEngineer(feature_set="rolling_stats", windows=[5])
    eng.fit(df)
    assert "s_1_rmean_5" in eng.feature_cols_
    assert "s_1_rstd_5" in eng.feature_cols_
    assert len(eng.feature_cols_) == 3 * 4 * 1  # 3 sensors * 4 stats * 1 window


def test_no_engine_id_in_output() -> None:
    """engine_id is never present in transform output."""
    df = _sensor_df()
    for fs in ["raw", "rolling_mean", "rolling_stats", "lag"]:
        kwargs = {"windows": [3]} if "rolling" in fs else {"lag_steps": [1]}
        eng = FeatureEngineer(feature_set=fs, **kwargs)  # type: ignore[arg-type]
        result = eng.fit_transform(df)
        assert "engine_id" not in result.columns, f"engine_id in output for {fs}"
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
pytest tests/features/test_engineering.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'FeatureEngineer'`

- [ ] **Step 3: Implement `src/turbofan/features/engineering.py`**

```python
"""Config-driven feature engineering transformer."""
from __future__ import annotations

from typing import Literal, Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FeatureSet = Literal[
    "raw",
    "rolling_mean",
    "rolling_stats",
    "raw_plus_rolling_mean",
    "raw_plus_rolling_stats",
    "lag",
]

_VALID_FEATURE_SETS: frozenset[str] = frozenset(
    ["raw", "rolling_mean", "rolling_stats",
     "raw_plus_rolling_mean", "raw_plus_rolling_stats", "lag"]
)


class FeatureEngineer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Apply a config-driven feature transformation to normalized sensor columns.

    Expects a DataFrame whose columns are ``s_*`` sensor columns (already
    normalized) and optionally ``engine_id`` for per-engine grouping.
    Returns only the engineered feature columns; ``engine_id`` is dropped.

    Args:
        feature_set: Which feature family to compute.
        windows: Rolling window sizes in cycles. Required for rolling feature sets.
            Default ``[10]``.
        lag_steps: Lag offsets in cycles. Required for ``lag`` feature set.
            Default ``[1]``.
    """

    def __init__(
        self,
        feature_set: str = "raw",
        windows: list[int] | None = None,
        lag_steps: list[int] | None = None,
    ) -> None:
        self.feature_set = feature_set
        self.windows = windows
        self.lag_steps = lag_steps

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Record input sensor columns and compute output column names.

        Args:
            X: Normalized sensor DataFrame (s_* columns, optionally engine_id).
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted transformer.

        Raises:
            ValueError: If ``feature_set`` is unsupported.
        """
        if self.feature_set not in _VALID_FEATURE_SETS:
            raise ValueError(
                f"Unsupported feature_set: {self.feature_set!r}. "
                f"Valid values: {sorted(_VALID_FEATURE_SETS)}"
            )
        self.sensor_cols_: list[str] = [
            c for c in X.columns if c.startswith("s_")
        ]
        self._windows: list[int] = self.windows if self.windows is not None else [10]
        self._lag_steps: list[int] = self.lag_steps if self.lag_steps is not None else [1]
        self.feature_cols_: list[str] = self._compute_output_cols()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering and return only the feature columns.

        Args:
            X: Sensor DataFrame. Must contain ``sensor_cols_`` from fit; may
                contain ``engine_id`` for per-engine grouping.

        Returns:
            DataFrame with exactly ``feature_cols_`` columns.
        """
        if self.feature_set == "raw":
            return X[self.sensor_cols_].copy()
        if self.feature_set == "rolling_mean":
            return self._rolling_mean(X)
        if self.feature_set == "rolling_stats":
            return self._rolling_stats(X)
        if self.feature_set == "raw_plus_rolling_mean":
            raw = X[self.sensor_cols_].copy()
            return pd.concat([raw, self._rolling_mean(X)], axis=1)
        if self.feature_set == "raw_plus_rolling_stats":
            raw = X[self.sensor_cols_].copy()
            return pd.concat([raw, self._rolling_stats(X)], axis=1)
        # lag
        return self._lag_features(X)

    def _compute_output_cols(self) -> list[str]:
        """Return the list of output column names for the configured feature set."""
        if self.feature_set == "raw":
            return list(self.sensor_cols_)
        if self.feature_set == "rolling_mean":
            return [
                f"{col}_rmean_{w}"
                for w in self._windows
                for col in self.sensor_cols_
            ]
        if self.feature_set == "rolling_stats":
            return [
                f"{col}_{stat}_{w}"
                for w in self._windows
                for col in self.sensor_cols_
                for stat in ["rmean", "rstd", "rmin", "rmax"]
            ]
        if self.feature_set == "raw_plus_rolling_mean":
            return list(self.sensor_cols_) + [
                f"{col}_rmean_{w}"
                for w in self._windows
                for col in self.sensor_cols_
            ]
        if self.feature_set == "raw_plus_rolling_stats":
            return list(self.sensor_cols_) + [
                f"{col}_{stat}_{w}"
                for w in self._windows
                for col in self.sensor_cols_
                for stat in ["rmean", "rstd", "rmin", "rmax"]
            ]
        # lag
        return [
            f"{col}_lag_{step}"
            for step in self._lag_steps
            for col in self.sensor_cols_
        ]

    def _rolling_mean(self, X: pd.DataFrame) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {}
        for w in self._windows:
            for col in self.sensor_cols_:
                grp = X.groupby("engine_id")[col]
                cols[f"{col}_rmean_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).mean()
                )
        return pd.DataFrame(cols, index=X.index)

    def _rolling_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {}
        for w in self._windows:
            for col in self.sensor_cols_:
                grp = X.groupby("engine_id")[col]
                cols[f"{col}_rmean_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).mean()
                )
                cols[f"{col}_rstd_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).std()
                ).fillna(0.0)
                cols[f"{col}_rmin_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).min()
                )
                cols[f"{col}_rmax_{w}"] = grp.transform(
                    lambda s, _w=w: s.rolling(_w, min_periods=1).max()
                )
        return pd.DataFrame(cols, index=X.index)

    def _lag_features(self, X: pd.DataFrame) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {}
        for step in self._lag_steps:
            for col in self.sensor_cols_:
                def _lag_and_backfill(s: pd.Series, _step: int = step) -> pd.Series:
                    return s.shift(_step).bfill()
                cols[f"{col}_lag_{step}"] = X.groupby("engine_id")[col].transform(
                    _lag_and_backfill
                )
        return pd.DataFrame(cols, index=X.index)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
pytest tests/features/test_engineering.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Lint and type-check**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
ruff check src/turbofan/features/engineering.py tests/features/test_engineering.py
mypy src/turbofan/features/engineering.py
```

Fix any issues before committing.

- [ ] **Step 6: Commit**

```powershell
git add src/turbofan/features/engineering.py tests/features/test_engineering.py
git commit -m "feat(features): add FeatureEngineer transformer for config-driven feature sets"
```

---

## Task 2: Schema changes (FeatureConfig + ModelConfig)

Add `feature_set`, `windows`, `lag_steps` to `FeatureConfig`. Remove `feature_set`, `windows` from `ModelConfig`. Also add `feature_set` to the Literal type in `FeatureConfig`.

**Files:**
- Modify: `src/turbofan/config/schema.py`

**Important:** Pydantic v2 ignores extra fields in YAML by default (no `extra="forbid"`), so existing YAML configs with `model.feature_set` or `model.windows` will silently skip those fields after this change — no subprocess test breaks. The only code that breaks is any Python that accesses `cfg.model.feature_set` or `cfg.model.windows` as attributes — that is fixed in Task 3.

- [ ] **Step 1: Update `src/turbofan/config/schema.py`**

Replace the `FeatureConfig` class:

```python
class FeatureConfig(BaseModel):
    """Configuration for feature engineering.

    Args:
        sensor_cols_to_drop: Sensor column names to remove before modeling.
            Determined from EDA; applied without recomputation at fit time.
        n_modes: Number of operating-mode clusters for normalization.
        feature_set: Which engineered feature family both models receive.
        windows: Rolling window sizes in cycles. Used by rolling feature sets.
        lag_steps: Lag offsets in cycles. Used by the lag feature set.
    """

    sensor_cols_to_drop: list[str] = Field(default_factory=list)
    n_modes: int = Field(default=1, gt=0)
    feature_set: Literal[
        "raw",
        "rolling_mean",
        "rolling_stats",
        "raw_plus_rolling_mean",
        "raw_plus_rolling_stats",
        "lag",
    ] = "raw"
    windows: list[PositiveWindow] = Field(default_factory=lambda: [10])
    lag_steps: list[PositiveWindow] = Field(default_factory=lambda: [1])
```

Replace the `ModelConfig` class (remove `feature_set` and `windows`):

```python
class ModelConfig(BaseModel):
    """Configuration for baseline model training.

    Args:
        name: Baseline model identifier.
        alpha: Ridge regularization strength.
        artifact_dir: Directory for local run artifacts.
    """

    name: Literal["ridge"] = "ridge"
    alpha: float = Field(default=100.0, gt=0.0)
    artifact_dir: Path = Path("artifacts/models")
```

Also add `"raw_plus_rolling_mean" | "raw_plus_rolling_stats" | "lag"` to the `Literal` import used — confirm `from typing import Annotated, Literal` is already imported (it is).

- [ ] **Step 2: Run full test suite to verify nothing broke**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
pytest -x -q 2>&1 | tail -30
```

Expected: still mostly PASS (some tests referencing `cfg.model.feature_set` via Python attribute access may error if accessed — those are caught in Task 3). If any tests fail, investigate before committing.

- [ ] **Step 3: Lint and type-check**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
ruff check src/turbofan/config/schema.py
mypy src/turbofan/config/schema.py
```

- [ ] **Step 4: Commit**

```powershell
git add src/turbofan/config/schema.py
git commit -m "feat(config): move feature_set/windows/lag_steps to FeatureConfig; remove from ModelConfig"
```

---

## Task 3: Rewrite `build_feature_pipeline` and simplify `baseline.py`

This is the largest single task. The old 3-step pipeline (SensorDropper → RollingFeatureExtractor → OperatingModeNormalizer) is replaced with 4 steps (SensorDropper → OperatingModeNormalizer → SensorColumnSelector → FeatureEngineer). `build_baseline_pipeline` simplifies from 7 steps to 2. All affected tests are updated in the same commit so the suite stays green.

**Files:**
- Modify: `src/turbofan/features/pipeline.py` (rewrite)
- Modify: `src/turbofan/models/baseline.py` (simplify)
- Modify: `tests/features/test_pipeline.py` (rewrite)
- Modify: `tests/models/test_baseline.py` (rewrite)
- Modify: `tests/models/test_train_baseline_cli.py` (update named-step assertions + YAML configs)

- [ ] **Step 1: Rewrite `tests/features/test_pipeline.py`**

Replace the entire file content with:

```python
"""Tests for turbofan.features.pipeline."""
from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd

from turbofan.features.pipeline import SensorColumnSelector, build_feature_pipeline
from turbofan.preprocessing.normalization import OperatingModeNormalizer


def _make_train_df() -> pd.DataFrame:
    """Training data: 2 engines, 15 cycles, 3 sensors."""
    rng = np.random.default_rng(42)
    rows = []
    for eid in [1, 2]:
        for cyc in range(1, 16):
            rows.append({
                "engine_id": eid,
                "cycle": cyc,
                "op_1": 0.0,
                "op_2": 0.0,
                "op_3": 0.0,
                "s_1": float(cyc) + rng.normal(0, 0.5),
                "s_2": 200.0,
                "s_3": 80.0 + rng.normal(0, 1.0),
            })
    return pd.DataFrame(rows)


def _make_test_df() -> pd.DataFrame:
    """Test data: 1 engine, 10 cycles."""
    rng = np.random.default_rng(99)
    rows = []
    for cyc in range(1, 11):
        rows.append({
            "engine_id": 3,
            "cycle": cyc,
            "op_1": 0.0,
            "op_2": 0.0,
            "op_3": 0.0,
            "s_1": float(cyc) + rng.normal(0, 0.5),
            "s_2": 200.0,
            "s_3": 80.0 + rng.normal(0, 1.0),
        })
    return pd.DataFrame(rows)


def test_pipeline_has_four_named_steps() -> None:
    """Pipeline has the four expected named steps."""
    pipe = build_feature_pipeline()
    assert list(pipe.named_steps) == [
        "sensor_dropper",
        "normalizer",
        "sensor_selector",
        "feature_engineer",
    ]


def test_output_contains_only_sensor_columns_for_raw() -> None:
    """feature_set=raw output contains only s_* columns."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="raw")
    result = pipe.fit_transform(train)
    assert all(c.startswith("s_") for c in result.columns)


def test_op_cols_absent_from_output() -> None:
    """op_1, op_2, op_3 are not in the pipeline output."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="raw")
    result = pipe.fit_transform(train)
    assert "op_1" not in result.columns
    assert "op_2" not in result.columns
    assert "op_3" not in result.columns


def test_engine_id_absent_from_output() -> None:
    """engine_id is not in the pipeline output."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="raw")
    result = pipe.fit_transform(train)
    assert "engine_id" not in result.columns


def test_sensor_drop_removes_listed_sensor() -> None:
    """Sensor listed in sensor_drop is absent from output."""
    train = _make_train_df()
    pipe = build_feature_pipeline(sensor_drop=["s_2"], feature_set="raw")
    result = pipe.fit_transform(train)
    assert "s_2" not in result.columns


def test_fit_transform_no_nans_raw() -> None:
    """Pipeline output has no NaN values for feature_set=raw."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="raw")
    result = pipe.fit_transform(train)
    assert not result.isna().any().any()


def test_transform_test_no_nans() -> None:
    """Fit on train, transform test produces no NaNs."""
    train = _make_train_df()
    test = _make_test_df()
    pipe = build_feature_pipeline(feature_set="raw")
    pipe.fit(train)
    result = pipe.transform(test)
    assert not result.isna().any().any()


def test_rolling_mean_columns_in_output() -> None:
    """feature_set=rolling_mean produces rmean columns in output."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="rolling_mean", windows=[5])
    result = pipe.fit_transform(train)
    assert "s_1_rmean_5" in result.columns
    assert "engine_id" not in result.columns


def test_rolling_respects_engine_boundaries() -> None:
    """Rolling mean for engine 2 cycle 1 is not contaminated by engine 1."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 2, 2],
        "cycle": [1, 2, 1, 2],
        "op_1": [0.0, 0.0, 0.0, 0.0],
        "op_2": [0.0, 0.0, 0.0, 0.0],
        "op_3": [0.0, 0.0, 0.0, 0.0],
        "s_1": [100.0, 200.0, 5.0, 10.0],
    })
    pipe = build_feature_pipeline(feature_set="rolling_mean", windows=[5])
    result = pipe.fit_transform(df)
    # Engine 2 first cycle rmean should be based on its own value (normalized),
    # not contaminated by engine 1's values.
    e2_idx = df["engine_id"] == 2
    e1_idx = df["engine_id"] == 1
    # Just verify the output has no NaN (engine boundary didn't bleed NaN)
    assert not result.loc[e2_idx.values, :].isna().any().any()
    assert not result.loc[e1_idx.values, :].isna().any().any()


def test_lag_no_cross_engine_bleed() -> None:
    """Lag backfills within engine group, does not cross engine boundary."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 2, 2],
        "cycle": [1, 2, 1, 2],
        "op_1": [0.0] * 4,
        "op_2": [0.0] * 4,
        "op_3": [0.0] * 4,
        "s_1": [10.0, 20.0, 100.0, 200.0],
    })
    pipe = build_feature_pipeline(feature_set="lag", lag_steps=[1])
    result = pipe.fit_transform(df)
    assert not result.isna().any().any()


def test_normalizer_step_is_operating_mode_normalizer() -> None:
    """normalizer step is OperatingModeNormalizer."""
    pipe = build_feature_pipeline(n_modes=2, random_state=7)
    assert isinstance(pipe.named_steps["normalizer"], OperatingModeNormalizer)
    assert pipe.named_steps["normalizer"].n_modes == 2


def test_output_is_dataframe() -> None:
    """Pipeline returns a pandas DataFrame."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="raw")
    result = pipe.fit_transform(train)
    assert isinstance(result, pd.DataFrame)


def test_joblib_serialization() -> None:
    """Fitted pipeline survives joblib round-trip."""
    train = _make_train_df()
    pipe = build_feature_pipeline(feature_set="raw")
    pipe.fit(train)
    buffer = io.BytesIO()
    joblib.dump(pipe, buffer)
    buffer.seek(0)
    loaded = joblib.load(buffer)
    test = _make_test_df()
    pd.testing.assert_frame_equal(pipe.transform(test), loaded.transform(test))


def test_sensor_column_selector_fit_records_sensor_cols() -> None:
    """SensorColumnSelector.fit records s_* columns as feature_cols_."""
    df = pd.DataFrame({
        "engine_id": [1, 2],
        "op_1": [0.0, 0.0],
        "s_1": [1.0, 2.0],
        "s_2": [3.0, 4.0],
    })
    sel = SensorColumnSelector()
    sel.fit(df)
    assert sel.feature_cols_ == ["s_1", "s_2"]


def test_sensor_column_selector_transform_keeps_engine_id() -> None:
    """SensorColumnSelector.transform keeps engine_id for downstream grouping."""
    df = pd.DataFrame({
        "engine_id": [1, 2],
        "op_1": [0.0, 0.0],
        "s_1": [1.0, 2.0],
        "s_2": [3.0, 4.0],
    })
    sel = SensorColumnSelector()
    sel.fit(df)
    result = sel.transform(df)
    assert "engine_id" in result.columns
    assert "op_1" not in result.columns
    assert "s_1" in result.columns
    assert "s_2" in result.columns


def test_multi_condition_pipeline() -> None:
    """Pipeline handles data with multiple op conditions (n_modes=2)."""
    rng = np.random.default_rng(7)
    rows = []
    for eid, op in [(1, 1.0), (2, 1.0), (3, 2.0), (4, 2.0)]:
        for cyc in range(1, 11):
            rows.append({
                "engine_id": eid,
                "cycle": cyc,
                "op_1": op,
                "op_2": 0.0,
                "op_3": 0.0,
                "s_1": op * 100 + rng.normal(0, 1.0),
            })
    df = pd.DataFrame(rows)
    pipe = build_feature_pipeline(n_modes=2, feature_set="raw", random_state=7)
    result = pipe.fit_transform(df)
    assert not result.isna().any().any()
```

- [ ] **Step 2: Run pipeline tests to verify they fail**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
pytest tests/features/test_pipeline.py -v 2>&1 | tail -20
```

Expected: multiple failures (old API doesn't match new tests yet).

- [ ] **Step 3: Rewrite `src/turbofan/features/pipeline.py`**

```python
"""Feature engineering pipeline factory."""
from __future__ import annotations

from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from turbofan.features.engineering import FeatureEngineer
from turbofan.features.sensor_dropper import SensorDropper
from turbofan.preprocessing.normalization import OperatingModeNormalizer


class SensorColumnSelector(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Select normalized sensor columns and keep engine_id for downstream grouping.

    ``fit`` records which columns start with ``s_``. ``transform`` returns
    those columns plus ``engine_id`` (when present) so that ``FeatureEngineer``
    can compute per-engine rolling and lag features.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Record sensor column names from training data.

        Args:
            X: DataFrame after normalization.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted selector.
        """
        self.feature_cols_: list[str] = [
            c for c in X.columns if c.startswith("s_")
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return sensor columns plus engine_id for downstream grouping.

        Args:
            X: Normalized DataFrame.

        Returns:
            DataFrame with ``s_*`` columns and ``engine_id`` (if present).
        """
        keep = self.feature_cols_ + (
            ["engine_id"] if "engine_id" in X.columns else []
        )
        return X[keep]


def build_feature_pipeline(
    op_cols: list[str] | None = None,
    sensor_drop: list[str] | None = None,
    n_modes: int = 1,
    random_state: int = 42,
    feature_set: str = "raw",
    windows: list[int] | None = None,
    lag_steps: list[int] | None = None,
) -> Pipeline:
    """Build the shared 4-step feature engineering pipeline.

    Steps: ``sensor_dropper`` → ``normalizer`` → ``sensor_selector``
    → ``feature_engineer``.

    The normalizer receives explicit ``feature_cols`` (kept sensor columns
    only) so op cols are available for KMeans but are not z-scored.
    ``SensorColumnSelector`` retains ``engine_id`` so that
    ``FeatureEngineer`` can compute per-engine rolling and lag features.
    The final output contains only the engineered feature columns.

    Args:
        op_cols: Operating-condition columns for KMeans clustering.
            Defaults to ``["op_1", "op_2", "op_3"]``.
        sensor_drop: Sensor column names to exclude before normalization.
            Determined from EDA; passed to ``SensorDropper``.
        n_modes: Number of operating-mode KMeans clusters.
        random_state: KMeans random seed.
        feature_set: Which engineered feature family to produce.
        windows: Rolling window sizes. Forwarded to ``FeatureEngineer``.
        lag_steps: Lag offsets. Forwarded to ``FeatureEngineer``.

    Returns:
        Unfitted sklearn Pipeline.
    """
    kept_sensors = [
        f"s_{i}"
        for i in range(1, 22)
        if f"s_{i}" not in (sensor_drop or [])
    ]
    return Pipeline(
        [
            ("sensor_dropper", SensorDropper(drop=sensor_drop)),
            (
                "normalizer",
                OperatingModeNormalizer(
                    feature_cols=kept_sensors,
                    op_cols=op_cols,
                    n_modes=n_modes,
                    random_state=random_state,
                ),
            ),
            ("sensor_selector", SensorColumnSelector()),
            (
                "feature_engineer",
                FeatureEngineer(
                    feature_set=feature_set,
                    windows=windows,
                    lag_steps=lag_steps,
                ),
            ),
        ]
    )
```

- [ ] **Step 4: Run pipeline tests to verify they pass**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
pytest tests/features/test_pipeline.py -v
```

Expected: all PASS.

- [ ] **Step 5: Update `tests/models/test_baseline.py`**

Replace the entire file:

```python
"""Tests for turbofan.models.baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from turbofan.models.baseline import build_baseline_pipeline


def _make_df() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic turbofan rows and RUL labels."""
    rng = np.random.default_rng(42)
    rows = []
    labels = []
    for engine_id in [1, 2, 3]:
        for cycle in range(1, 8):
            rows.append({
                "engine_id": engine_id,
                "cycle": cycle,
                "op_1": 0.0,
                "op_2": 0.0,
                "op_3": 0.0,
                "s_1": float(cycle) + rng.normal(0.0, 0.1),
                "s_2": 200.0,
                "s_3": float(engine_id) + rng.normal(0.0, 0.1),
            })
            labels.append(float(8 - cycle))
    return pd.DataFrame(rows), pd.Series(labels, name="rul")


def test_build_baseline_pipeline_returns_two_step_pipeline() -> None:
    """Baseline pipeline has exactly two steps: features and model."""
    pipe = build_baseline_pipeline()
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["features", "model"]


def test_default_model_is_ridge() -> None:
    """The default baseline estimator is Ridge."""
    pipe = build_baseline_pipeline()
    assert isinstance(pipe.named_steps["model"], Ridge)


def test_configures_ridge_alpha() -> None:
    """Ridge alpha is passed into the estimator."""
    pipe = build_baseline_pipeline(alpha=2.5)
    assert pipe.named_steps["model"].alpha == 2.5


def test_default_ridge_alpha() -> None:
    """Default Ridge alpha is 100.0."""
    assert build_baseline_pipeline().named_steps["model"].alpha == 100.0


def test_features_step_is_pipeline() -> None:
    """The features step is itself a Pipeline with expected steps."""
    pipe = build_baseline_pipeline()
    features = pipe.named_steps["features"]
    assert isinstance(features, Pipeline)
    assert list(features.named_steps) == [
        "sensor_dropper",
        "normalizer",
        "sensor_selector",
        "feature_engineer",
    ]


def test_configures_sensor_drop() -> None:
    """sensor_drop is forwarded into the feature pipeline's sensor_dropper."""
    pipe = build_baseline_pipeline(sensor_drop=["s_1", "s_5"])
    dropper = pipe.named_steps["features"].named_steps["sensor_dropper"]
    assert dropper.drop == ["s_1", "s_5"]


def test_pipeline_can_fit_and_predict() -> None:
    """Synthetic data can be fit and predicted without NaNs."""
    X, y = _make_df()
    pipe = build_baseline_pipeline(feature_set="raw")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert not np.isnan(preds).any()


def test_rolling_mean_feature_set() -> None:
    """feature_set=rolling_mean produces rolling mean columns for Ridge."""
    X, y = _make_df()
    pipe = build_baseline_pipeline(feature_set="rolling_mean", windows=[3])
    pipe.fit(X, y)
    assert any("_rmean_" in c for c in pipe.named_steps["model"].feature_names_in_)


def test_model_receives_dataframe_feature_names() -> None:
    """Ridge keeps sklearn feature_names_in_ metadata."""
    X, y = _make_df()
    pipe = build_baseline_pipeline(feature_set="raw")
    pipe.fit(X, y)
    assert hasattr(pipe.named_steps["model"], "feature_names_in_")
    assert "engine_id" not in set(pipe.named_steps["model"].feature_names_in_)
    assert "cycle" not in set(pipe.named_steps["model"].feature_names_in_)
    assert "op_1" not in set(pipe.named_steps["model"].feature_names_in_)


def test_unknown_model_name_raises() -> None:
    """Unsupported model names fail fast."""
    with pytest.raises(ValueError, match="Unsupported model"):
        build_baseline_pipeline(model_name="random_forest")  # type: ignore[arg-type]
```

- [ ] **Step 6: Simplify `src/turbofan/models/baseline.py`**

Replace the entire file:

```python
"""Baseline sklearn pipeline factory."""
from __future__ import annotations

from typing import Literal

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from turbofan.features.pipeline import build_feature_pipeline


def build_baseline_pipeline(
    model_name: Literal["ridge"] = "ridge",
    alpha: float = 100.0,
    op_cols: list[str] | None = None,
    sensor_drop: list[str] | None = None,
    n_modes: int = 1,
    random_state: int = 42,
    feature_set: str = "raw",
    windows: list[int] | None = None,
    lag_steps: list[int] | None = None,
) -> Pipeline:
    """Build an unfitted feature-plus-regressor sklearn Pipeline.

    The pipeline has two steps: ``features`` (the shared feature pipeline)
    and ``model`` (a Ridge regressor). All feature engineering is handled
    by ``build_feature_pipeline``.

    Args:
        model_name: Baseline model identifier. Only ``"ridge"`` is supported.
        alpha: Ridge regularization strength.
        op_cols: Operating-condition columns for KMeans clustering.
        sensor_drop: Sensor column names to remove before feature engineering.
        n_modes: Operating-mode count for ``OperatingModeNormalizer``.
        random_state: KMeans random seed for the normalizer.
        feature_set: Which engineered feature family to expose to Ridge.
        windows: Rolling window sizes. Forwarded to ``FeatureEngineer``.
        lag_steps: Lag offsets. Forwarded to ``FeatureEngineer``.

    Returns:
        Unfitted sklearn Pipeline with feature engineering and Ridge.

    Raises:
        ValueError: If ``model_name`` is unsupported.
    """
    if model_name != "ridge":
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(
        [
            (
                "features",
                build_feature_pipeline(
                    op_cols=op_cols,
                    sensor_drop=sensor_drop,
                    n_modes=n_modes,
                    random_state=random_state,
                    feature_set=feature_set,
                    windows=windows,
                    lag_steps=lag_steps,
                ),
            ),
            ("model", Ridge(alpha=alpha)),
        ]
    )
```

- [ ] **Step 7: Run baseline tests to verify they pass**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
pytest tests/features/test_pipeline.py tests/models/test_baseline.py -v
```

Expected: all PASS.

- [ ] **Step 8: Update `tests/models/test_train_baseline_cli.py`**

The integration test that checks named steps and rolling/selector assertions needs updating. Find and replace the block that checks `rolling_features`, `select_model_features`, and `rolling.windows`:

```python
# OLD (lines ~140-145 in test_train_baseline_cli_writes_artifacts):
dropper = estimator.named_steps["features"].named_steps["sensor_dropper"]
assert dropper.drop == ["s_2"]
rolling = estimator.named_steps["features"].named_steps["rolling_features"]
selector = estimator.named_steps["select_model_features"]
assert rolling.windows == [5]
assert selector.feature_set == "raw_plus_rolling"

# NEW:
dropper = estimator.named_steps["features"].named_steps["sensor_dropper"]
assert dropper.drop == ["s_2"]
feature_engineer = estimator.named_steps["features"].named_steps["feature_engineer"]
assert feature_engineer.feature_set == "raw_plus_rolling_mean"
assert feature_engineer.windows == [5]
```

Also update the YAML config that uses `model.feature_set: raw_plus_rolling` and `model.windows: [5]`. Those fields now live under `features:` and the feature set name is `raw_plus_rolling_mean`:

```yaml
# OLD (inside _write_config or cfg_path.write_text in that test):
model:
  name: ridge
  alpha: 1.0
  feature_set: raw_plus_rolling
  windows:
    - 5
  artifact_dir: ...
features:
  sensor_cols_to_drop:
    - s_2

# NEW:
model:
  name: ridge
  alpha: 1.0
  artifact_dir: ...
features:
  sensor_cols_to_drop:
    - s_2
  feature_set: raw_plus_rolling_mean
  windows:
    - 5
```

Read the full test file to find all occurrences that need updating:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
python -m pytest tests/models/test_train_baseline_cli.py -v 2>&1 | tail -20
```

Fix all failing assertions. The key assertions to update are all references to `rolling_features`, `select_model_features`, `rolling.windows`, and `selector.feature_set`.

- [ ] **Step 9: Run full test suite to confirm green**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
pytest -x -q 2>&1 | tail -20
```

Expected: all PASS (except maybe test_rolling.py which references the old rolling.py — that is deleted in Task 7).

- [ ] **Step 10: Lint and type-check**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
ruff check src/turbofan/features/pipeline.py src/turbofan/models/baseline.py
mypy src/turbofan/features/pipeline.py src/turbofan/models/baseline.py
```

- [ ] **Step 11: Commit**

```powershell
git add src/turbofan/features/pipeline.py src/turbofan/models/baseline.py tests/features/test_pipeline.py tests/models/test_baseline.py tests/models/test_train_baseline_cli.py
git commit -m "refactor(pipeline): rewrite build_feature_pipeline (4-step) and simplify baseline to 2-step"
```

---

## Task 4: Update `train_baseline.py` CLI

Wire the CLI to read `feature_set`, `windows`, `lag_steps` from `cfg.features` instead of the now-removed `cfg.model.feature_set`/`cfg.model.windows`.

**Files:**
- Modify: `src/turbofan/cli/train_baseline.py`

- [ ] **Step 1: Update `main()` in `src/turbofan/cli/train_baseline.py`**

Find the `build_baseline_pipeline` call in `main()`:

```python
# OLD:
estimator = build_baseline_pipeline(
    model_name=cfg.model.name,
    alpha=cfg.model.alpha,
    windows=cfg.model.windows,
    feature_set=cfg.model.feature_set,
    sensor_drop=cfg.features.sensor_cols_to_drop or None,
    n_modes=cfg.features.n_modes,
    random_state=cfg.data.random_seed,
)

# NEW:
estimator = build_baseline_pipeline(
    model_name=cfg.model.name,
    alpha=cfg.model.alpha,
    sensor_drop=cfg.features.sensor_cols_to_drop or None,
    n_modes=cfg.features.n_modes,
    random_state=cfg.data.random_seed,
    feature_set=cfg.features.feature_set,
    windows=cfg.features.windows,
    lag_steps=cfg.features.lag_steps,
)
```

- [ ] **Step 2: Run baseline CLI tests**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
pytest tests/models/test_train_baseline_cli.py -v
```

Expected: all PASS.

- [ ] **Step 3: Lint and type-check**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
ruff check src/turbofan/cli/train_baseline.py
mypy src/turbofan/cli/train_baseline.py
```

- [ ] **Step 4: Commit**

```powershell
git add src/turbofan/cli/train_baseline.py
git commit -m "feat(cli): wire train_baseline to read feature_set/windows/lag_steps from cfg.features"
```

---

## Task 5: Update `train_sequence_gru.py` CLI

Replace the manual `OperatingModeNormalizer` and `default_feature_cols()` with `build_feature_pipeline`. Update `_evaluate_official_test` to use the full pipeline. Update the checkpoint payload to extract the normalizer from the pipeline.

**Files:**
- Modify: `src/turbofan/cli/train_sequence_gru.py`
- Modify: `tests/models/test_train_sequence_gru_cli.py`

- [ ] **Step 1: Update `src/turbofan/cli/train_sequence_gru.py`**

**Replace imports** — remove `default_feature_cols`, `OperatingModeNormalizer` direct import; add `build_feature_pipeline`:

```python
# Remove these imports:
from turbofan.preprocessing.normalization import OperatingModeNormalizer
from turbofan.sequences.normalize import default_feature_cols

# Add this import (add to features imports section):
from turbofan.features.pipeline import build_feature_pipeline
from sklearn.pipeline import Pipeline
```

**Update `_evaluate_official_test` signature** — change `normalizer: OperatingModeNormalizer` to `pipeline: Pipeline`:

```python
def _evaluate_official_test(
    cfg: ProjectConfig,
    model: GRURULRegressor,
    pipeline: Pipeline,
    feature_cols: list[str],
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame] | None:
    """Evaluate final-cycle official test labels when files exist.

    Args:
        cfg: Project config.
        model: Trained GRU model.
        pipeline: Fitted shared feature pipeline.
        feature_cols: Feature columns used by the model.
        device: Torch device used for inference.

    Returns:
        Metrics and prediction rows, or None when official files are missing.
    """
    try:
        test_raw = load_raw_test(cfg.data)
        rul_labels = load_rul_labels(cfg.data)
    except FileNotFoundError:
        return None

    test_df = pipeline.transform(test_raw)
    test_windows = build_final_windows(
        test_df,
        feature_cols=feature_cols,
        window_size=cfg.sequence.window_size,
        target_col=None,
    )
    loader = build_sequence_loader(
        test_windows,
        batch_size=cfg.sequence.batch_size,
        shuffle=False,
    )
    y_pred = np.clip(
        predict_windows(model, loader, device, max_rul=cfg.data.max_rul), 0.0, None
    )
    y_true = align_labels_to_eligible_engines(
        test_windows.metadata,
        rul_labels,
    )
    metrics = regression_metrics(y_true, y_pred)
    predictions = _prediction_frame(test_windows, y_true, y_pred)
    return metrics, predictions
```

**Update `_model_payload`** — replace `normalizer: OperatingModeNormalizer` with `pipeline: Pipeline`:

```python
def _model_payload(
    model: GRURULRegressor,
    cfg: ProjectConfig,
    feature_cols: list[str],
    pipeline: Pipeline,
) -> dict[str, object]:
    """Build the serialized model checkpoint payload.

    Args:
        model: Trained GRU model.
        cfg: Project config.
        feature_cols: Feature columns used by the model.
        pipeline: Fitted shared feature pipeline.

    Returns:
        Torch-serializable model payload.
    """
    from turbofan.preprocessing.normalization import OperatingModeNormalizer
    normalizer = pipeline.named_steps["normalizer"]
    assert isinstance(normalizer, OperatingModeNormalizer)
    return {
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "sequence_config": cfg.sequence.model_dump(mode="json"),
        "normalizer_type": "operating_mode",
        "normalizer_payload": normalizer.to_payload(),
        "fd_subset": cfg.data.fd_subset,
        "random_seed": cfg.data.random_seed,
        "max_rul": cfg.data.max_rul,
    }
```

**Update `main()`** — replace the normalizer block and `default_feature_cols()` call:

```python
# OLD block in main():
feature_cols = default_feature_cols()
...
normalizer = OperatingModeNormalizer(
    feature_cols=feature_cols,
    n_modes=cfg.features.n_modes,
    random_state=cfg.data.random_seed,
)
train_normalized = normalizer.fit_transform(train_df)
val_normalized = normalizer.transform(val_df)

# NEW block in main() (after split_by_engine call):
pipeline = build_feature_pipeline(
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

**Update the `_evaluate_official_test` call** — pass `pipeline` not `normalizer`:

```python
# OLD:
official = _evaluate_official_test(
    cfg,
    result.model,
    normalizer,
    feature_cols,
    device,
)

# NEW:
official = _evaluate_official_test(
    cfg,
    result.model,
    pipeline,
    feature_cols,
    device,
)
```

**Update `_model_payload` call**:

```python
# OLD:
torch.save(
    _model_payload(result.model, cfg, feature_cols, normalizer),
    run_dir / "model.pt",
)

# NEW:
torch.save(
    _model_payload(result.model, cfg, feature_cols, pipeline),
    run_dir / "model.pt",
)
```

Also remove the `input_size=len(feature_cols)` issue — after the refactor, `feature_cols` is set from `pipeline.named_steps["feature_engineer"].feature_cols_` BEFORE the model is constructed. Verify the model construction line still correctly uses `len(feature_cols)`:

```python
model = GRURULRegressor(
    input_size=len(feature_cols),
    ...
)
```

- [ ] **Step 2: Update `tests/models/test_train_sequence_gru_cli.py`**

The monkeypatched unit tests patch `default_feature_cols` and `OperatingModeNormalizer`. After the refactor, they need to patch `build_feature_pipeline` instead. Add a `_FakePipeline` class and update monkeypatches.

**Add `_FakePipeline` class** near the top of the test file (after `_FakeNormalizer`):

```python
class _FakePipeline:
    """Minimal pipeline test double returning input unchanged."""

    def __init__(self, feature_cols: list[str] | None = None) -> None:
        self._feature_cols = feature_cols or ["s1", "s2"]
        _fake_norm = _FakeNormalizer(feature_cols=self._feature_cols)
        _fake_fe = type("FakeFeatureEngineer", (), {
            "feature_cols_": self._feature_cols,
        })()
        self.named_steps: dict[str, object] = {
            "normalizer": _fake_norm,
            "feature_engineer": _fake_fe,
        }

    def fit_transform(self, frame: object) -> object:
        """Return input unchanged.

        Args:
            frame: Input frame.

        Returns:
            Unchanged frame.
        """
        return frame

    def transform(self, frame: object) -> object:
        """Return input unchanged.

        Args:
            frame: Input frame.

        Returns:
            Unchanged frame.
        """
        return frame
```

**In `test_train_sequence_gru_cli_seeds_model_initialization`**, replace:
```python
monkeypatch.setattr(module, "default_feature_cols", lambda: ["s1", "s2"])
...
monkeypatch.setattr(module, "OperatingModeNormalizer", _FakeNormalizer)
```
with:
```python
monkeypatch.setattr(
    module, "build_feature_pipeline", lambda **kwargs: _FakePipeline(["s1", "s2"])
)
```

Also remove any `build_sliding_windows` and `build_sequence_loader` patches that still reference old variables. Verify the expected `input_size` in the seeding test matches the fake feature_cols length (2).

**In `test_train_sequence_gru_cli_appends_training_log_entry`**, same swap: replace `default_feature_cols` + `OperatingModeNormalizer` patches with `build_feature_pipeline`.

**In `test_train_sequence_gru_cli_uses_subset_derived_mode_count`**: this test captured kwargs passed to `OperatingModeNormalizer`. After the refactor, we instead capture `build_feature_pipeline` kwargs:

```python
captured: list[dict[str, object]] = []

def _capturing_build_feature_pipeline(**kwargs: object) -> _FakePipeline:
    captured.append(dict(kwargs))
    return _FakePipeline(["s1", "s2"])

monkeypatch.setattr(module, "build_feature_pipeline", _capturing_build_feature_pipeline)
```

Then assert:
```python
assert captured[0]["n_modes"] == 1  # FD001 → 1 mode (from cfg.features.n_modes)
assert captured[0]["random_state"] == 77
```

Remove the old `OperatingModeNormalizer` and `default_feature_cols` patches from that test.

- [ ] **Step 3: Run sequence GRU CLI tests**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
pytest tests/models/test_train_sequence_gru_cli.py -v
```

Expected: all PASS.

- [ ] **Step 4: Lint and type-check**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
ruff check src/turbofan/cli/train_sequence_gru.py
mypy src/turbofan/cli/train_sequence_gru.py
```

- [ ] **Step 5: Commit**

```powershell
git add src/turbofan/cli/train_sequence_gru.py tests/models/test_train_sequence_gru_cli.py
git commit -m "feat(gru): replace manual normalizer with build_feature_pipeline in train_sequence_gru"
```

---

## Task 6: Update `sequence_gru_sweep.py`

Same substitution as Task 5 but for the sweep script and its tests.

**Files:**
- Modify: `src/turbofan/experiments/sequence_gru_sweep.py`
- Modify: `tests/models/test_sweep_sequence_gru.py`

- [ ] **Step 1: Update `src/turbofan/experiments/sequence_gru_sweep.py`**

**Replace imports** — remove `default_feature_cols`/`OperatingModeNormalizer`; add `build_feature_pipeline`:

```python
# Remove:
from turbofan.preprocessing.normalization import OperatingModeNormalizer
from turbofan.sequences.normalize import default_feature_cols

# Add:
from turbofan.features.pipeline import build_feature_pipeline
from sklearn.pipeline import Pipeline
```

**In `run_gru_sweep()`**, replace the normalizer block and feature_cols:

```python
# OLD:
feature_cols = default_feature_cols()
...
normalizer = OperatingModeNormalizer(
    feature_cols=feature_cols,
    n_modes=cfg.features.n_modes,
    random_state=cfg.data.random_seed,
)
train_normalized = normalizer.fit_transform(train_df)
val_normalized = normalizer.transform(val_df)

# NEW (after split_by_engine call):
pipeline = build_feature_pipeline(
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

The rest of `run_gru_sweep` (building windows, training, metrics) stays the same since `feature_cols` is now derived from the pipeline.

- [ ] **Step 2: Update `tests/models/test_sweep_sequence_gru.py`**

The monkeypatched tests patch `default_feature_cols` and `OperatingModeNormalizer`. Replace with `build_feature_pipeline`:

**In `test_gru_sweep_reports_validation_window_metrics`**:

Replace `FakeNormalizer` + `monkeypatch.setattr(module, "default_feature_cols", lambda: ["s1"])` + `monkeypatch.setattr(module, "OperatingModeNormalizer", FakeNormalizer)` with:

```python
class FakePipeline:
    """Minimal pipeline double for sweep tests."""

    def __init__(self, **kwargs: object) -> None:
        self._feature_cols = ["s1"]
        self.named_steps = {
            "feature_engineer": type("FE", (), {"feature_cols_": ["s1"]})(),
        }

    def fit_transform(self, frame: object) -> str:
        """Return training frame sentinel.

        Args:
            frame: Training frame.

        Returns:
            Training frame sentinel.
        """
        del frame
        return "train_normalized"

    def transform(self, frame: object) -> str:
        """Return validation frame sentinel.

        Args:
            frame: Validation frame.

        Returns:
            Validation frame sentinel.
        """
        del frame
        return "validation_normalized"

monkeypatch.setattr(module, "build_feature_pipeline", FakePipeline)
```

Remove the old `OperatingModeNormalizer` and `default_feature_cols` patches. Adjust `fake_build_sliding_windows` if it relies on specific sentinel values.

**In `test_gru_sweep_appends_training_log_entry_per_completed_config`**: same pattern — replace `FakeNormalizer` + old patches with `FakePipeline`.

**In `test_gru_sweep_returns_expected_rows`** and `test_gru_sweep_validates_inputs`: these are integration tests that load real data. They should pass without changes if the sweep function is updated correctly.

- [ ] **Step 3: Run sweep tests**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
pytest tests/models/test_sweep_sequence_gru.py -v
```

Expected: all PASS.

- [ ] **Step 4: Lint and type-check**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
ruff check src/turbofan/experiments/sequence_gru_sweep.py
mypy src/turbofan/experiments/sequence_gru_sweep.py
```

- [ ] **Step 5: Commit**

```powershell
git add src/turbofan/experiments/sequence_gru_sweep.py tests/models/test_sweep_sequence_gru.py
git commit -m "feat(sweep): replace manual normalizer with build_feature_pipeline in sequence_gru_sweep"
```

---

## Task 7: Update `test_evaluation.py` — type annotation fix

`evaluate_test_from_df` currently takes `normalizer: SequenceNormalizer`. Update it to accept `OperatingModeNormalizer` (same `.transform()` interface, just correct type annotation). Update the test file to match.

**Files:**
- Modify: `src/turbofan/models/test_evaluation.py`
- Modify: `tests/models/test_test_evaluation.py`

- [ ] **Step 1: Update `src/turbofan/models/test_evaluation.py`**

Replace the `SequenceNormalizer` import with `OperatingModeNormalizer`:

```python
# Remove:
from turbofan.sequences.normalize import SequenceNormalizer

# Add:
from turbofan.preprocessing.normalization import OperatingModeNormalizer
```

Update the `evaluate_test_from_df` signature and docstring:

```python
def evaluate_test_from_df(
    test_df: pd.DataFrame,
    rul_labels: pd.Series,
    model: GRURULRegressor,
    normalizer: OperatingModeNormalizer,
    feature_cols: list[str],
    device: torch.device,
    window_size: int,
    batch_size: int,
    max_rul: int,
) -> dict[str, float]:
    """Evaluate a trained model on pre-loaded test data.

    Normalizes the test DataFrame, builds final windows, predicts,
    aligns labels, and computes regression metrics.

    Args:
        test_df: Raw or feature-engineered test DataFrame.
        rul_labels: Official RUL labels in full test engine order.
        model: Trained GRU model.
        normalizer: Fitted operating-mode normalizer (trained on training data).
        feature_cols: Feature columns matching the model's input.
        device: Torch device for inference.
        window_size: Sequence window size.
        batch_size: Inference batch size.
        max_rul: Maximum RUL cap for prediction rescaling.

    Returns:
        Dict with ``test_rmse``, ``test_mae``, ``test_phm08_score``.
    """
    ...  # body unchanged
```

Update `evaluate_official_test` signature similarly:

```python
def evaluate_official_test(
    data_config: DataConfig,
    model: GRURULRegressor,
    normalizer: OperatingModeNormalizer,
    feature_cols: list[str],
    device: torch.device,
    window_size: int,
    batch_size: int,
) -> dict[str, float] | None:
```

- [ ] **Step 2: Update `tests/models/test_test_evaluation.py`**

Replace `SequenceNormalizer` with `OperatingModeNormalizer` throughout:

```python
# Remove:
from turbofan.sequences.normalize import SequenceNormalizer

# Add:
from turbofan.preprocessing.normalization import OperatingModeNormalizer
```

In `TestEvaluateTestFromDf.test_returns_test_metrics`, replace:
```python
# OLD:
normalizer = SequenceNormalizer(feature_cols=feature_cols)
normalizer.fit_transform(test_df.copy())

# NEW:
normalizer = OperatingModeNormalizer(feature_cols=feature_cols)
normalizer.fit(test_df)
```

In `TestEvaluateOfficialTest.test_returns_none_when_test_files_missing`:
```python
# OLD:
normalizer = SequenceNormalizer(feature_cols=feature_cols)

# NEW:
normalizer = OperatingModeNormalizer(feature_cols=feature_cols)
# Also fit it on some dummy data (OperatingModeNormalizer requires fit before transform):
rows = []
for eid in range(1, 3):
    for cycle in range(1, 6):
        rows.append({"engine_id": eid, "cycle": cycle, "op_1": 0.0, "s_1": float(cycle)})
normalizer.fit(pd.DataFrame(rows))
```

In `TestEvaluateOfficialTest.test_returns_metrics_when_files_exist`, replace `SequenceNormalizer` with `OperatingModeNormalizer` and update `fit_transform` to `fit` (since `OperatingModeNormalizer` is sklearn-compatible, it has `fit_transform`):
```python
# OLD:
normalizer = SequenceNormalizer(feature_cols=feature_cols)
train_raw = load_raw_train(data_config)
normalizer.fit_transform(train_raw)

# NEW:
normalizer = OperatingModeNormalizer(feature_cols=feature_cols)
train_raw = load_raw_train(data_config)
normalizer.fit(train_raw)
```

- [ ] **Step 3: Run test_evaluation tests**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
pytest tests/models/test_test_evaluation.py -v
```

Expected: all PASS.

- [ ] **Step 4: Lint and type-check**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
ruff check src/turbofan/models/test_evaluation.py tests/models/test_test_evaluation.py
mypy src/turbofan/models/test_evaluation.py
```

- [ ] **Step 5: Commit**

```powershell
git add src/turbofan/models/test_evaluation.py tests/models/test_test_evaluation.py
git commit -m "fix(test_evaluation): update evaluate_test_from_df to accept OperatingModeNormalizer"
```

---

## Task 8: Delete dead code

Remove `default_feature_cols` from `sequences/normalize.py`, delete `rolling.py` and its tests, and clean up `test_normalize.py`.

**Files:**
- Modify: `src/turbofan/sequences/normalize.py` (remove `default_feature_cols`)
- Modify: `tests/sequences/test_normalize.py` (remove `default_feature_cols` tests)
- Delete: `src/turbofan/features/rolling.py`
- Delete: `tests/features/test_rolling.py`

- [ ] **Step 1: Verify no remaining imports of dead symbols**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
python -c "
import subprocess, sys
for sym in ['default_feature_cols', 'RollingFeatureExtractor', 'SequenceNormalizer']:
    result = subprocess.run(['python', '-m', 'grep', '-r', sym, 'src/'], capture_output=True, text=True)
    print(sym, ':', result.stdout.strip() or 'not found')
"
```

Use Grep instead:
```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
# Search for any remaining imports (should be empty in src/ after all tasks)
```

Manually check:
- `default_feature_cols` — should only appear in `sequences/normalize.py` and `test_normalize.py`
- `RollingFeatureExtractor` — should only appear in `features/rolling.py` and `test_rolling.py`
- `SequenceNormalizer` — should only appear in `sequences/normalize.py` (definition)

If any of these appear elsewhere in `src/`, fix those imports BEFORE deleting.

- [ ] **Step 2: Remove `default_feature_cols` from `src/turbofan/sequences/normalize.py`**

Delete the `default_feature_cols` function (lines 10-16 in the current file). Keep `SequenceNormalizer` (it is still the class definition, not harmful dead code, and removing it is deferred to a future cleanup if it has no callers). Actually since `test_evaluation.py` and its tests no longer reference `SequenceNormalizer` after Task 7, check if it has any callers:

If `SequenceNormalizer` has no callers remaining in `src/` or `tests/`, delete the entire class from `sequences/normalize.py` too. If the file becomes empty after both deletions, delete it.

- [ ] **Step 3: Update `tests/sequences/test_normalize.py`**

Remove the two tests that use `default_feature_cols`:

```python
# Delete:
def test_default_feature_cols_returns_ops_and_sensors() -> None: ...
def test_operating_mode_normalizer_works_with_default_sequence_feature_cols() -> None: ...
```

If the file is now empty, delete it entirely.

- [ ] **Step 4: Delete `src/turbofan/features/rolling.py` and its tests**

```powershell
Remove-Item src/turbofan/features/rolling.py
Remove-Item tests/features/test_rolling.py
```

- [ ] **Step 5: Run full test suite to confirm green**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
pytest -q 2>&1 | tail -20
```

Expected: all PASS, no import errors.

- [ ] **Step 6: Final lint and type-check**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"; conda activate mlops
ruff check src/ tests/
mypy src/turbofan
```

Fix any remaining issues.

- [ ] **Step 7: Commit**

```powershell
git add -A
git commit -m "chore: delete dead code (rolling.py, default_feature_cols, SequenceNormalizer)"
```

---

## Self-Review Against Spec

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `SensorDropper → OperatingModeNormalizer → SensorColumnSelector → FeatureEngineer` | Task 3 |
| `FeatureConfig.feature_set`, `windows`, `lag_steps` | Task 2 |
| Remove `ModelConfig.windows`, `ModelConfig.feature_set` | Task 2 |
| `FeatureEngineer` in `engineering.py` with all 6 feature sets | Task 1 |
| `SensorColumnSelector` in `pipeline.py` | Task 3 |
| `build_feature_pipeline` new 4-step signature | Task 3 |
| Ridge: 2-step pipeline, delete dead helpers | Task 3 |
| `train_baseline.py` reads from `cfg.features.*` | Task 4 |
| GRU: replace normalizer with `build_feature_pipeline` | Task 5 |
| GRU sweep: same | Task 6 |
| GRU `feature_cols` from `pipeline.named_steps["feature_engineer"].feature_cols_` | Task 5 |
| `_evaluate_official_test` uses `pipeline.transform` | Task 5 |
| Checkpoint stores `normalizer_payload` from `pipeline.named_steps["normalizer"]` | Task 5 |
| `evaluate_test_from_df` accepts `OperatingModeNormalizer` | Task 7 |
| Delete `default_feature_cols` | Task 8 |
| Delete `SequenceNormalizer` (if dead code) | Task 8 |
| Delete `rolling.py` | Task 8 |
| New pipeline tests (sensor-only output, engine_id absent, engine boundary) | Task 3 |
| New `FeatureEngineer` tests (all feature sets, boundaries, backfill) | Task 1 |

All spec sections are covered. No placeholders remain in the plan.
