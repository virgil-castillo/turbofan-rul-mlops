# Feature Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an sklearn-compatible feature pipeline that drops constant sensors, computes multi-window rolling statistics, and normalizes by operational condition — supporting all FD subsets.

**Architecture:** Three custom sklearn transformers (`SensorDropper`, `RollingFeatureExtractor`, `OperationalNormalizer`) chained into an `sklearn.pipeline.Pipeline` by a factory function. Each transformer uses fit/transform semantics to prevent data leakage. All accept and return pandas DataFrames.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Add scikit-learn to main dependencies, sklearn mypy override |
| `src/turbofan/features/__init__.py` | Subpackage marker |
| `src/turbofan/features/sensor_dropper.py` | `SensorDropper` transformer |
| `src/turbofan/features/rolling.py` | `RollingFeatureExtractor` transformer |
| `src/turbofan/features/normalizer.py` | `OperationalNormalizer` transformer |
| `src/turbofan/features/pipeline.py` | `build_feature_pipeline()` factory |
| `tests/features/__init__.py` | Test subpackage marker |
| `tests/features/test_sensor_dropper.py` | SensorDropper tests |
| `tests/features/test_rolling.py` | RollingFeatureExtractor tests |
| `tests/features/test_normalizer.py` | OperationalNormalizer tests |
| `tests/features/test_pipeline.py` | End-to-end pipeline tests |

---

## Task 1: Project config and subpackage scaffolding

**Files:**
- Modify: `pyproject.toml:9-14` and `pyproject.toml:46-48`
- Create: `src/turbofan/features/__init__.py`
- Create: `tests/features/__init__.py`

- [ ] **Step 1: Add scikit-learn to main dependencies in `pyproject.toml`**

Replace the `dependencies` list at lines 9-14:

```toml
dependencies = [
    "numpy",
    "pandas>=2.0",
    "pydantic>=2.0",
    "pyyaml",
    "scikit-learn",
]
```

- [ ] **Step 2: Add sklearn mypy override in `pyproject.toml`**

After the existing `scipy.*` override (lines 46-48), add:

```toml
[[tool.mypy.overrides]]
module = "sklearn.*"
ignore_missing_imports = true
```

- [ ] **Step 3: Create `src/turbofan/features/__init__.py`**

```python
"""Feature engineering pipeline for C-MAPSS turbofan data."""
```

- [ ] **Step 4: Create `tests/features/__init__.py`**

Empty file (test package marker).

- [ ] **Step 5: Install updated dependencies**

Run:
```bash
pip install -e ".[dev]"
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/turbofan/features/__init__.py tests/features/__init__.py
git commit -m "chore: add scikit-learn dep and features subpackage"
```

---

## Task 2: SensorDropper transformer (TDD)

**Files:**
- Create: `tests/features/test_sensor_dropper.py`
- Create: `src/turbofan/features/sensor_dropper.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_sensor_dropper.py`:

```python
"""Tests for turbofan.features.sensor_dropper."""
from __future__ import annotations

import pandas as pd

from turbofan.features.sensor_dropper import SensorDropper


def _make_df_with_constant() -> pd.DataFrame:
    """DataFrame with s_1 varying, s_2 constant, s_3 varying."""
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "op_1": [0.0, 0.0, 0.0, 0.0, 0.0],
            "s_1": [100.0, 101.0, 102.0, 100.5, 101.5],
            "s_2": [200.0, 200.0, 200.0, 200.0, 200.0],
            "s_3": [50.0, 51.0, 52.0, 50.5, 51.5],
        }
    )


def test_drops_constant_sensor() -> None:
    """Constant sensor s_2 is removed after fit + transform."""
    df = _make_df_with_constant()
    dropper = SensorDropper()
    dropper.fit(df)
    result = dropper.transform(df)
    assert "s_2" not in result.columns


def test_keeps_varying_sensors() -> None:
    """Non-constant sensors s_1 and s_3 survive."""
    df = _make_df_with_constant()
    dropper = SensorDropper()
    result = dropper.fit_transform(df)
    assert "s_1" in result.columns
    assert "s_3" in result.columns


def test_keeps_non_sensor_columns() -> None:
    """engine_id, cycle, op_1 are never dropped."""
    df = _make_df_with_constant()
    dropper = SensorDropper()
    result = dropper.fit_transform(df)
    assert "engine_id" in result.columns
    assert "cycle" in result.columns
    assert "op_1" in result.columns


def test_fit_train_transform_test() -> None:
    """Fit on train, transform on test drops same columns."""
    train = _make_df_with_constant()
    test = pd.DataFrame(
        {
            "engine_id": [3, 3],
            "cycle": [1, 2],
            "op_1": [0.0, 0.0],
            "s_1": [99.0, 100.0],
            "s_2": [999.0, 998.0],
            "s_3": [60.0, 61.0],
        }
    )
    dropper = SensorDropper()
    dropper.fit(train)
    result = dropper.transform(test)
    assert "s_2" not in result.columns
    assert "s_1" in result.columns


def test_keep_parameter_prevents_drop() -> None:
    """Force-keeping a constant sensor via the keep parameter."""
    df = _make_df_with_constant()
    dropper = SensorDropper(keep=["s_2"])
    dropper.fit(df)
    result = dropper.transform(df)
    assert "s_2" in result.columns


def test_empty_drop_list_when_all_vary() -> None:
    """No sensors dropped when all have nonzero variance."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 2],
            "cycle": [1, 1],
            "s_1": [1.0, 2.0],
            "s_2": [3.0, 4.0],
        }
    )
    dropper = SensorDropper()
    dropper.fit(df)
    assert dropper.columns_to_drop_ == []
    result = dropper.transform(df)
    assert "s_1" in result.columns
    assert "s_2" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/features/test_sensor_dropper.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'turbofan.features.sensor_dropper'`

- [ ] **Step 3: Implement `src/turbofan/features/sensor_dropper.py`**

```python
"""Constant sensor removal transformer."""
from __future__ import annotations

from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SensorDropper(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Drop sensor columns with zero variance in the training set.

    Discovers sensor columns by the ``s_*`` naming convention. Columns
    with zero standard deviation across the entire training set carry
    no information and are removed. Non-sensor columns (engine_id,
    cycle, op_*) are always preserved.

    Args:
        keep: Optional list of sensor column names to force-keep
            even if they appear constant in training data.
    """

    def __init__(
        self, keep: list[str] | None = None
    ) -> None:
        self.keep = keep

    def fit(
        self, X: pd.DataFrame, y: object = None
    ) -> Self:
        """Identify constant sensor columns to drop.

        Args:
            X: Training DataFrame with sensor columns.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted transformer.
        """
        keep_set = set(self.keep or [])
        sensor_cols = [
            c for c in X.columns if c.startswith("s_")
        ]
        self.columns_to_drop_: list[str] = [
            col
            for col in sensor_cols
            if X[col].std() == 0.0 and col not in keep_set
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove constant sensor columns.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with constant sensors removed.
        """
        return X.drop(columns=self.columns_to_drop_)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/features/test_sensor_dropper.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/features/test_sensor_dropper.py src/turbofan/features/sensor_dropper.py
git commit -m "feat: add SensorDropper transformer (drops zero-variance sensors)"
```

---

## Task 3: RollingFeatureExtractor transformer (TDD)

**Files:**
- Create: `tests/features/test_rolling.py`
- Create: `src/turbofan/features/rolling.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_rolling.py`:

```python
"""Tests for turbofan.features.rolling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbofan.features.rolling import RollingFeatureExtractor


def _make_two_engine_df() -> pd.DataFrame:
    """Two engines, 10 cycles each, 2 sensors."""
    rng = np.random.default_rng(42)
    rows = []
    for eid in [1, 2]:
        for cyc in range(1, 11):
            rows.append(
                {
                    "engine_id": eid,
                    "cycle": cyc,
                    "op_1": 0.0,
                    "s_1": float(cyc) + rng.normal(0, 0.1),
                    "s_2": 50.0 + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_correct_number_of_new_columns() -> None:
    """2 sensors x 2 windows x 4 stats = 16 new columns."""
    df = _make_two_engine_df()
    ext = RollingFeatureExtractor(windows=[5, 10])
    result = ext.fit_transform(df)
    original_cols = len(df.columns)
    expected_new = 2 * 2 * 4  # sensors * windows * stats
    assert len(result.columns) == original_cols + expected_new


def test_column_naming_convention() -> None:
    """Rolling columns follow {sensor}_{stat}_{window} pattern."""
    df = _make_two_engine_df()
    ext = RollingFeatureExtractor(windows=[5])
    result = ext.fit_transform(df)
    assert "s_1_rmean_5" in result.columns
    assert "s_1_rstd_5" in result.columns
    assert "s_1_rmin_5" in result.columns
    assert "s_1_rmax_5" in result.columns


def test_rolling_mean_on_constant_series() -> None:
    """Rolling mean of a constant series equals that constant."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5],
            "s_1": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    ext = RollingFeatureExtractor(windows=[3])
    result = ext.fit_transform(df)
    assert np.allclose(result["s_1_rmean_3"], 10.0)


def test_no_cross_engine_bleed() -> None:
    """Rolling stats reset at engine boundaries."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "s_1": [100.0, 200.0, 0.0, 0.0],
        }
    )
    ext = RollingFeatureExtractor(windows=[3])
    result = ext.fit_transform(df)
    engine2_first = result.loc[
        result["engine_id"] == 2, "s_1_rmean_3"
    ].iloc[0]
    assert engine2_first == 0.0


def test_min_periods_fills_early_cycles() -> None:
    """First cycle of each engine has values, not NaN."""
    df = _make_two_engine_df()
    ext = RollingFeatureExtractor(windows=[5])
    result = ext.fit_transform(df)
    first_rows = result.groupby("engine_id").first()
    assert not first_rows["s_1_rmean_5"].isna().any()
    assert not first_rows["s_1_rstd_5"].isna().any()
    assert not first_rows["s_1_rmin_5"].isna().any()
    assert not first_rows["s_1_rmax_5"].isna().any()


def test_original_columns_preserved() -> None:
    """Original columns remain unchanged."""
    df = _make_two_engine_df()
    ext = RollingFeatureExtractor(windows=[5])
    result = ext.fit_transform(df)
    pd.testing.assert_series_equal(
        result["s_1"], df["s_1"], check_names=True
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/features/test_rolling.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'turbofan.features.rolling'`

- [ ] **Step 3: Implement `src/turbofan/features/rolling.py`**

```python
"""Multi-window rolling feature extraction transformer."""
from __future__ import annotations

from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RollingFeatureExtractor(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Compute rolling statistics per engine for each sensor column.

    For each sensor and each window size, computes four rolling
    aggregations grouped by engine_id: mean, std, min, max.
    Uses ``min_periods=1`` so early cycles get values instead of
    NaN.

    Args:
        windows: List of rolling window sizes in cycles.
            Default ``[5, 10, 20]``.
    """

    def __init__(
        self, windows: list[int] | None = None
    ) -> None:
        self.windows = windows if windows is not None else [5, 10, 20]

    def fit(
        self, X: pd.DataFrame, y: object = None
    ) -> Self:
        """Discover sensor columns from the training data.

        Args:
            X: Training DataFrame with sensor columns.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted transformer.
        """
        self.sensor_cols_: list[str] = [
            c for c in X.columns if c.startswith("s_")
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add rolling feature columns to the DataFrame.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with original columns plus rolling features.
        """
        result = X.copy()
        for window in self.windows:
            for col in self.sensor_cols_:
                grp = X.groupby("engine_id")[col]
                result[f"{col}_rmean_{window}"] = grp.transform(
                    lambda s: s.rolling(
                        window, min_periods=1
                    ).mean()
                )
                result[f"{col}_rstd_{window}"] = (
                    grp.transform(
                        lambda s: s.rolling(
                            window, min_periods=1
                        ).std()
                    ).fillna(0.0)
                )
                result[f"{col}_rmin_{window}"] = grp.transform(
                    lambda s: s.rolling(
                        window, min_periods=1
                    ).min()
                )
                result[f"{col}_rmax_{window}"] = grp.transform(
                    lambda s: s.rolling(
                        window, min_periods=1
                    ).max()
                )
        return result
```

Note: `.fillna(0.0)` on the rolling std handles the first cycle per engine where `std()` with `ddof=1` returns NaN for a single value.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/features/test_rolling.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/features/test_rolling.py src/turbofan/features/rolling.py
git commit -m "feat: add RollingFeatureExtractor (multi-window rolling stats per engine)"
```

---

## Task 4: OperationalNormalizer transformer (TDD)

**Files:**
- Create: `tests/features/test_normalizer.py`
- Create: `src/turbofan/features/normalizer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_normalizer.py`:

```python
"""Tests for turbofan.features.normalizer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbofan.features.normalizer import OperationalNormalizer


def _make_single_condition_df() -> pd.DataFrame:
    """Single operating condition, 2 sensors."""
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2, 3, 4],
            "op_1": [0.0] * 8,
            "op_2": [0.0] * 8,
            "op_3": [0.0] * 8,
            "s_1": [
                100.0, 102.0, 104.0, 106.0,
                101.0, 103.0, 105.0, 107.0,
            ],
            "s_2": [
                50.0, 52.0, 54.0, 56.0,
                51.0, 53.0, 55.0, 57.0,
            ],
        }
    )


def _make_multi_condition_df() -> pd.DataFrame:
    """Two operating conditions with different sensor ranges."""
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2, 3, 4],
            "op_1": [
                1.0, 1.0, 1.0, 1.0,
                2.0, 2.0, 2.0, 2.0,
            ],
            "op_2": [0.0] * 8,
            "op_3": [0.0] * 8,
            "s_1": [
                98.0, 100.0, 102.0, 104.0,
                198.0, 200.0, 202.0, 204.0,
            ],
        }
    )


def test_zscore_mean_near_zero() -> None:
    """After fit_transform, sensor columns have ~0 mean."""
    df = _make_single_condition_df()
    norm = OperationalNormalizer()
    result = norm.fit_transform(df)
    assert abs(result["s_1"].mean()) < 1e-10
    assert abs(result["s_2"].mean()) < 1e-10


def test_zscore_std_near_one() -> None:
    """After fit_transform, sensor columns have ~1 std."""
    df = _make_single_condition_df()
    norm = OperationalNormalizer()
    result = norm.fit_transform(df)
    assert abs(result["s_1"].std() - 1.0) < 1e-10
    assert abs(result["s_2"].std() - 1.0) < 1e-10


def test_multi_condition_normalization() -> None:
    """Each condition is normalized independently."""
    df = _make_multi_condition_df()
    norm = OperationalNormalizer()
    result = norm.fit_transform(df)
    cond_a = result[result["op_1"] == 1.0]["s_1"]
    cond_b = result[result["op_1"] == 2.0]["s_1"]
    assert abs(cond_a.mean()) < 1e-10
    assert abs(cond_b.mean()) < 1e-10


def test_unseen_condition_uses_global_stats() -> None:
    """Transform with an unseen condition falls back to globals."""
    train = _make_multi_condition_df()
    test = pd.DataFrame(
        {
            "engine_id": [3, 3],
            "cycle": [1, 2],
            "op_1": [9.0, 9.0],
            "op_2": [0.0, 0.0],
            "op_3": [0.0, 0.0],
            "s_1": [150.0, 152.0],
        }
    )
    norm = OperationalNormalizer()
    norm.fit(train)
    result = norm.transform(test)
    assert not result["s_1"].isna().any()


def test_zero_std_no_nan() -> None:
    """Constant sensor within a condition produces 0, not NaN."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "op_1": [0.0] * 4,
            "op_2": [0.0] * 4,
            "op_3": [0.0] * 4,
            "s_1": [100.0, 100.0, 100.0, 100.0],
        }
    )
    norm = OperationalNormalizer()
    result = norm.fit_transform(df)
    assert not result["s_1"].isna().any()
    assert not np.isinf(result["s_1"]).any()


def test_non_sensor_columns_unchanged() -> None:
    """engine_id, cycle, op_cols pass through untouched."""
    df = _make_single_condition_df()
    norm = OperationalNormalizer()
    result = norm.fit_transform(df)
    pd.testing.assert_series_equal(
        result["engine_id"], df["engine_id"]
    )
    pd.testing.assert_series_equal(
        result["cycle"], df["cycle"]
    )
    pd.testing.assert_series_equal(
        result["op_1"], df["op_1"]
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/features/test_normalizer.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'turbofan.features.normalizer'`

- [ ] **Step 3: Implement `src/turbofan/features/normalizer.py`**

```python
"""Operational-condition-aware z-score normalization."""
from __future__ import annotations

from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OperationalNormalizer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Z-score normalize sensor readings by operational condition.

    Groups data by unique combinations of operational setting
    columns, then applies per-group z-score normalization. Falls
    back to global statistics for operating conditions seen at
    test time but not in training.

    Args:
        op_cols: Operational setting column names.
            Default ``["op_1", "op_2", "op_3"]``.
    """

    def __init__(
        self, op_cols: list[str] | None = None
    ) -> None:
        self.op_cols = (
            op_cols
            if op_cols is not None
            else ["op_1", "op_2", "op_3"]
        )

    def fit(
        self, X: pd.DataFrame, y: object = None
    ) -> Self:
        """Compute per-condition and global normalization stats.

        Args:
            X: Training DataFrame.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted transformer.
        """
        exclude = {"engine_id", "cycle", *self.op_cols}
        self.numeric_cols_: list[str] = [
            c
            for c in X.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(X[c])
        ]
        grouped = X.groupby(self.op_cols)[self.numeric_cols_]
        self.group_means_: pd.DataFrame = grouped.mean()
        self.group_stds_: pd.DataFrame = (
            grouped.std().fillna(1.0).replace(0.0, 1.0)
        )
        self.global_mean_: pd.Series[float] = (
            X[self.numeric_cols_].mean()
        )
        global_std = X[self.numeric_cols_].std()
        self.global_std_: pd.Series[float] = (
            global_std.fillna(1.0).replace(0.0, 1.0)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply per-condition z-score normalization.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with normalized numeric columns.
        """
        result = X.copy()
        for condition, group in result.groupby(self.op_cols):
            if not isinstance(condition, tuple):
                condition = (condition,)
            idx = group.index
            if condition in self.group_means_.index:
                means = self.group_means_.loc[condition]
                stds = self.group_stds_.loc[condition]
            else:
                means = self.global_mean_
                stds = self.global_std_
            for col in self.numeric_cols_:
                if col in result.columns:
                    result.loc[idx, col] = (
                        (X.loc[idx, col] - means[col])
                        / stds[col]
                    )
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/features/test_normalizer.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/features/test_normalizer.py src/turbofan/features/normalizer.py
git commit -m "feat: add OperationalNormalizer (condition-aware z-score)"
```

---

## Task 5: Pipeline factory (TDD)

**Files:**
- Create: `tests/features/test_pipeline.py`
- Create: `src/turbofan/features/pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_pipeline.py`:

```python
"""Tests for turbofan.features.pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbofan.features.pipeline import build_feature_pipeline


def _make_train_df() -> pd.DataFrame:
    """Training data: 2 engines, 15 cycles, 3 sensors (1 constant)."""
    rng = np.random.default_rng(42)
    rows = []
    for eid in [1, 2]:
        for cyc in range(1, 16):
            rows.append(
                {
                    "engine_id": eid,
                    "cycle": cyc,
                    "op_1": 0.0,
                    "op_2": 0.0,
                    "op_3": 0.0,
                    "s_1": float(cyc) + rng.normal(0, 0.5),
                    "s_2": 200.0,
                    "s_3": 80.0 + rng.normal(0, 1.0),
                }
            )
    return pd.DataFrame(rows)


def _make_test_df() -> pd.DataFrame:
    """Test data: 1 engine, 10 cycles, same columns."""
    rng = np.random.default_rng(99)
    rows = []
    for cyc in range(1, 11):
        rows.append(
            {
                "engine_id": 3,
                "cycle": cyc,
                "op_1": 0.0,
                "op_2": 0.0,
                "op_3": 0.0,
                "s_1": float(cyc) + rng.normal(0, 0.5),
                "s_2": 200.0,
                "s_3": 80.0 + rng.normal(0, 1.0),
            }
        )
    return pd.DataFrame(rows)


def test_fit_transform_no_nans() -> None:
    """Pipeline output has no NaN values."""
    train = _make_train_df()
    pipe = build_feature_pipeline(windows=[3, 5])
    result = pipe.fit_transform(train)
    assert not result.isna().any().any()


def test_transform_test_no_nans() -> None:
    """Fit on train, transform test produces no NaNs."""
    train = _make_train_df()
    test = _make_test_df()
    pipe = build_feature_pipeline(windows=[3, 5])
    pipe.fit(train)
    result = pipe.transform(test)
    assert not result.isna().any().any()


def test_constant_sensor_dropped() -> None:
    """Constant sensor s_2 is removed by the pipeline."""
    train = _make_train_df()
    pipe = build_feature_pipeline(windows=[3])
    result = pipe.fit_transform(train)
    assert "s_2" not in result.columns
    s2_rolling = [
        c for c in result.columns if c.startswith("s_2_")
    ]
    assert s2_rolling == []


def test_rolling_columns_present() -> None:
    """Rolling feature columns exist in output."""
    train = _make_train_df()
    pipe = build_feature_pipeline(windows=[5])
    result = pipe.fit_transform(train)
    assert "s_1_rmean_5" in result.columns
    assert "s_3_rstd_5" in result.columns


def test_output_is_dataframe() -> None:
    """Pipeline returns a pandas DataFrame, not numpy array."""
    train = _make_train_df()
    pipe = build_feature_pipeline(windows=[3])
    result = pipe.fit_transform(train)
    assert isinstance(result, pd.DataFrame)


def test_pipeline_named_steps() -> None:
    """Pipeline has the three expected named steps."""
    pipe = build_feature_pipeline()
    assert "sensor_dropper" in pipe.named_steps
    assert "rolling_features" in pipe.named_steps
    assert "normalizer" in pipe.named_steps


def test_multi_condition_pipeline() -> None:
    """Pipeline handles data with multiple op conditions."""
    rng = np.random.default_rng(7)
    rows = []
    for eid, op in [(1, 1.0), (2, 1.0), (3, 2.0), (4, 2.0)]:
        for cyc in range(1, 11):
            rows.append(
                {
                    "engine_id": eid,
                    "cycle": cyc,
                    "op_1": op,
                    "op_2": 0.0,
                    "op_3": 0.0,
                    "s_1": op * 100 + rng.normal(0, 1.0),
                }
            )
    df = pd.DataFrame(rows)
    pipe = build_feature_pipeline(windows=[3])
    result = pipe.fit_transform(df)
    assert not result.isna().any().any()


def test_joblib_serialization() -> None:
    """Fitted pipeline survives joblib round-trip."""
    import io
    import joblib

    train = _make_train_df()
    pipe = build_feature_pipeline(windows=[3])
    pipe.fit(train)

    buffer = io.BytesIO()
    joblib.dump(pipe, buffer)
    buffer.seek(0)
    loaded = joblib.load(buffer)

    test = _make_test_df()
    result_original = pipe.transform(test)
    result_loaded = loaded.transform(test)
    pd.testing.assert_frame_equal(result_original, result_loaded)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/features/test_pipeline.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'turbofan.features.pipeline'`

- [ ] **Step 3: Implement `src/turbofan/features/pipeline.py`**

```python
"""Feature engineering pipeline factory."""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from turbofan.features.normalizer import OperationalNormalizer
from turbofan.features.rolling import RollingFeatureExtractor
from turbofan.features.sensor_dropper import SensorDropper


def build_feature_pipeline(
    windows: list[int] | None = None,
    op_cols: list[str] | None = None,
) -> Pipeline:  # type: ignore[type-arg]
    """Build an unfitted feature engineering pipeline.

    Returns an sklearn Pipeline with three named steps:
    ``sensor_dropper``, ``rolling_features``, ``normalizer``.

    Args:
        windows: Rolling window sizes. Default ``[5, 10, 20]``.
        op_cols: Operational setting columns.
            Default ``["op_1", "op_2", "op_3"]``.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline(
        [
            ("sensor_dropper", SensorDropper()),
            (
                "rolling_features",
                RollingFeatureExtractor(windows=windows),
            ),
            (
                "normalizer",
                OperationalNormalizer(op_cols=op_cols),
            ),
        ]
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/features/test_pipeline.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Run full test suite**

Run:
```bash
pytest -v
```

Expected: all tests pass (50 existing + 6 dropper + 7 rolling + 7 normalizer + 8 pipeline = 78).

- [ ] **Step 6: Commit**

```bash
git add tests/features/test_pipeline.py src/turbofan/features/pipeline.py
git commit -m "feat: add build_feature_pipeline factory with end-to-end tests"
```

---

## Task 6: Lint and type-check pass

**Files:** no new files — fix any issues found

- [ ] **Step 1: Run ruff**

Run:
```bash
ruff check src/ tests/
```

Expected: `All checks passed.`

If errors, fix them: `ruff check --fix src/ tests/` for auto-fixable issues, then manual fixes for the rest.

- [ ] **Step 2: Run mypy**

Run:
```bash
mypy src/turbofan
```

Expected: `Success: no issues found in N source files`

If mypy complains about sklearn type annotations, verify the `[[tool.mypy.overrides]]` for `sklearn.*` is in `pyproject.toml`. If it complains about `Self`, verify `from typing import Self` is at the top of each transformer file.

- [ ] **Step 3: Run full test suite**

Run:
```bash
pytest -v --tb=short
```

Expected: all 78 tests pass.

- [ ] **Step 4: Commit (only if changes were made)**

```bash
git add -u
git commit -m "chore: fix lint and type-check issues for features module"
```

---

## Task 7: Push to remote

- [ ] **Step 1: Run the full test suite one final time**

Run:
```bash
pytest -v --tb=short
```

Expected: all tests pass.

- [ ] **Step 2: Push all commits**

Run:
```bash
git push
```

Expected: all commits pushed to `origin/master`.

---

## Self-Review Checklist

**Spec coverage:**
- [x] `SensorDropper` with `keep` parameter — Task 2
- [x] `SensorDropper` discovers by `s_*` prefix, not hardcoded — Task 2
- [x] `RollingFeatureExtractor` with configurable windows `[5, 10, 20]` — Task 3
- [x] Rolling mean, std, min, max per sensor per window — Task 3
- [x] Rolling stats grouped by engine_id — Task 3
- [x] `min_periods=1` for early cycles — Task 3
- [x] Column naming `{sensor}_{stat}_{window}` — Task 3
- [x] `OperationalNormalizer` with configurable op_cols — Task 4
- [x] Per-condition z-score normalization — Task 4
- [x] Global stats fallback for unseen conditions — Task 4
- [x] Zero-std handling (replace with 1.0) — Task 4
- [x] `build_feature_pipeline()` factory — Task 5
- [x] Three named steps: sensor_dropper, rolling_features, normalizer — Task 5
- [x] Pipeline serialization with joblib — Task 5
- [x] Multi-condition support (FD002+ style) — Tasks 4, 5
- [x] scikit-learn in main dependencies — Task 1
- [x] sklearn mypy override — Task 1
- [x] All tests use synthetic DataFrames — Tasks 2, 3, 4, 5

**Type consistency:**
- `SensorDropper.__init__(keep: list[str] | None)` — consistent Task 2 ✓
- `SensorDropper.fit(X: pd.DataFrame, y: object = None) -> Self` — consistent ✓
- `SensorDropper.transform(X: pd.DataFrame) -> pd.DataFrame` — consistent ✓
- `RollingFeatureExtractor.__init__(windows: list[int] | None)` — consistent Task 3 ✓
- `RollingFeatureExtractor.sensor_cols_: list[str]` — consistent ✓
- `OperationalNormalizer.__init__(op_cols: list[str] | None)` — consistent Task 4 ✓
- `OperationalNormalizer.group_means_: pd.DataFrame` — consistent ✓
- `OperationalNormalizer.group_stds_: pd.DataFrame` — consistent ✓
- `OperationalNormalizer.global_mean_: pd.Series[float]` — consistent ✓
- `OperationalNormalizer.global_std_: pd.Series[float]` — consistent ✓
- `build_feature_pipeline(windows, op_cols) -> Pipeline` — consistent Task 5 ✓
