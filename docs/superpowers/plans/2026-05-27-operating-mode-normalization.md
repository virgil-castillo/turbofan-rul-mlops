# Operating-Mode Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace both legacy normalizers (`OperationalNormalizer` and `SequenceNormalizer`) with a single shared `OperatingModeNormalizer` that normalizes sensor features per learned operating-mode cluster and is used consistently across baseline training, GRU training, GRU sweeps, and GRU inference.

**Architecture:** `OperatingModeNormalizer` lives in a new `turbofan.preprocessing.normalization` module, is sklearn-compatible (fits on training rows only), and serializes its state via `to_payload`/`from_payload` so GRU artifacts become self-contained. The existing legacy paths (`OperationalNormalizer`, `SequenceNormalizer`) are deleted from active usage and their tests removed; all callers are updated to use the new class.

**Tech Stack:** Python 3.11+, pandas, scikit-learn (`BaseEstimator`, `TransformerMixin`, `KMeans`), NumPy, PyTorch (serialization only), pytest, mypy strict, ruff.

---

## File Map

| Action | Path |
|--------|------|
| Create | `src/turbofan/preprocessing/__init__.py` |
| Create | `src/turbofan/preprocessing/normalization.py` |
| Create | `tests/preprocessing/__init__.py` |
| Create | `tests/preprocessing/test_normalization.py` |
| Modify | `src/turbofan/features/pipeline.py` — use `OperatingModeNormalizer`, add `n_modes`/`random_state` params |
| Modify | `src/turbofan/features/normalizer.py` — delete `OperationalNormalizer` class; keep file as empty module stub |
| Modify | `src/turbofan/models/baseline.py` — add `n_modes`/`random_state` to `build_baseline_pipeline` |
| Modify | `src/turbofan/cli/train_baseline.py` — derive `n_modes` from `mode_count_for_subset` |
| Modify | `src/turbofan/cli/train_sequence_gru.py` — replace `SequenceNormalizer` with `OperatingModeNormalizer`; update checkpoint payload |
| Modify | `src/turbofan/experiments/sequence_gru_sweep.py` — replace `SequenceNormalizer` with `OperatingModeNormalizer` |
| Modify | `src/turbofan/experiments/feature_gru_sweep.py` — replace `SequenceNormalizer` with `OperatingModeNormalizer` |
| Modify | `src/turbofan/inference/predictors.py` — require `normalizer_type == "operating_mode"`; reject legacy flat-stat checkpoints |
| Modify | `tests/features/test_normalizer.py` — delete all tests (class no longer exists) |
| Modify | `tests/features/test_pipeline.py` — add `n_modes`/`random_state` smoke test |
| Modify | `tests/sequences/test_normalize.py` — remove `SequenceNormalizer` tests |
| Modify | `tests/models/test_train_baseline_cli.py` — assert `OperatingModeNormalizer` is used |
| Modify | `tests/models/test_train_sequence_gru_cli.py` — update to new payload format |
| Modify | `tests/models/test_sweep_sequence_gru.py` — update to `OperatingModeNormalizer` |
| Modify | `tests/models/test_sweep_feature_gru.py` — update to `OperatingModeNormalizer` |
| Modify | `tests/inference/test_predictors.py` — update artifact helpers; add legacy rejection test |

---

## Task 1 — Create preprocessing package with mode-count constants

**Files:**
- Create: `src/turbofan/preprocessing/__init__.py`
- Create: `src/turbofan/preprocessing/normalization.py` (constants + `mode_count_for_subset` only)
- Create: `tests/preprocessing/__init__.py`
- Create: `tests/preprocessing/test_normalization.py`

- [ ] **Step 1: Write failing tests for mode-count helpers**

```python
# tests/preprocessing/test_normalization.py
"""Tests for turbofan.preprocessing.normalization."""
from __future__ import annotations

import pytest

from turbofan.preprocessing.normalization import (
    CMAPSS_SUBSET_MODE_COUNTS,
    mode_count_for_subset,
)


def test_all_subset_mode_counts_are_positive_integers() -> None:
    """Every entry in CMAPSS_SUBSET_MODE_COUNTS is a positive integer."""
    for subset, count in CMAPSS_SUBSET_MODE_COUNTS.items():
        assert isinstance(count, int), f"{subset}: expected int, got {type(count)}"
        assert count > 0, f"{subset}: mode count must be positive"


def test_single_condition_subsets_have_mode_count_one() -> None:
    """FD001 and FD003 are treated as single-condition subsets."""
    assert mode_count_for_subset("FD001") == 1
    assert mode_count_for_subset("FD003") == 1


def test_multi_condition_subsets_have_mode_count_six() -> None:
    """FD002 and FD004 are treated as six-condition subsets."""
    assert mode_count_for_subset("FD002") == 6
    assert mode_count_for_subset("FD004") == 6


def test_unsupported_subset_raises_value_error() -> None:
    """mode_count_for_subset raises ValueError for unknown subset names."""
    with pytest.raises(ValueError, match="FD999"):
        mode_count_for_subset("FD999")
```

- [ ] **Step 2: Run tests to confirm they fail**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
pytest tests/preprocessing/test_normalization.py -v
```

Expected: `ModuleNotFoundError` (package doesn't exist yet).

- [ ] **Step 3: Create the package files**

```python
# src/turbofan/preprocessing/__init__.py
"""Shared preprocessing utilities for turbofan models."""
```

```python
# src/turbofan/preprocessing/normalization.py
"""Operating-mode-aware normalization for C-MAPSS turbofan data."""
from __future__ import annotations

CMAPSS_SUBSET_MODE_COUNTS: dict[str, int] = {
    "FD001": 1,
    "FD002": 6,
    "FD003": 1,
    "FD004": 6,
}


def mode_count_for_subset(fd_subset: str) -> int:
    """Return the EDA-confirmed operating-mode count for a C-MAPSS subset.

    Args:
        fd_subset: C-MAPSS subset name, e.g. ``"FD001"``.

    Returns:
        Number of operating modes for the subset.

    Raises:
        ValueError: If ``fd_subset`` is not a supported C-MAPSS subset.
    """
    if fd_subset not in CMAPSS_SUBSET_MODE_COUNTS:
        supported = sorted(CMAPSS_SUBSET_MODE_COUNTS)
        raise ValueError(
            f"Unsupported C-MAPSS subset: {fd_subset!r}. "
            f"Supported subsets: {supported}."
        )
    return CMAPSS_SUBSET_MODE_COUNTS[fd_subset]
```

```python
# tests/preprocessing/__init__.py
```

- [ ] **Step 4: Run tests to confirm they pass**

```powershell
pytest tests/preprocessing/test_normalization.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```powershell
git add src/turbofan/preprocessing/ tests/preprocessing/
git commit -m "feat(preprocessing): add operating-mode count constants and mode_count_for_subset"
```

---

## Task 2 — Implement `OperatingModeNormalizer` (full class, single-mode path)

**Files:**
- Modify: `src/turbofan/preprocessing/normalization.py`
- Modify: `tests/preprocessing/test_normalization.py`

- [ ] **Step 1: Add single-mode normalizer tests**

Append to `tests/preprocessing/test_normalization.py`:

```python
import numpy as np
import pandas as pd

from turbofan.preprocessing.normalization import OperatingModeNormalizer


# ── helpers ────────────────────────────────────────────────────────────────


def _make_df_single_mode() -> pd.DataFrame:
    """Four rows, one operating condition, two sensors, with rul."""
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "op_1": [0.0, 0.0, 0.0, 0.0],
            "op_2": [0.0, 0.0, 0.0, 0.0],
            "op_3": [0.0, 0.0, 0.0, 0.0],
            "s_1": [100.0, 102.0, 104.0, 106.0],
            "s_2": [200.0, 204.0, 208.0, 212.0],
            "rul": [10, 9, 8, 7],
        }
    )


def _make_df_multi_mode() -> pd.DataFrame:
    """Eight rows, two operating modes with different sensor ranges."""
    return pd.DataFrame(
        {
            "engine_id": list(range(1, 9)),
            "cycle": [1] * 8,
            "op_1": [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0],
            "op_2": [0.0] * 8,
            "op_3": [0.0] * 8,
            "s_1": [100.0, 101.0, 99.0, 102.0, 200.0, 201.0, 199.0, 202.0],
        }
    )


# ── constructor validation ─────────────────────────────────────────────────


def test_constructor_raises_for_n_modes_zero() -> None:
    """Constructor raises ValueError when n_modes <= 0."""
    with pytest.raises(ValueError, match="n_modes"):
        OperatingModeNormalizer(n_modes=0)


def test_constructor_raises_for_negative_std_floor() -> None:
    """Constructor raises ValueError when std_floor < 0."""
    with pytest.raises(ValueError, match="std_floor"):
        OperatingModeNormalizer(std_floor=-0.001)


# ── metadata preservation ──────────────────────────────────────────────────


def test_engine_id_cycle_rul_preserved() -> None:
    """engine_id, cycle, and rul pass through unchanged."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1", "s_2"], n_modes=1)
    result = norm.fit_transform(df)
    pd.testing.assert_series_equal(result["engine_id"], df["engine_id"])
    pd.testing.assert_series_equal(result["cycle"], df["cycle"])
    pd.testing.assert_series_equal(result["rul"], df["rul"])


def test_op_cols_not_in_feature_cols_preserved() -> None:
    """Op columns excluded from feature_cols are not modified."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=1)
    result = norm.fit_transform(df)
    pd.testing.assert_series_equal(result["op_1"], df["op_1"])
    pd.testing.assert_series_equal(result["op_2"], df["op_2"])
    pd.testing.assert_series_equal(result["op_3"], df["op_3"])


# ── single-mode normalization ──────────────────────────────────────────────


def test_single_mode_explicit_feature_cols_normalizes_sensors() -> None:
    """Single-mode normalizer z-scores explicit sensor columns (ddof=0)."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1", "s_2"], n_modes=1)
    result = norm.fit_transform(df)
    assert abs(result["s_1"].mean()) < 1e-10
    assert abs(result["s_2"].mean()) < 1e-10


def test_single_mode_inferred_feature_cols_normalizes_sensors() -> None:
    """When feature_cols=None, infer numeric cols excluding engine_id/cycle/rul."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(n_modes=1)
    result = norm.fit_transform(df)
    # s_1, s_2 should be normalized; rul unchanged
    assert abs(result["s_1"].mean()) < 1e-10
    pd.testing.assert_series_equal(result["rul"], df["rul"])


def test_op_cols_in_feature_cols_normalized_globally() -> None:
    """Explicitly included op_cols are normalized with global stats."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(
        feature_cols=["op_1", "op_2", "op_3", "s_1"],
        n_modes=1,
    )
    result = norm.fit_transform(df)
    # op_1 is constant 0.0; std hits floor → normalized value is 0.0
    assert list(result["op_1"]) == [0.0, 0.0, 0.0, 0.0]


def test_near_zero_std_replaced_with_one_in_global_stats() -> None:
    """Global std at or below std_floor is replaced with 1.0."""
    df = pd.DataFrame(
        {
            "engine_id": [1],
            "cycle": [1],
            "op_1": [0.0],
            "op_2": [0.0],
            "op_3": [0.0],
            "s_1": [100.0],  # single row → NaN std → replaced with 1.0
        }
    )
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=1)
    norm.fit(df)
    assert norm.global_stds_["s_1"] == pytest.approx(1.0)


def test_transform_before_fit_raises_runtime_error() -> None:
    """transform raises RuntimeError when called before fit."""
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=1)
    df = pd.DataFrame(
        {"op_1": [0.0], "op_2": [0.0], "op_3": [0.0], "s_1": [1.0]}
    )
    with pytest.raises(RuntimeError, match="fit"):
        norm.transform(df)


def test_fit_raises_key_error_for_missing_op_cols() -> None:
    """fit raises KeyError when op_cols are not in the DataFrame."""
    df = pd.DataFrame({"s_1": [1.0, 2.0]})
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=1)
    with pytest.raises(KeyError):
        norm.fit(df)


def test_transform_returns_copy() -> None:
    """transform returns a new DataFrame; original is not modified."""
    df = _make_df_single_mode()
    original_s1 = df["s_1"].copy()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=1)
    norm.fit(df)
    norm.transform(df)
    pd.testing.assert_series_equal(df["s_1"], original_s1)
```

- [ ] **Step 2: Run new tests to confirm they fail**

```powershell
pytest tests/preprocessing/test_normalization.py -v -k "not test_all_subset and not test_single_condition and not test_multi_condition and not test_unsupported"
```

Expected: `ImportError` (class not defined yet).

- [ ] **Step 3: Implement `OperatingModeNormalizer` (single-mode path)**

Replace the content of `src/turbofan/preprocessing/normalization.py` with:

```python
"""Operating-mode-aware normalization for C-MAPSS turbofan data."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

CMAPSS_SUBSET_MODE_COUNTS: dict[str, int] = {
    "FD001": 1,
    "FD002": 6,
    "FD003": 1,
    "FD004": 6,
}

_DEFAULT_OP_COLS: list[str] = ["op_1", "op_2", "op_3"]
_PASSTHROUGH_COLS: frozenset[str] = frozenset({"engine_id", "cycle", "rul"})


def mode_count_for_subset(fd_subset: str) -> int:
    """Return the EDA-confirmed operating-mode count for a C-MAPSS subset.

    Args:
        fd_subset: C-MAPSS subset name, e.g. ``"FD001"``.

    Returns:
        Number of operating modes for the subset.

    Raises:
        ValueError: If ``fd_subset`` is not a supported C-MAPSS subset.
    """
    if fd_subset not in CMAPSS_SUBSET_MODE_COUNTS:
        supported = sorted(CMAPSS_SUBSET_MODE_COUNTS)
        raise ValueError(
            f"Unsupported C-MAPSS subset: {fd_subset!r}. "
            f"Supported subsets: {supported}."
        )
    return CMAPSS_SUBSET_MODE_COUNTS[fd_subset]


class OperatingModeNormalizer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Z-score normalize sensor features per learned operating-mode cluster.

    Operating-setting columns (``op_cols``) are used to assign each row to a
    mode via KMeans (or mode 0 for single-condition subsets).  Sensor and
    sensor-derived columns are normalized with per-mode statistics; operating
    setting columns that are included in ``feature_cols`` are normalized with
    global statistics.  Metadata columns (``engine_id``, ``cycle``, ``rul``)
    and operating-setting columns not in ``feature_cols`` are copied through
    unchanged.

    Args:
        feature_cols: Columns to normalize.  ``None`` means infer all numeric
            columns excluding ``engine_id``, ``cycle``, and ``rul`` during
            ``fit``.
        op_cols: Operating-setting column names used for mode assignment.
            Default ``["op_1", "op_2", "op_3"]``.
        n_modes: Number of operating modes.  Pass ``1`` to skip KMeans and
            assign every row to mode 0.
        std_floor: Minimum meaningful standard deviation; values at or below
            this are replaced with ``1.0`` to prevent division blow-ups.
        random_state: KMeans random seed.

    Raises:
        ValueError: If ``n_modes <= 0`` or ``std_floor < 0``.
    """

    def __init__(
        self,
        feature_cols: Sequence[str] | None = None,
        op_cols: Sequence[str] | None = None,
        n_modes: int = 1,
        std_floor: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        if n_modes <= 0:
            raise ValueError(f"n_modes must be positive, got {n_modes}.")
        if std_floor < 0:
            raise ValueError(f"std_floor must be >= 0, got {std_floor}.")
        self.feature_cols = feature_cols
        self.op_cols = op_cols
        self.n_modes = n_modes
        self.std_floor = std_floor
        self.random_state = random_state

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Fit mode centers and per-mode normalization statistics.

        Args:
            X: Training DataFrame containing operating-setting and sensor
                columns.
            y: Ignored.  Present for sklearn compatibility.

        Returns:
            Fitted normalizer.

        Raises:
            KeyError: If required ``op_cols`` or explicit ``feature_cols`` are
                missing from ``X``.
            ValueError: If ``n_modes`` exceeds the number of training rows.
        """
        op_cols = list(self.op_cols) if self.op_cols is not None else _DEFAULT_OP_COLS
        missing_op = [c for c in op_cols if c not in X.columns]
        if missing_op:
            raise KeyError(f"Missing op_cols in DataFrame: {missing_op}")

        if self.feature_cols is not None:
            feature_cols: list[str] = list(self.feature_cols)
            missing_feat = [c for c in feature_cols if c not in X.columns]
            if missing_feat:
                raise KeyError(f"Missing feature_cols in DataFrame: {missing_feat}")
        else:
            feature_cols = [
                c
                for c in X.columns
                if c not in _PASSTHROUGH_COLS
                and pd.api.types.is_numeric_dtype(X[c])
            ]

        if self.n_modes > len(X):
            raise ValueError(
                f"n_modes ({self.n_modes}) exceeds number of training rows ({len(X)})."
            )

        self.op_cols_: list[str] = op_cols
        self.feature_cols_: list[str] = feature_cols
        op_col_set = set(op_cols)
        self.sensor_feature_cols_: list[str] = [
            c for c in feature_cols if c not in op_col_set
        ]

        # Mode centers via KMeans (or trivial single-mode).
        if self.n_modes == 1:
            self.mode_centers_: list[list[float]] | None = None
            mode_labels = np.zeros(len(X), dtype=np.int64)
        else:
            kmeans = KMeans(
                n_clusters=self.n_modes,
                n_init=10,
                random_state=self.random_state,
            )
            kmeans.fit(X[op_cols].to_numpy(dtype=np.float64))
            self.mode_centers_ = kmeans.cluster_centers_.tolist()
            mode_labels = kmeans.labels_.astype(np.int64)

        # Global stats over all feature_cols.
        self.global_means_: pd.Series[float] = X[feature_cols].mean()
        global_std = X[feature_cols].std(ddof=0).fillna(1.0)
        self.global_stds_: pd.Series[float] = global_std.mask(
            global_std.abs() <= self.std_floor, 1.0
        )

        # Per-mode stats over sensor_feature_cols_.
        self.mode_means_: dict[int, pd.Series[float]] = {}
        self.mode_stds_: dict[int, pd.Series[float]] = {}
        sensor_cols = self.sensor_feature_cols_
        if sensor_cols:
            for mode_idx in range(self.n_modes):
                rows = X.loc[mode_labels == mode_idx, sensor_cols]
                m = rows.mean()
                s = rows.std(ddof=0).fillna(1.0)
                self.mode_means_[mode_idx] = m
                self.mode_stds_[mode_idx] = s.mask(s.abs() <= self.std_floor, 1.0)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply operating-mode-aware z-score normalization.

        Args:
            X: DataFrame to normalize.  Must contain ``op_cols``.

        Returns:
            Copy of ``X`` with feature columns normalized in place.

        Raises:
            RuntimeError: If called before ``fit``.
            KeyError: If ``op_cols`` are missing from ``X``.
        """
        if not hasattr(self, "feature_cols_"):
            raise RuntimeError(
                "OperatingModeNormalizer must be fit before transform."
            )
        missing_op = [c for c in self.op_cols_ if c not in X.columns]
        if missing_op:
            raise KeyError(f"op_cols missing from DataFrame: {missing_op}")

        result = X.copy()
        present_feature_cols = [c for c in self.feature_cols_ if c in result.columns]
        result[present_feature_cols] = result[present_feature_cols].astype("float64")

        mode_labels = self._assign_modes(result)
        op_col_set = set(self.op_cols_)
        sensor_cols_present = [
            c for c in self.sensor_feature_cols_ if c in result.columns
        ]
        global_op_cols_present = [
            c for c in present_feature_cols if c in op_col_set
        ]

        # Normalize sensor cols per mode.
        if sensor_cols_present:
            for mode_idx in range(self.n_modes):
                mask = mode_labels == mode_idx
                if not mask.any():
                    continue
                idx = result.index[mask]
                if mode_idx in self.mode_means_:
                    means = self.mode_means_[mode_idx]
                    stds = self.mode_stds_[mode_idx]
                else:
                    means = self.global_means_
                    stds = self.global_stds_
                for col in sensor_cols_present:
                    col_mean = means.get(col, self.global_means_[col]) if hasattr(means, 'get') else means[col]
                    col_std = stds.get(col, self.global_stds_[col]) if hasattr(stds, 'get') else stds[col]
                    result.loc[idx, col] = (result.loc[idx, col] - col_mean) / col_std

        # Normalize op cols globally.
        for col in global_op_cols_present:
            result[col] = (result[col] - self.global_means_[col]) / self.global_stds_[col]

        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_payload(self) -> dict[str, object]:
        """Return a JSON/Torch-serializable normalizer payload.

        Returns:
            Payload dict with all state needed to reconstruct the normalizer.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not hasattr(self, "feature_cols_"):
            raise RuntimeError(
                "OperatingModeNormalizer must be fit before to_payload."
            )
        return {
            "schema_version": 1,
            "normalizer_type": "operating_mode",
            "feature_cols": list(self.feature_cols_),
            "op_cols": list(self.op_cols_),
            "sensor_feature_cols": list(self.sensor_feature_cols_),
            "n_modes": self.n_modes,
            "std_floor": self.std_floor,
            "random_state": self.random_state,
            "mode_centers": self.mode_centers_,
            "global_means": {
                str(k): float(v) for k, v in self.global_means_.items()
            },
            "global_stds": {
                str(k): float(v) for k, v in self.global_stds_.items()
            },
            "mode_means": {
                str(mode_idx): {str(k): float(v) for k, v in series.items()}
                for mode_idx, series in self.mode_means_.items()
            },
            "mode_stds": {
                str(mode_idx): {str(k): float(v) for k, v in series.items()}
                for mode_idx, series in self.mode_stds_.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "OperatingModeNormalizer":
        """Reconstruct a fitted normalizer from a serialized payload.

        Args:
            payload: Payload produced by ``to_payload``.

        Returns:
            Reconstructed fitted normalizer.

        Raises:
            ValueError: If the payload has an unsupported schema version,
                is missing required keys, or contains non-numeric statistics.
        """
        _require_key(payload, "schema_version")
        if payload["schema_version"] != 1:
            raise ValueError(
                f"Unsupported schema_version: {payload['schema_version']!r}. "
                "Expected 1."
            )
        required = [
            "normalizer_type",
            "feature_cols",
            "op_cols",
            "sensor_feature_cols",
            "n_modes",
            "std_floor",
            "random_state",
            "mode_centers",
            "global_means",
            "global_stds",
            "mode_means",
            "mode_stds",
        ]
        for key in required:
            _require_key(payload, key)

        n_modes = payload["n_modes"]
        std_floor = payload["std_floor"]
        random_state = payload["random_state"]
        if not isinstance(n_modes, int) or isinstance(n_modes, bool) or n_modes <= 0:
            raise ValueError(f"payload 'n_modes' must be a positive integer.")
        if not isinstance(std_floor, int | float) or isinstance(std_floor, bool) or std_floor < 0:
            raise ValueError(f"payload 'std_floor' must be a non-negative number.")
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise ValueError("payload 'random_state' must be an integer.")

        feature_cols = list(payload["feature_cols"])  # type: ignore[arg-type]
        op_cols = list(payload["op_cols"])  # type: ignore[arg-type]
        sensor_feature_cols = list(payload["sensor_feature_cols"])  # type: ignore[arg-type]

        norm = cls(
            feature_cols=feature_cols,
            op_cols=op_cols,
            n_modes=n_modes,
            std_floor=float(std_floor),
            random_state=int(random_state),
        )
        norm.feature_cols_ = feature_cols
        norm.op_cols_ = op_cols
        norm.sensor_feature_cols_ = sensor_feature_cols

        raw_centers = payload["mode_centers"]
        norm.mode_centers_ = (
            [list(c) for c in raw_centers]  # type: ignore[union-attr]
            if raw_centers is not None
            else None
        )

        norm.global_means_ = _series_from_mapping(
            payload["global_means"], feature_cols  # type: ignore[arg-type]
        )
        norm.global_stds_ = _series_from_mapping(
            payload["global_stds"], feature_cols  # type: ignore[arg-type]
        )

        raw_mode_means = payload["mode_means"]
        raw_mode_stds = payload["mode_stds"]
        if not isinstance(raw_mode_means, Mapping) or not isinstance(raw_mode_stds, Mapping):
            raise ValueError("payload 'mode_means' and 'mode_stds' must be mappings.")
        norm.mode_means_ = {
            int(k): _series_from_mapping(v, sensor_feature_cols)  # type: ignore[arg-type]
            for k, v in raw_mode_means.items()
        }
        norm.mode_stds_ = {
            int(k): _series_from_mapping(v, sensor_feature_cols)  # type: ignore[arg-type]
            for k, v in raw_mode_stds.items()
        }
        return norm

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assign_modes(self, X: pd.DataFrame) -> npt.NDArray[np.int64]:
        """Assign each row to its nearest operating mode.

        Args:
            X: DataFrame with op_cols present.

        Returns:
            Integer array of mode indices, shape ``(len(X),)``.
        """
        if self.n_modes == 1:
            return np.zeros(len(X), dtype=np.int64)
        op_values = X[self.op_cols_].to_numpy(dtype=np.float64)  # (n, n_op)
        centers = np.array(self.mode_centers_, dtype=np.float64)  # (k, n_op)
        diffs = op_values[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)  # (n, k)
        return np.argmin(distances, axis=1).astype(np.int64)


# ------------------------------------------------------------------
# Module-private helpers
# ------------------------------------------------------------------


def _require_key(payload: Mapping[str, object], key: str) -> None:
    if key not in payload:
        raise ValueError(f"Normalizer payload missing required key: {key!r}.")


def _series_from_mapping(
    mapping: Mapping[str, object], cols: list[str]
) -> "pd.Series[float]":
    values: dict[str, float] = {}
    missing: list[str] = []
    for col in cols:
        val = mapping.get(col)
        if val is None:
            missing.append(col)
            continue
        if not isinstance(val, int | float) or isinstance(val, bool):
            raise ValueError(
                f"Normalizer statistic for {col!r} must be numeric, got {type(val)}."
            )
        values[col] = float(val)
    if missing:
        raise ValueError(f"Normalizer payload missing stat columns: {missing}.")
    return pd.Series(values, dtype="float64")
```

- [ ] **Step 4: Run single-mode tests to confirm they pass**

```powershell
pytest tests/preprocessing/test_normalization.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/turbofan/preprocessing/normalization.py tests/preprocessing/test_normalization.py
git commit -m "feat(preprocessing): implement OperatingModeNormalizer single-mode path"
```

---

## Task 3 — Add multi-mode tests and verify multi-mode path

**Files:**
- Modify: `tests/preprocessing/test_normalization.py`

- [ ] **Step 1: Add multi-mode tests**

Append to `tests/preprocessing/test_normalization.py`:

```python
# ── multi-mode normalization ───────────────────────────────────────────────


def test_multi_mode_fits_mode_centers() -> None:
    """Multi-mode normalizer stores one center per mode."""
    df = _make_df_multi_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=2, random_state=42)
    norm.fit(df)
    assert norm.mode_centers_ is not None
    assert len(norm.mode_centers_) == 2


def test_multi_mode_normalizes_sensors_per_mode() -> None:
    """Sensor mean within each learned mode is approximately zero."""
    df = _make_df_multi_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=2, random_state=42)
    result = norm.fit_transform(df)
    # The two modes should be well-separated; within-mode mean near zero.
    mode0 = result[result["op_1"] == 1.0]["s_1"]
    mode1 = result[result["op_1"] == 10.0]["s_1"]
    assert abs(mode0.mean()) < 0.2
    assert abs(mode1.mean()) < 0.2


def test_multi_mode_uses_global_fallback_for_unseen_mode() -> None:
    """Transform with an op-setting not matching any mode uses global stats."""
    train = _make_df_multi_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=2, random_state=42)
    norm.fit(train)
    # op_1=5.5 is midway between the two training modes.
    test = pd.DataFrame(
        {
            "engine_id": [99],
            "cycle": [1],
            "op_1": [5.5],
            "op_2": [0.0],
            "op_3": [0.0],
            "s_1": [150.0],
        }
    )
    result = norm.transform(test)
    # Result should be a real number — not NaN or Inf.
    assert not result["s_1"].isna().any()
    assert not np.isinf(result["s_1"].to_numpy()).any()


def test_n_modes_exceeds_rows_raises_value_error() -> None:
    """fit raises ValueError when n_modes > number of training rows."""
    df = _make_df_single_mode()  # 4 rows
    norm = OperatingModeNormalizer(n_modes=10)
    with pytest.raises(ValueError, match="n_modes"):
        norm.fit(df)
```

- [ ] **Step 2: Run multi-mode tests**

```powershell
pytest tests/preprocessing/test_normalization.py -v -k "multi_mode or unseen_mode or exceeds_rows"
```

Expected: All 4 PASSED.

- [ ] **Step 3: Commit**

```powershell
git add tests/preprocessing/test_normalization.py
git commit -m "test(preprocessing): add multi-mode normalization tests"
```

---

## Task 4 — Add `to_payload` / `from_payload` tests

**Files:**
- Modify: `tests/preprocessing/test_normalization.py`

- [ ] **Step 1: Add round-trip tests**

Append to `tests/preprocessing/test_normalization.py`:

```python
# ── payload serialization ──────────────────────────────────────────────────


def test_payload_has_schema_version_one_and_type() -> None:
    """to_payload always sets schema_version=1 and normalizer_type."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1", "s_2"], n_modes=1)
    norm.fit(df)
    payload = norm.to_payload()
    assert payload["schema_version"] == 1
    assert payload["normalizer_type"] == "operating_mode"


def test_single_mode_round_trip_produces_identical_transforms() -> None:
    """to_payload → from_payload → transform matches original transform."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1", "s_2"], n_modes=1)
    norm.fit(df)
    restored = OperatingModeNormalizer.from_payload(norm.to_payload())
    pd.testing.assert_frame_equal(norm.transform(df), restored.transform(df))


def test_multi_mode_round_trip_produces_identical_transforms() -> None:
    """Multi-mode to_payload → from_payload → transform matches original."""
    df = _make_df_multi_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=2, random_state=42)
    norm.fit(df)
    restored = OperatingModeNormalizer.from_payload(norm.to_payload())
    pd.testing.assert_frame_equal(norm.transform(df), restored.transform(df))


def test_from_payload_unsupported_schema_version_raises() -> None:
    """from_payload raises ValueError for schema_version != 1."""
    with pytest.raises(ValueError, match="schema_version"):
        OperatingModeNormalizer.from_payload(
            {"schema_version": 99, "normalizer_type": "operating_mode"}
        )


def test_from_payload_missing_required_key_raises() -> None:
    """from_payload raises ValueError when a required key is absent."""
    with pytest.raises(ValueError, match="missing required key"):
        OperatingModeNormalizer.from_payload({})


def test_from_payload_non_numeric_stat_raises() -> None:
    """from_payload raises ValueError when a statistic value is not numeric."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=1)
    norm.fit(df)
    payload = norm.to_payload()
    bad: dict[str, object] = dict(payload)
    bad["global_means"] = {"s_1": "not-a-number"}
    with pytest.raises(ValueError, match="numeric"):
        OperatingModeNormalizer.from_payload(bad)
```

- [ ] **Step 2: Run payload tests**

```powershell
pytest tests/preprocessing/test_normalization.py -v -k "payload or round_trip"
```

Expected: All PASSED.

- [ ] **Step 3: Run full preprocessing test suite**

```powershell
pytest tests/preprocessing/ -v
```

Expected: All PASSED.

- [ ] **Step 4: Commit**

```powershell
git add tests/preprocessing/test_normalization.py
git commit -m "test(preprocessing): add to_payload/from_payload round-trip tests"
```

---

## Task 5 — Replace `OperationalNormalizer` in feature pipeline

**Files:**
- Modify: `src/turbofan/features/pipeline.py`
- Modify: `src/turbofan/features/normalizer.py`
- Modify: `tests/features/test_normalizer.py`
- Modify: `tests/features/test_pipeline.py`

- [ ] **Step 1: Delete `tests/features/test_normalizer.py` content**

The `OperationalNormalizer` class is being removed; its tests become stale. Replace the entire file:

```python
# tests/features/test_normalizer.py
"""turbofan.features.normalizer is now empty; no tests needed."""
```

- [ ] **Step 2: Update `src/turbofan/features/normalizer.py`**

Replace the entire file content:

```python
"""Deprecated normalizer module — use turbofan.preprocessing.normalization instead."""
```

- [ ] **Step 3: Update `src/turbofan/features/pipeline.py`**

Replace the entire file:

```python
"""Feature engineering pipeline factory."""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from turbofan.features.rolling import RollingFeatureExtractor
from turbofan.features.sensor_dropper import SensorDropper
from turbofan.preprocessing.normalization import OperatingModeNormalizer


def build_feature_pipeline(
    windows: list[int] | None = None,
    op_cols: list[str] | None = None,
    sensor_std_threshold: float = 0.0,
    sensor_keep: list[str] | None = None,
    n_modes: int = 1,
    random_state: int = 42,
) -> Pipeline:
    """Build an unfitted feature engineering pipeline.

    Returns an sklearn Pipeline with three named steps:
    ``sensor_dropper``, ``rolling_features``, ``normalizer``.

    Args:
        windows: Rolling window sizes. Default ``[5, 10, 20]``.
        op_cols: Operational setting columns.
            Default ``["op_1", "op_2", "op_3"]``.
        sensor_std_threshold: Maximum training standard deviation at
            which sensor columns are dropped.
        sensor_keep: Sensor columns to force-keep even when low-variance.
        n_modes: Number of operating modes for the normalizer.  Derived from
            ``fd_subset`` by the caller via ``mode_count_for_subset``.
        random_state: KMeans random seed for the normalizer.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline(
        [
            (
                "sensor_dropper",
                SensorDropper(
                    std_threshold=sensor_std_threshold,
                    keep=sensor_keep,
                ),
            ),
            (
                "rolling_features",
                RollingFeatureExtractor(windows=windows),
            ),
            (
                "normalizer",
                OperatingModeNormalizer(
                    op_cols=op_cols,
                    n_modes=n_modes,
                    random_state=random_state,
                ),
            ),
        ]
    )
```

- [ ] **Step 4: Add pipeline test for `n_modes`/`random_state` forwarding**

Append to `tests/features/test_pipeline.py`:

```python
def test_pipeline_normalizer_is_operating_mode_normalizer() -> None:
    """Pipeline normalizer step is OperatingModeNormalizer."""
    from turbofan.preprocessing.normalization import OperatingModeNormalizer

    pipe = build_feature_pipeline(n_modes=2, random_state=7)
    assert isinstance(pipe.named_steps["normalizer"], OperatingModeNormalizer)
    assert pipe.named_steps["normalizer"].n_modes == 2
    assert pipe.named_steps["normalizer"].random_state == 7
```

- [ ] **Step 5: Run impacted tests**

```powershell
pytest tests/features/ -v
```

Expected: All PASSED. (The old `test_normalizer.py` now has no tests; `test_pipeline.py` still passes because the API is backward-compatible for n_modes=1.)

- [ ] **Step 6: Commit**

```powershell
git add src/turbofan/features/pipeline.py src/turbofan/features/normalizer.py tests/features/test_normalizer.py tests/features/test_pipeline.py
git commit -m "feat(features): replace OperationalNormalizer with OperatingModeNormalizer in pipeline"
```

---

## Task 6 — Update baseline pipeline and training CLI

**Files:**
- Modify: `src/turbofan/models/baseline.py`
- Modify: `src/turbofan/cli/train_baseline.py`
- Modify: `tests/models/test_train_baseline_cli.py`

- [ ] **Step 1: Update `build_baseline_pipeline` to accept `n_modes`/`random_state`**

In `src/turbofan/models/baseline.py`, update `build_baseline_pipeline`:

```python
def build_baseline_pipeline(
    model_name: Literal["ridge"] = "ridge",
    alpha: float = 100.0,
    windows: list[int] | None = None,
    op_cols: list[str] | None = None,
    sensor_std_threshold: float = 0.0,
    sensor_keep: list[str] | None = None,
    feature_set: BaselineFeatureSet = "rolling",
    n_modes: int = 1,
    random_state: int = 42,
) -> Pipeline:
    """Build an unfitted feature-plus-regressor sklearn Pipeline.

    Args:
        model_name: Baseline model identifier. Only ``"ridge"`` is supported.
        alpha: Ridge regularization strength.
        windows: Rolling window sizes for the feature pipeline.
            Default ``[10]`` for the PHM08-selected baseline.
        op_cols: Operational setting columns for normalization.
        sensor_std_threshold: Maximum training standard deviation at
            which sensor columns are dropped.
        sensor_keep: Sensor columns to force-keep even when low-variance.
        feature_set: Sensor-derived feature family to expose to the estimator.
        n_modes: Operating-mode count for ``OperatingModeNormalizer``.
            Derived from ``fd_subset`` by the caller via
            ``mode_count_for_subset``.
        random_state: KMeans random seed for the normalizer.

    Returns:
        Unfitted sklearn Pipeline with feature engineering, identifier
        dropping, imputation, scaling, and model steps.

    Raises:
        ValueError: If ``model_name`` or ``feature_set`` is unsupported.
    """
    if model_name != "ridge":
        raise ValueError(f"Unsupported model: {model_name}")
    if feature_set not in {"raw", "raw_plus_rolling", "rolling"}:
        raise ValueError(f"Unsupported feature_set: {feature_set}")
    effective_windows = (
        list(DEFAULT_BASELINE_WINDOWS) if windows is None else windows
    )
    return Pipeline(
        [
            (
                "features",
                build_feature_pipeline(
                    windows=effective_windows,
                    op_cols=op_cols,
                    sensor_std_threshold=sensor_std_threshold,
                    sensor_keep=sensor_keep,
                    n_modes=n_modes,
                    random_state=random_state,
                ),
            ),
            (
                "drop_identifiers",
                FunctionTransformer(_drop_identifier_columns, validate=False),
            ),
            (
                "select_model_features",
                _ModelFeatureSelector(feature_set=feature_set),
            ),
            ("low_variance_filter", _LowVarianceFeatureDropper()),
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ).set_output(transform="pandas"),
            ),
            (
                "scaler",
                StandardScaler().set_output(transform="pandas"),
            ),
            ("model", Ridge(alpha=alpha)),
        ]
    )
```

- [ ] **Step 2: Update `src/turbofan/cli/train_baseline.py` to derive `n_modes`**

At the top of the file, add the import after existing imports:

```python
from turbofan.preprocessing.normalization import mode_count_for_subset
```

In `main()`, replace the `build_baseline_pipeline` call:

```python
    estimator = build_baseline_pipeline(
        model_name=cfg.model.name,
        alpha=cfg.model.alpha,
        windows=cfg.model.windows,
        feature_set=cfg.model.feature_set,
        sensor_std_threshold=cfg.features.sensor_std_threshold,
        sensor_keep=cfg.features.sensor_keep,
        n_modes=mode_count_for_subset(cfg.data.fd_subset),
        random_state=cfg.data.random_seed,
    )
```

- [ ] **Step 3: Add test assertion for `OperatingModeNormalizer` in baseline test**

In `tests/models/test_train_baseline_cli.py`, in `test_train_baseline_cli_writes_artifacts`, after loading the estimator, add assertions:

```python
    from turbofan.preprocessing.normalization import OperatingModeNormalizer

    normalizer = estimator.named_steps["features"].named_steps["normalizer"]
    assert isinstance(normalizer, OperatingModeNormalizer)
    # FD001 → n_modes=1
    assert normalizer.n_modes == 1
    assert normalizer.random_state == 42
```

(Insert this block after the existing `rolling = ...` and `selector = ...` assertions at line ~142.)

- [ ] **Step 4: Run baseline tests**

```powershell
pytest tests/models/test_train_baseline_cli.py -v
```

Expected: All PASSED.

- [ ] **Step 5: Commit**

```powershell
git add src/turbofan/models/baseline.py src/turbofan/cli/train_baseline.py tests/models/test_train_baseline_cli.py
git commit -m "feat(baseline): derive operating-mode count from fd_subset in baseline training"
```

---

## Task 7 — Update GRU training CLI

**Files:**
- Modify: `src/turbofan/cli/train_sequence_gru.py`
- Modify: `tests/models/test_train_sequence_gru_cli.py`

- [ ] **Step 1: Update `src/turbofan/cli/train_sequence_gru.py`**

Replace the import of `SequenceNormalizer` and `default_feature_cols` with imports from the new module. In the imports section, remove:

```python
from turbofan.sequences.normalize import SequenceNormalizer, default_feature_cols
```

And add:

```python
from turbofan.preprocessing.normalization import (
    OperatingModeNormalizer,
    mode_count_for_subset,
)
from turbofan.sequences.normalize import default_feature_cols
```

Update `_evaluate_official_test` signature to accept `OperatingModeNormalizer`:

```python
def _evaluate_official_test(
    cfg: ProjectConfig,
    model: GRURULRegressor,
    normalizer: OperatingModeNormalizer,
    feature_cols: list[str],
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame] | None:
```

Update `_model_payload` to write the new payload format:

```python
def _model_payload(
    model: GRURULRegressor,
    cfg: ProjectConfig,
    feature_cols: list[str],
    normalizer: OperatingModeNormalizer,
) -> dict[str, object]:
    """Build the serialized model checkpoint payload.

    Args:
        model: Trained GRU model.
        cfg: Project config.
        feature_cols: Feature columns used by the model.
        normalizer: Fitted operating-mode normalizer.

    Returns:
        Torch-serializable model payload.
    """
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

In `main()`, replace `SequenceNormalizer` construction:

```python
    normalizer = OperatingModeNormalizer(
        feature_cols=feature_cols,
        n_modes=mode_count_for_subset(cfg.data.fd_subset),
        random_state=cfg.data.random_seed,
    )
    train_normalized = normalizer.fit_transform(train_df)
    val_normalized = normalizer.transform(val_df)
```

- [ ] **Step 2: Update `tests/models/test_train_sequence_gru_cli.py`**

Replace `_FakeNormalizer` class with one that has a `to_payload` method:

```python
class _FakeNormalizer:
    """Minimal operating-mode normalizer test double."""

    def __init__(self, feature_cols: list[str], **kwargs: object) -> None:
        self.feature_cols = feature_cols

    def fit_transform(self, frame: object) -> object:
        """Return the input training frame unchanged."""
        return frame

    def transform(self, frame: object) -> object:
        """Return the input frame unchanged."""
        return frame

    def to_payload(self) -> dict[str, object]:
        """Return a minimal normalizer payload."""
        return {
            "schema_version": 1,
            "normalizer_type": "operating_mode",
            "feature_cols": self.feature_cols,
            "op_cols": ["op_1", "op_2", "op_3"],
            "sensor_feature_cols": self.feature_cols,
            "n_modes": 1,
            "std_floor": 1e-3,
            "random_state": 42,
            "mode_centers": None,
            "global_means": {col: 0.0 for col in self.feature_cols},
            "global_stds": {col: 1.0 for col in self.feature_cols},
            "mode_means": {"0": {col: 0.0 for col in self.feature_cols}},
            "mode_stds": {"0": {col: 1.0 for col in self.feature_cols}},
        }
```

In `test_train_sequence_gru_cli_seeds_model_initialization`, change:

```python
    monkeypatch.setattr(module, "SequenceNormalizer", _FakeNormalizer)
```

to:

```python
    monkeypatch.setattr(module, "OperatingModeNormalizer", _FakeNormalizer)
```

In `test_train_sequence_gru_cli_appends_training_log_entry`, apply the same change.

Add a new test that asserts the checkpoint contains the new payload format:

```python
def test_train_sequence_gru_cli_checkpoint_uses_normalizer_payload(
    tmp_path: Path,
) -> None:
    """Saved checkpoint contains normalizer_type and normalizer_payload."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)

    artifact_dir = tmp_path / "artifacts"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_dir, artifact_dir, tmp_path)

    _run_cli(cfg_path)

    run_dir = next((artifact_dir / "sequence_gru").iterdir())
    payload = torch.load(run_dir / "model.pt", map_location="cpu")
    assert payload["normalizer_type"] == "operating_mode"
    assert "normalizer_payload" in payload
    assert payload["normalizer_payload"]["schema_version"] == 1
    assert "normalizer_means" not in payload
    assert "normalizer_stds" not in payload


def test_train_sequence_gru_cli_uses_subset_derived_mode_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GRU training constructs OperatingModeNormalizer with subset mode count."""
    module = _load_train_sequence_gru_module()
    captured: list[dict[str, object]] = []

    class _CapturingNormalizer(_FakeNormalizer):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(feature_cols=kwargs.get("feature_cols", []))  # type: ignore[arg-type]
            captured.append(dict(kwargs))

    cfg = ProjectConfig(
        project_name="test",
        data=DataConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            interim_dir=tmp_path / "interim",
            fd_subset="FD001",
            random_seed=77,
        ),
        sequence=SequenceConfig(
            architecture="gru",
            window_size=3,
            batch_size=4,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            epochs=1,
            artifact_dir=tmp_path / "artifacts",
        ),
    )

    def fake_train(**kwargs: object) -> TrainingResult:
        return TrainingResult(
            model=GRURULRegressor(input_size=2, hidden_size=4, num_layers=1, dropout=0.0),
            history=pd.DataFrame([{"epoch": 1}]),
            best_epoch=1,
            best_metric=0.0,
        )

    monkeypatch.setattr(module, "_parse_args", lambda: argparse.Namespace(config=tmp_path / "c.yaml"))
    monkeypatch.setattr(module, "load_config", lambda p: cfg)
    monkeypatch.setattr(module, "resolve_device", lambda r: torch.device("cpu"))
    monkeypatch.setattr(module, "default_feature_cols", lambda: ["s1", "s2"])
    monkeypatch.setattr(module, "load_raw_train", lambda c: object())
    monkeypatch.setattr(module, "add_rul_column", lambda f, max_rul: f)
    monkeypatch.setattr(module, "split_by_engine", lambda f, test_size, random_seed: (f, f))
    monkeypatch.setattr(module, "OperatingModeNormalizer", _CapturingNormalizer)
    monkeypatch.setattr(module, "build_sliding_windows", lambda *a, **k: object())
    monkeypatch.setattr(module, "build_sequence_loader", lambda *a, **k: object())
    monkeypatch.setattr(module, "train_gru_model", fake_train)
    monkeypatch.setattr(module, "_evaluate_windows", lambda *a, **k: ({"rmse": 0.0, "mae": 0.0, "phm08_score": 0.0}, pd.DataFrame()))
    monkeypatch.setattr(module, "_evaluate_official_test", lambda *a, **k: None)
    monkeypatch.setattr(module, "create_run_dir", lambda a, n: tmp_path)
    monkeypatch.setattr(module, "save_json", lambda p, pa: None)
    monkeypatch.setattr(module, "save_predictions", lambda f, p: None)
    monkeypatch.setattr(module.torch, "save", lambda p, pa: None)
    monkeypatch.setattr(module, "append_training_log", lambda e: None)

    module.main()

    assert captured, "OperatingModeNormalizer was never constructed"
    assert captured[0]["n_modes"] == 1  # FD001 → 1 mode
    assert captured[0]["random_state"] == 77
```

- [ ] **Step 3: Run GRU CLI tests**

```powershell
pytest tests/models/test_train_sequence_gru_cli.py -v
```

Expected: All PASSED.

- [ ] **Step 4: Commit**

```powershell
git add src/turbofan/cli/train_sequence_gru.py tests/models/test_train_sequence_gru_cli.py
git commit -m "feat(gru): replace SequenceNormalizer with OperatingModeNormalizer in GRU training"
```

---

## Task 8 — Update GRU sweep experiments

**Files:**
- Modify: `src/turbofan/experiments/sequence_gru_sweep.py`
- Modify: `src/turbofan/experiments/feature_gru_sweep.py`
- Modify: `tests/models/test_sweep_sequence_gru.py`
- Modify: `tests/models/test_sweep_feature_gru.py`

- [ ] **Step 1: Update `sequence_gru_sweep.py`**

Remove the `SequenceNormalizer` and `default_feature_cols` import:

```python
from turbofan.sequences.normalize import SequenceNormalizer, default_feature_cols
```

Replace with:

```python
from turbofan.preprocessing.normalization import (
    OperatingModeNormalizer,
    mode_count_for_subset,
)
from turbofan.sequences.normalize import default_feature_cols
```

In `run_gru_sweep`, replace:

```python
    normalizer = SequenceNormalizer(feature_cols=feature_cols)
    train_normalized = normalizer.fit_transform(train_df)
    val_normalized = normalizer.transform(val_df)
```

with:

```python
    normalizer = OperatingModeNormalizer(
        feature_cols=feature_cols,
        n_modes=mode_count_for_subset(cfg.data.fd_subset),
        random_state=cfg.data.random_seed,
    )
    train_normalized = normalizer.fit_transform(train_df)
    val_normalized = normalizer.transform(val_df)
```

- [ ] **Step 2: Update `feature_gru_sweep.py`**

Remove the `SequenceNormalizer` import:

```python
from turbofan.sequences.normalize import SequenceNormalizer
```

Replace with:

```python
from turbofan.preprocessing.normalization import (
    OperatingModeNormalizer,
    mode_count_for_subset,
)
```

In `run_feature_sweep`, replace (inside the per-run loop):

```python
        normalizer = SequenceNormalizer(feature_cols=feature_cols)
        train_normalized = normalizer.fit_transform(current_train_df)
        val_normalized = normalizer.transform(current_val_df)
```

with:

```python
        normalizer = OperatingModeNormalizer(
            feature_cols=feature_cols,
            n_modes=mode_count_for_subset(cfg.data.fd_subset),
            random_state=cfg.data.random_seed,
        )
        train_normalized = normalizer.fit_transform(current_train_df)
        val_normalized = normalizer.transform(current_val_df)
```

- [ ] **Step 3: Update sweep tests**

Read `tests/models/test_sweep_sequence_gru.py` and find where `SequenceNormalizer` is patched, then update to `OperatingModeNormalizer`. Also read `tests/models/test_sweep_feature_gru.py` for the same.

In each sweep test file, find any line like:
```python
monkeypatch.setattr(module, "SequenceNormalizer", ...)
```
and change to:
```python
monkeypatch.setattr(module, "OperatingModeNormalizer", ...)
```

Note: If the sweep test uses a fake normalizer class, it must also have a `to_payload` method (though sweeps don't save checkpoints, so this may not be needed — just ensure the constructor accepts keyword args `feature_cols`, `n_modes`, and `random_state`).

- [ ] **Step 4: Run sweep tests**

```powershell
pytest tests/models/test_sweep_sequence_gru.py tests/models/test_sweep_feature_gru.py -v
```

Expected: All PASSED.

- [ ] **Step 5: Commit**

```powershell
git add src/turbofan/experiments/sequence_gru_sweep.py src/turbofan/experiments/feature_gru_sweep.py tests/models/test_sweep_sequence_gru.py tests/models/test_sweep_feature_gru.py
git commit -m "feat(sweeps): replace SequenceNormalizer with OperatingModeNormalizer in GRU sweeps"
```

---

## Task 9 — Update GRU inference predictor

**Files:**
- Modify: `src/turbofan/inference/predictors.py`
- Modify: `tests/inference/test_predictors.py`

- [ ] **Step 1: Update `predictors.py`**

Remove the import of `SequenceNormalizer`:

```python
from turbofan.sequences.normalize import SequenceNormalizer
```

Add:

```python
from turbofan.preprocessing.normalization import OperatingModeNormalizer
```

Replace `_load_torch_payload` to require the new keys:

```python
def _load_torch_payload(path: Path) -> Mapping[str, object]:
    """Load and validate a GRU checkpoint payload.

    Args:
        path: Path to the saved ``.pt`` checkpoint.

    Returns:
        Validated checkpoint mapping.

    Raises:
        ValueError: If the payload is not a mapping, is missing required keys,
            or uses the legacy flat-stat normalizer format.
    """
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("GRU checkpoint payload must be a mapping.")
    for key in [
        "model_state_dict",
        "sequence_config",
        "feature_cols",
        "max_rul",
    ]:
        if key not in payload:
            raise ValueError(f"GRU checkpoint payload missing {key!r}.")
    if "normalizer_means" in payload or "normalizer_stds" in payload:
        raise ValueError(
            "GRU checkpoint uses a legacy flat-stat normalizer format "
            "(normalizer_means / normalizer_stds). Retrain the model with "
            "an operating-mode normalizer payload."
        )
    if "normalizer_type" not in payload or "normalizer_payload" not in payload:
        raise ValueError(
            "GRU checkpoint payload missing 'normalizer_type' or "
            "'normalizer_payload'. Retrain the model with an operating-mode "
            "normalizer payload."
        )
    if payload["normalizer_type"] != "operating_mode":
        raise ValueError(
            f"Unsupported normalizer_type: {payload['normalizer_type']!r}. "
            "Expected 'operating_mode'."
        )
    return cast(Mapping[str, object], payload)
```

Replace `_normalizer_from_payload`:

```python
def _normalizer_from_payload(
    payload: Mapping[str, object],
    feature_cols: Sequence[str],
) -> OperatingModeNormalizer:
    """Reconstruct an OperatingModeNormalizer from a checkpoint payload.

    Args:
        payload: Full checkpoint payload.
        feature_cols: Feature columns expected by the model.

    Returns:
        Reconstructed fitted normalizer.

    Raises:
        ValueError: If the normalizer payload is missing or invalid.
    """
    raw = payload.get("normalizer_payload")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "GRU checkpoint 'normalizer_payload' must be a mapping."
        )
    return OperatingModeNormalizer.from_payload(cast(Mapping[str, object], raw))
```

Remove the now-unused `_float_series` helper (it was only used by the old `_normalizer_from_payload`). Double-check no other code references it before removing.

- [ ] **Step 2: Update `tests/inference/test_predictors.py`**

Update `_gru_artifact` to write the new checkpoint format. Replace the `torch.save(...)` call:

```python
def _gru_artifact(
    tmp_path: Path, *, window_size: int = 3, max_rul: int = 125
) -> Path:
    """Create a synthetic GRU artifact with new normalizer payload format."""
    from turbofan.preprocessing.normalization import OperatingModeNormalizer

    artifact_dir = tmp_path / "gru"
    artifact_dir.mkdir()
    model = GRURULRegressor(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    model.regressor.bias.data.fill_(-2.0)

    # Build a fitted identity normalizer (means=0, stds=1).
    dummy_df = pd.DataFrame(
        {col: [0.0, 1.0] for col in FEATURE_COLUMNS},
    )
    dummy_df["engine_id"] = [1, 2]
    dummy_df["cycle"] = [1, 1]
    normalizer = OperatingModeNormalizer(
        feature_cols=list(FEATURE_COLUMNS), n_modes=1, random_state=42
    )
    normalizer.fit(dummy_df)
    # Force identity stats so the model's zero weights + negative bias remain.
    import numpy as np
    normalizer.global_means_ = pd.Series(
        {col: 0.0 for col in FEATURE_COLUMNS}, dtype="float64"
    )
    normalizer.global_stds_ = pd.Series(
        {col: 1.0 for col in FEATURE_COLUMNS}, dtype="float64"
    )
    sensor_cols = normalizer.sensor_feature_cols_
    normalizer.mode_means_ = {
        0: pd.Series({col: 0.0 for col in sensor_cols}, dtype="float64")
    }
    normalizer.mode_stds_ = {
        0: pd.Series({col: 1.0 for col in sensor_cols}, dtype="float64")
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sequence_config": {
                "architecture": "gru",
                "window_size": window_size,
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "feature_cols": FEATURE_COLUMNS,
            "normalizer_type": "operating_mode",
            "normalizer_payload": normalizer.to_payload(),
            "max_rul": max_rul,
        },
        artifact_dir / "model.pt",
    )
    return _write_manifest(
        artifact_dir,
        model_type="gru",
        artifact_id="gru-test",
        prediction_scope="final_window",
        model_path="model.pt",
    )
```

Also update `test_gru_predictor_rescales_output_by_max_rul` to use the same pattern.

Add a new test for legacy checkpoint rejection:

```python
def test_gru_predictor_rejects_legacy_flat_stat_checkpoint(tmp_path: Path) -> None:
    """GRU predictor rejects checkpoints with normalizer_means/normalizer_stds."""
    from turbofan.inference.predictors import load_predictor

    artifact_dir = tmp_path / "gru_legacy"
    artifact_dir.mkdir()
    model = GRURULRegressor(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sequence_config": {
                "architecture": "gru",
                "window_size": 3,
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "feature_cols": FEATURE_COLUMNS,
            "normalizer_means": {column: 0.0 for column in FEATURE_COLUMNS},
            "normalizer_stds": {column: 1.0 for column in FEATURE_COLUMNS},
            "max_rul": 125,
        },
        artifact_dir / "model.pt",
    )
    _write_manifest(
        artifact_dir,
        model_type="gru",
        artifact_id="gru-legacy",
        prediction_scope="final_window",
        model_path="model.pt",
    )

    with pytest.raises(ValueError, match="[Rr]etrain"):
        load_predictor(artifact_dir / "model_manifest.json")
```

Remove the old `test_gru_predictor_rejects_checkpoint_without_max_rul` test if it used the old payload format keys (update it to use the new format without `max_rul`).

- [ ] **Step 3: Run inference tests**

```powershell
pytest tests/inference/test_predictors.py -v
```

Expected: All PASSED.

- [ ] **Step 4: Commit**

```powershell
git add src/turbofan/inference/predictors.py tests/inference/test_predictors.py
git commit -m "feat(inference): require operating_mode normalizer payload; reject legacy flat-stat checkpoints"
```

---

## Task 10 — Remove stale `SequenceNormalizer` tests

**Files:**
- Modify: `tests/sequences/test_normalize.py`

- [ ] **Step 1: Replace stale tests**

The spec requires removing tests that validate `SequenceNormalizer` behavior that no active path uses. Replace `tests/sequences/test_normalize.py` with a file that only keeps `default_feature_cols` tests and adds a test proving `OperatingModeNormalizer` works as a drop-in for sequence code:

```python
"""Tests for turbofan.sequences.normalize and sequence-facing normalizer usage."""
from __future__ import annotations

import pandas as pd

from turbofan.sequences.normalize import default_feature_cols


def test_default_feature_cols_returns_ops_and_sensors() -> None:
    """Default feature columns include operating settings and all 21 sensors."""
    expected = ["op_1", "op_2", "op_3"] + [f"s_{idx}" for idx in range(1, 22)]
    result = default_feature_cols()
    assert result == expected
    assert len(result) == 24


def test_operating_mode_normalizer_usable_with_explicit_sequence_feature_cols() -> None:
    """OperatingModeNormalizer with explicit feature_cols works for sequence data."""
    from turbofan.preprocessing.normalization import OperatingModeNormalizer

    feature_cols = default_feature_cols()
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "op_1": [0.0] * 4,
            "op_2": [0.0] * 4,
            "op_3": [0.0] * 4,
            **{f"s_{i}": [float(i)] * 4 for i in range(1, 22)},
        }
    )
    norm = OperatingModeNormalizer(feature_cols=feature_cols, n_modes=1)
    result = norm.fit_transform(df)
    assert set(result.columns) >= {"engine_id", "cycle", "op_1", "s_1"}
    assert not result[feature_cols].isna().any().any()
```

- [ ] **Step 2: Run the updated test file**

```powershell
pytest tests/sequences/test_normalize.py -v
```

Expected: 2 PASSED.

- [ ] **Step 3: Commit**

```powershell
git add tests/sequences/test_normalize.py
git commit -m "test(sequences): remove stale SequenceNormalizer tests; add OperatingModeNormalizer sequence smoke test"
```

---

## Task 11 — Final verification

- [ ] **Step 1: Run lint**

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
ruff check src/ tests/
```

Expected: No errors. Fix any ruff errors before proceeding.

- [ ] **Step 2: Run type-checker**

```powershell
mypy src/turbofan
```

Expected: No errors. Pay attention to:
- `OperatingModeNormalizer` `from_payload` return type (use `"OperatingModeNormalizer"` string annotation or import `Self`)
- `mode_means_` / `mode_stds_` typed as `dict[int, pd.Series[float]]` — mypy may need explicit `pd.Series[Any]`
- Any `cast` calls needed for payload dict access

Fix errors inline until mypy is clean.

- [ ] **Step 3: Run full test suite**

```powershell
pytest
```

Expected: All PASSED.

- [ ] **Step 4: Final commit**

```powershell
git add -u
git commit -m "chore: fix lint and type errors from operating-mode normalization migration"
```

---

## Self-Review Checklist

### Spec coverage
- [x] `CMAPSS_SUBSET_MODE_COUNTS` dict with FD001=1, FD002=6, FD003=1, FD004=6 → Task 1
- [x] `mode_count_for_subset` raises ValueError for unsupported subset → Task 1
- [x] `OperatingModeNormalizer` constructor with all parameters → Task 2
- [x] `feature_cols=None` infers numeric cols excluding engine_id/cycle/rul → Task 2
- [x] Single-mode path: no KMeans, all rows mode 0 → Task 2
- [x] Multi-mode path: KMeans with n_init=10, random_state → Task 3
- [x] Per-mode mean/std for sensor_feature_cols_ (ddof=0) → Task 2/3
- [x] Global mean/std for all feature_cols_ → Task 2
- [x] std_floor replacement for near-zero std → Task 2
- [x] Mode fallback to global stats when mode missing at transform → Task 2/3
- [x] `to_payload` / `from_payload` round-trip → Task 4
- [x] `from_payload` rejects bad schema_version, missing keys, non-numeric stats → Task 4
- [x] Baseline pipeline updated → Task 5/6
- [x] `train_baseline.py` derives n_modes from `mode_count_for_subset` → Task 6
- [x] GRU training uses `OperatingModeNormalizer` → Task 7
- [x] GRU checkpoint writes `normalizer_type`/`normalizer_payload` → Task 7
- [x] GRU sweeps updated → Task 8
- [x] Inference rejects legacy flat-stat checkpoints with retrain message → Task 9
- [x] Inference reconstructs from new payload format → Task 9
- [x] Stale `SequenceNormalizer` tests removed → Task 10
- [x] `OperationalNormalizer` deleted from active code → Task 5
- [x] ruff, mypy strict, pytest all pass → Task 11

### Placeholder scan
No placeholders found.

### Type consistency
- `OperatingModeNormalizer` is consistently named across all tasks.
- `to_payload()` → `dict[str, object]` used consistently.
- `from_payload(cls, payload: Mapping[str, object]) -> Self` used consistently.
- `mode_means_: dict[int, pd.Series[float]]` — note: mypy may require `pd.Series[Any]` in practice.
- `global_means_: pd.Series[float]` matches usage in `to_payload` (`.items()`).
