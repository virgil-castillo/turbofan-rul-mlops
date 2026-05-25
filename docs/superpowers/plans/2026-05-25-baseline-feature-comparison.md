# Baseline Feature Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parallel CLI that compares Ridge baselines across raw, raw-plus-rolling, and rolling-only sensor feature sets.

**Architecture:** Add a model feature selector inside `src/turbofan/models/baseline.py` so every baseline pipeline can consistently drop `engine_id`, `cycle`, and `op_*` before Ridge. Add `scripts/compare_baseline_features.py` as a thin orchestration layer that builds independent experiment specs, evaluates them with `joblib.Parallel`, and returns a sorted results DataFrame.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn Pipeline, joblib, pytest, ruff, mypy.

---

## File Structure

- Modify `src/turbofan/models/baseline.py`: define `BaselineFeatureSet`, add feature-family selection, and pass the feature-set into `build_baseline_pipeline()`.
- Modify `tests/models/test_baseline.py`: add tests for raw, raw-plus-rolling, rolling-only, and unsupported feature-set behavior.
- Create `scripts/compare_baseline_features.py`: CLI and testable helper functions for feature comparison.
- Create `tests/models/test_compare_baseline_features_cli.py`: helper and subprocess smoke tests for the comparison script.

## Task 1: Baseline Feature-Set Selection

**Files:**
- Modify: `tests/models/test_baseline.py`
- Modify: `src/turbofan/models/baseline.py`

- [ ] **Step 1: Write failing tests for estimator feature selection**

Add these tests to `tests/models/test_baseline.py`:

```python
def _fit_columns(feature_set: str) -> set[str]:
    """Return Ridge feature names for a fitted baseline feature set."""
    X, y = _make_df()
    pipe = build_baseline_pipeline(
        windows=[3],
        feature_set=feature_set,  # type: ignore[arg-type]
    )
    pipe.fit(X, y)
    model = pipe.named_steps["model"]
    return set(model.feature_names_in_)


def test_raw_feature_set_exposes_only_raw_sensor_features() -> None:
    """Raw feature set excludes identifiers, cycle, op columns, and rolling."""
    columns = _fit_columns("raw")

    assert {"engine_id", "cycle", "op_1", "op_2", "op_3"}.isdisjoint(columns)
    assert {"s_1", "s_3"}.issubset(columns)
    assert all("_rmean_" not in column for column in columns)


def test_raw_plus_rolling_feature_set_exposes_raw_and_rolling_features() -> None:
    """Raw plus rolling feature set keeps raw and rolling sensor features."""
    columns = _fit_columns("raw_plus_rolling")

    assert {"engine_id", "cycle", "op_1", "op_2", "op_3"}.isdisjoint(columns)
    assert "s_1" in columns
    assert "s_1_rmean_3" in columns


def test_rolling_feature_set_exposes_only_rolling_sensor_features() -> None:
    """Rolling feature set drops raw sensors and keeps rolling features."""
    columns = _fit_columns("rolling")

    assert {"engine_id", "cycle", "op_1", "op_2", "op_3"}.isdisjoint(columns)
    assert "s_1" not in columns
    assert "s_1_rmean_3" in columns


def test_unknown_feature_set_raises() -> None:
    """Unsupported feature-set names fail fast."""
    with pytest.raises(ValueError, match="Unsupported feature_set"):
        build_baseline_pipeline(feature_set="bad")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
pytest -p no:cacheprovider --basetemp=.pytest-basetemp-codex tests/models/test_baseline.py -q
```

Expected: fail because `build_baseline_pipeline()` does not accept
`feature_set`.

- [ ] **Step 3: Implement feature selection**

In `src/turbofan/models/baseline.py`, add a literal type and selector:

```python
BaselineFeatureSet = Literal["raw", "raw_plus_rolling", "rolling"]

ROLLING_MARKERS = ("_rmean_", "_rstd_", "_rmin_", "_rmax_")


def _is_rolling_feature(column: str) -> bool:
    """Return whether a column is a rolling sensor feature."""
    return column.startswith("s_") and any(
        marker in column for marker in ROLLING_MARKERS
    )


def _is_raw_sensor_feature(column: str) -> bool:
    """Return whether a column is a raw sensor feature."""
    return column.startswith("s_") and not _is_rolling_feature(column)


class _ModelFeatureSelector(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Select estimator-facing columns for a baseline feature family."""

    def __init__(self, feature_set: BaselineFeatureSet = "raw_plus_rolling") -> None:
        self.feature_set = feature_set

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Validate and store selected model feature columns.

        Args:
            X: Engineered feature matrix.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Fitted selector.

        Raises:
            ValueError: If the feature set is unsupported or empty.
        """
        if self.feature_set not in {"raw", "raw_plus_rolling", "rolling"}:
            raise ValueError(f"Unsupported feature_set: {self.feature_set}")
        self.columns_: list[str] = [
            column for column in X.columns if self._should_keep(column)
        ]
        if not self.columns_:
            raise ValueError(
                f"Feature set {self.feature_set!r} produced no model features."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return estimator-facing feature columns.

        Args:
            X: Engineered feature matrix.

        Returns:
            DataFrame containing only selected feature columns.
        """
        return X.loc[:, [column for column in self.columns_ if column in X.columns]]

    def _should_keep(self, column: str) -> bool:
        if self.feature_set == "raw":
            return _is_raw_sensor_feature(column)
        if self.feature_set == "rolling":
            return _is_rolling_feature(column)
        return _is_raw_sensor_feature(column) or _is_rolling_feature(column)
```

Update `build_baseline_pipeline()`:

```python
def build_baseline_pipeline(
    model_name: Literal["ridge"] = "ridge",
    alpha: float = 100.0,
    windows: list[int] | None = None,
    op_cols: list[str] | None = None,
    sensor_std_threshold: float = 0.0,
    sensor_keep: list[str] | None = None,
    feature_set: BaselineFeatureSet = "raw_plus_rolling",
) -> Pipeline:
    ...
    ("select_model_features", _ModelFeatureSelector(feature_set=feature_set)),
    ("low_variance_filter", _LowVarianceFeatureDropper()),
    ...
```

Update the named-step test to include `"select_model_features"` between
`"drop_identifiers"` and `"low_variance_filter"`. Update existing assertions
that currently expect `cycle` to remain present before Ridge so they now expect
`cycle` to be absent.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
pytest -p no:cacheprovider --basetemp=.pytest-basetemp-codex tests/models/test_baseline.py -q
```

Expected: all `test_baseline.py` tests pass.

## Task 2: Comparison Script Helper API

**Files:**
- Create: `tests/models/test_compare_baseline_features_cli.py`
- Create: `scripts/compare_baseline_features.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/models/test_compare_baseline_features_cli.py` with imports,
module loader, synthetic C-MAPSS writer, and tests:

```python
"""Tests for scripts/compare_baseline_features.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def _load_module(project_root: Path) -> ModuleType:
    """Load the comparison script as a module for helper testing."""
    script_path = project_root / "scripts" / "compare_baseline_features.py"
    spec = importlib.util.spec_from_file_location(
        "compare_baseline_features",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cmapps_file(path: Path, n_engines: int, n_cycles: int) -> None:
    """Write a small C-MAPSS-style whitespace-delimited file."""
    lines = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, n_cycles + 1):
            op_cols = [0.0, 0.0, 0.0]
            sensors = [float(cycle + i + engine_id) for i in range(1, 22)]
            values = [engine_id, cycle, *op_cols, *sensors]
            lines.append(" ".join(str(value) for value in values))
    path.write_text("\n".join(lines))


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal project config for comparison tests."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=8)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "project_name: test",
                "data:",
                f"  raw_dir: {raw_dir.as_posix()}",
                f"  processed_dir: {(tmp_path / 'processed').as_posix()}",
                f"  interim_dir: {(tmp_path / 'interim').as_posix()}",
                "  fd_subset: FD001",
                "  max_rul: 30",
                "  test_size: 0.25",
                "  random_seed: 42",
                "model:",
                "  name: ridge",
                "  alpha: 1.0",
            ]
        )
    )
    return cfg_path


def test_feature_comparison_returns_expected_rows(tmp_path: Path) -> None:
    """Comparison helper evaluates raw once and rolling variants per window."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)

    results = module.run_feature_comparison(
        config_path=cfg_path,
        feature_sets=["raw", "raw_plus_rolling", "rolling"],
        windows=[3, 5],
        n_jobs=1,
    )

    assert len(results) == 5
    observed_specs = {
        (row.feature_set, row.windows) for row in results.itertuples(index=False)
    }
    assert observed_specs == {
        ("raw", ""),
        ("raw_plus_rolling", "3"),
        ("rolling", "3"),
        ("raw_plus_rolling", "5"),
        ("rolling", "5"),
    }
    assert set(results.columns) == {
        "feature_set",
        "windows",
        "alpha",
        "n_features",
        "raw_prediction_min",
        "raw_prediction_max",
        "rmse",
        "mae",
        "phm08_score",
    }


def test_feature_comparison_validates_inputs(tmp_path: Path) -> None:
    """Invalid comparison inputs fail before training."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)

    with pytest.raises(ValueError, match="positive"):
        module.run_feature_comparison(cfg_path, ["raw"], [0], n_jobs=1)
    with pytest.raises(ValueError, match="n_jobs"):
        module.run_feature_comparison(cfg_path, ["raw"], [3], n_jobs=0)
    with pytest.raises(ValueError, match="Unsupported feature_set"):
        module.run_feature_comparison(cfg_path, ["bad"], [3], n_jobs=1)
```

- [ ] **Step 2: Run helper tests to verify RED**

Run:

```powershell
pytest -p no:cacheprovider --basetemp=.pytest-basetemp-codex tests/models/test_compare_baseline_features_cli.py -q
```

Expected: fail because `scripts/compare_baseline_features.py` does not exist.

- [ ] **Step 3: Implement comparison helper and CLI**

Create `scripts/compare_baseline_features.py` with:

- `ExperimentSpec` dataclass containing `feature_set: BaselineFeatureSet` and
  `windows: tuple[int, ...]`.
- `_build_experiment_specs(feature_sets, windows)` that emits `raw` once with
  empty windows and each rolling family once per window.
- `_evaluate_spec(...)` that fits `build_baseline_pipeline(...,
  feature_set=spec.feature_set, windows=list(spec.windows) or [])`, clips
  predictions to `[0, max_rul]`, computes metrics, and records `n_features`
  from `estimator.named_steps["model"].feature_names_in_`.
- `run_feature_comparison(config_path, feature_sets, windows, n_jobs)` that
  validates inputs, loads/splits data once, evaluates specs with
  `joblib.Parallel`, and returns a DataFrame sorted by `rmse`.
- `main()` that prints `results.to_string(index=False)` and writes CSV if
  `--output` is provided.

- [ ] **Step 4: Run helper tests to verify GREEN**

Run:

```powershell
pytest -p no:cacheprovider --basetemp=.pytest-basetemp-codex tests/models/test_compare_baseline_features_cli.py -q
```

Expected: all helper tests pass.

## Task 3: CLI Smoke Test and Final Verification

**Files:**
- Modify: `tests/models/test_compare_baseline_features_cli.py`

- [ ] **Step 1: Add subprocess smoke test**

Append a CLI test:

```python
def test_compare_baseline_features_cli_writes_csv(tmp_path: Path) -> None:
    """CLI writes a sorted comparison CSV when output is supplied."""
    import os
    import subprocess
    import sys

    project_root = Path(__file__).parent.parent.parent
    cfg_path = _write_config(tmp_path)
    output_path = tmp_path / "comparison.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_baseline_features.py",
            "--config",
            str(cfg_path),
            "--feature-sets",
            "raw",
            "raw_plus_rolling",
            "rolling",
            "--windows",
            "3",
            "--n-jobs",
            "1",
            "--output",
            str(output_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "feature_set" in result.stdout
    results = pd.read_csv(output_path)
    assert len(results) == 3
    assert results["rmse"].is_monotonic_increasing
```

- [ ] **Step 2: Run CLI test after adding the smoke test**

Run:

```powershell
pytest -p no:cacheprovider --basetemp=.pytest-basetemp-codex tests/models/test_compare_baseline_features_cli.py -q
```

Expected: pass after Task 2 implementation because the subprocess exercises the
same script entry point.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
pytest -p no:cacheprovider --basetemp=.pytest-basetemp-codex tests/models/test_baseline.py tests/models/test_compare_baseline_features_cli.py -q
```

Expected: all focused tests pass.

- [ ] **Step 4: Run lint and type checks**

Run:

```powershell
ruff check src/ tests/ scripts/
mypy src/turbofan
```

Expected: both commands pass.

- [ ] **Step 5: Review git diff**

Run:

```powershell
git -c safe.directory=C:/Users/Virgil/Code/turbofan-rul-mlops diff --stat
git -c safe.directory=C:/Users/Virgil/Code/turbofan-rul-mlops diff -- scripts/compare_baseline_features.py src/turbofan/models/baseline.py
```

Expected: only baseline feature selection, comparison script, tests, and this
plan are changed.
