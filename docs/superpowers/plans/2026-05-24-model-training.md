# Model Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tabular Ridge baseline training workflow for C-MAPSS RUL prediction using the existing feature pipeline.

**Architecture:** Add a focused `turbofan.models` package for engine-level splitting, metrics, baseline pipeline construction, evaluation helpers, and local artifact persistence. Keep `scripts/train_baseline.py` as a thin CLI that loads config, trains the sklearn pipeline, evaluates validation and optional official test metrics, then writes local artifacts.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, joblib, pydantic, pytest, ruff, mypy

---

## Environment

Run Python commands from PowerShell with the project conda environment active:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
```

Use these verification commands before final handoff:

```powershell
ruff check src/ tests/ scripts/
mypy src/turbofan
pytest
```

---

## File Map

| File | Responsibility |
|------|----------------|
| `.gitignore` | Ignore generated model artifact directories |
| `pyproject.toml` | Add direct `joblib` runtime dependency |
| `configs/default.yaml` | Add default baseline model config |
| `src/turbofan/config/schema.py` | Add `ModelConfig` and attach it to `ProjectConfig` |
| `tests/config/test_schema.py` | Cover default and custom model config loading |
| `src/turbofan/models/__init__.py` | Model-training subpackage marker |
| `src/turbofan/models/split.py` | Engine-level train/validation split |
| `src/turbofan/models/metrics.py` | RMSE, MAE, PHM08, combined metrics |
| `src/turbofan/models/baseline.py` | Feature-plus-Ridge sklearn pipeline factory |
| `src/turbofan/models/evaluate.py` | RUL column, feature/target split, row/test evaluation helpers |
| `src/turbofan/models/artifacts.py` | Local run directory, joblib, JSON, and prediction CSV persistence |
| `scripts/train_baseline.py` | CLI orchestration for one baseline run |
| `tests/models/*.py` | Synthetic unit and smoke tests for all model-training behavior |

---

## Task 1: Config, dependency, gitignore, and package scaffolding

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Modify: `configs/default.yaml`
- Modify: `src/turbofan/config/schema.py`
- Modify: `tests/config/test_schema.py`
- Create: `src/turbofan/models/__init__.py`
- Create: `tests/models/__init__.py`

- [ ] **Step 1: Write failing config tests**

Append these tests to `tests/config/test_schema.py`:

```python
def test_model_config_defaults_when_section_omitted(tmp_path: Path) -> None:
    """Model config has stable defaults when omitted from YAML."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
        },
    )
    cfg = load_config(cfg_file)
    assert cfg.model.name == "ridge"
    assert cfg.model.alpha == 1.0
    assert cfg.model.artifact_dir == Path("artifacts/models")


def test_model_config_loads_custom_values(tmp_path: Path) -> None:
    """Model config accepts configured baseline values."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "model": {
                "name": "ridge",
                "alpha": 2.5,
                "artifact_dir": "artifacts/custom",
            },
        },
    )
    cfg = load_config(cfg_file)
    assert cfg.model.name == "ridge"
    assert cfg.model.alpha == 2.5
    assert cfg.model.artifact_dir == Path("artifacts/custom")


def test_invalid_model_name_raises(tmp_path: Path) -> None:
    """Unsupported model name raises ValidationError."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "model": {"name": "neural-net"},
        },
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_invalid_model_alpha_raises(tmp_path: Path) -> None:
    """Ridge alpha must be positive."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "model": {"alpha": 0.0},
        },
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/config/test_schema.py -v
```

Expected: failures with `AttributeError: 'ProjectConfig' object has no attribute 'model'` or Pydantic validation errors because `ModelConfig` does not exist yet.

- [ ] **Step 3: Update `pyproject.toml`**

Add `joblib` to the main dependency list because `turbofan.models.artifacts` imports it directly:

```toml
dependencies = [
    "numpy",
    "pandas>=2.0",
    "pydantic>=2.0",
    "pyyaml",
    "scikit-learn",
    "joblib",
]
```

- [ ] **Step 4: Update `.gitignore`**

Add this section after the existing Outputs block:

```gitignore
# Model artifacts
artifacts/
```

- [ ] **Step 5: Update `configs/default.yaml`**

Add the model section at the end:

```yaml
model:
  name: ridge
  alpha: 1.0
  artifact_dir: artifacts/models
```

- [ ] **Step 6: Implement `ModelConfig`**

Replace `src/turbofan/config/schema.py` with:

```python
"""Configuration schema for the turbofan package."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for the data layer.

    Args:
        raw_dir: Path to raw data directory.
        processed_dir: Path to processed data directory.
        interim_dir: Path to interim data directory.
        fd_subset: C-MAPSS fault dataset subset identifier.
        max_rul: Maximum RUL cap for piecewise-linear labels.
        test_size: Fraction of training engines held out for validation.
        random_seed: Seed for all random operations.
    """

    raw_dir: Path
    processed_dir: Path
    interim_dir: Path
    fd_subset: Literal["FD001", "FD002", "FD003", "FD004"] = "FD001"
    max_rul: int = Field(default=125, gt=0)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_seed: int = 42


class ModelConfig(BaseModel):
    """Configuration for baseline model training.

    Args:
        name: Baseline model identifier.
        alpha: Ridge regularization strength.
        artifact_dir: Directory for local run artifacts.
    """

    name: Literal["ridge"] = "ridge"
    alpha: float = Field(default=1.0, gt=0.0)
    artifact_dir: Path = Path("artifacts/models")


class ProjectConfig(BaseModel):
    """Top-level project configuration.

    Args:
        project_name: Human-readable project name.
        data: Data layer configuration.
        model: Baseline model training configuration.
    """

    project_name: str
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)


def load_config(path: Path) -> ProjectConfig:
    """Load and validate project configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated ProjectConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If the config structure is invalid.
    """
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Failed to parse config file {path}: {exc}") from exc
    return ProjectConfig.model_validate(raw)
```

- [ ] **Step 7: Create package markers**

Create `src/turbofan/models/__init__.py`:

```python
"""Model training utilities for C-MAPSS RUL prediction."""
```

Create `tests/models/__init__.py` as an empty file.

- [ ] **Step 8: Run tests**

Run:

```powershell
pytest tests/config/test_schema.py -v
```

Expected: all config tests pass.

- [ ] **Step 9: Commit**

Run:

```powershell
git add pyproject.toml .gitignore configs/default.yaml src/turbofan/config/schema.py src/turbofan/models/__init__.py tests/config/test_schema.py tests/models/__init__.py
git commit -m "chore: add model training config"
```

---

## Task 2: Engine-level train/validation split

**Files:**
- Create: `tests/models/test_split.py`
- Create: `src/turbofan/models/split.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/test_split.py`:

```python
"""Tests for turbofan.models.split."""
from __future__ import annotations

import pandas as pd
import pytest

from turbofan.models.split import split_by_engine


def _make_df(n_engines: int = 5, n_cycles: int = 3) -> pd.DataFrame:
    """Build synthetic engine-cycle rows."""
    rows = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, n_cycles + 1):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "s_1": float(engine_id * cycle),
                }
            )
    return pd.DataFrame(rows)


def test_split_preserves_all_rows() -> None:
    """Train and validation outputs contain every input row once."""
    df = _make_df()
    train, val = split_by_engine(df, test_size=0.4, random_seed=42)
    assert len(train) + len(val) == len(df)
    assert set(train.index).isdisjoint(set(val.index))


def test_no_engine_leakage_between_splits() -> None:
    """No engine_id appears in both train and validation outputs."""
    df = _make_df()
    train, val = split_by_engine(df, test_size=0.4, random_seed=42)
    assert set(train["engine_id"]).isdisjoint(set(val["engine_id"]))


def test_split_is_deterministic_for_seed() -> None:
    """Same seed produces the same engine assignment."""
    df = _make_df()
    train_a, val_a = split_by_engine(df, test_size=0.4, random_seed=7)
    train_b, val_b = split_by_engine(df, test_size=0.4, random_seed=7)
    assert list(train_a["engine_id"]) == list(train_b["engine_id"])
    assert list(val_a["engine_id"]) == list(val_b["engine_id"])


def test_split_resets_indexes() -> None:
    """Returned DataFrames have clean RangeIndex values."""
    df = _make_df()
    train, val = split_by_engine(df, test_size=0.4, random_seed=42)
    assert list(train.index) == list(range(len(train)))
    assert list(val.index) == list(range(len(val)))


def test_requires_engine_id_column() -> None:
    """Missing engine_id raises ValueError."""
    df = pd.DataFrame({"cycle": [1, 2], "s_1": [1.0, 2.0]})
    with pytest.raises(ValueError, match="engine_id"):
        split_by_engine(df, test_size=0.5, random_seed=42)


def test_requires_at_least_two_engines() -> None:
    """A validation split needs at least two engines."""
    df = _make_df(n_engines=1)
    with pytest.raises(ValueError, match="at least two"):
        split_by_engine(df, test_size=0.5, random_seed=42)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/models/test_split.py -v
```

Expected: `ModuleNotFoundError: No module named 'turbofan.models.split'`.

- [ ] **Step 3: Implement `split.py`**

Create `src/turbofan/models/split.py`:

```python
"""Engine-level train/validation splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_engine(
    df: pd.DataFrame,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training DataFrame into train/validation engine groups.

    Args:
        df: Training rows containing an ``engine_id`` column.
        test_size: Fraction of engines assigned to validation.
        random_seed: Seed used to shuffle engine IDs.

    Returns:
        Tuple of train rows and validation rows with reset indexes.

    Raises:
        ValueError: If ``engine_id`` is missing, fewer than two engines are
            present, or the split would create an empty output.
    """
    if "engine_id" not in df.columns:
        raise ValueError("DataFrame must contain an engine_id column.")

    engine_ids = np.asarray(sorted(df["engine_id"].unique()))
    if len(engine_ids) < 2:
        raise ValueError("Engine-level split requires at least two engines.")

    n_val = int(round(len(engine_ids) * test_size))
    n_val = max(1, n_val)
    if n_val >= len(engine_ids):
        raise ValueError("Validation split would leave no training engines.")

    rng = np.random.default_rng(random_seed)
    shuffled = engine_ids.copy()
    rng.shuffle(shuffled)

    val_ids = set(shuffled[:n_val].tolist())
    val_mask = df["engine_id"].isin(val_ids)
    train = df.loc[~val_mask].reset_index(drop=True)
    val = df.loc[val_mask].reset_index(drop=True)
    return train, val
```

- [ ] **Step 4: Run tests**

Run:

```powershell
pytest tests/models/test_split.py -v
```

Expected: all split tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/turbofan/models/split.py tests/models/test_split.py
git commit -m "feat: add engine-level validation split"
```

---

## Task 3: RUL regression metrics

**Files:**
- Create: `tests/models/test_metrics.py`
- Create: `src/turbofan/models/metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/test_metrics.py`:

```python
"""Tests for turbofan.models.metrics."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from turbofan.models.metrics import mae, phm08_score, regression_metrics, rmse


def test_rmse_matches_hand_computation() -> None:
    """RMSE equals sqrt(mean squared error)."""
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = pd.Series([12.0, 18.0, 33.0])
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(17.0 / 3.0))


def test_mae_matches_hand_computation() -> None:
    """MAE equals mean absolute error."""
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = pd.Series([12.0, 18.0, 33.0])
    assert mae(y_true, y_pred) == pytest.approx(7.0 / 3.0)


def test_phm08_score_matches_early_and_late_errors() -> None:
    """PHM08 uses asymmetric penalties for early and late predictions."""
    y_true = pd.Series([100.0, 100.0])
    y_pred = pd.Series([90.0, 110.0])
    expected = (math.exp(10.0 / 13.0) - 1.0) + (math.exp(10.0 / 10.0) - 1.0)
    assert phm08_score(y_true, y_pred) == pytest.approx(expected)


def test_regression_metrics_returns_all_metrics() -> None:
    """Combined metrics returns the expected keys."""
    metrics = regression_metrics(
        pd.Series([1.0, 2.0]),
        pd.Series([1.5, 2.5]),
    )
    assert set(metrics) == {"rmse", "mae", "phm08_score"}


def test_metrics_reject_unequal_lengths() -> None:
    """Mismatched input lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        rmse(pd.Series([1.0]), pd.Series([1.0, 2.0]))


def test_metrics_reject_nan_inputs() -> None:
    """NaN values raise ValueError."""
    with pytest.raises(ValueError, match="NaN"):
        mae(pd.Series([1.0, np.nan]), pd.Series([1.0, 2.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/models/test_metrics.py -v
```

Expected: `ModuleNotFoundError: No module named 'turbofan.models.metrics'`.

- [ ] **Step 3: Implement `metrics.py`**

Create `src/turbofan/models/metrics.py`:

```python
"""Regression metrics for RUL prediction."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

MetricInput = pd.Series | npt.NDArray[np.float64]


def _as_arrays(
    y_true: MetricInput,
    y_pred: MetricInput,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate and convert metric inputs to float arrays.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Tuple of one-dimensional float arrays.

    Raises:
        ValueError: If lengths differ or inputs contain NaN values.
    """
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    if true_arr.shape[0] != pred_arr.shape[0]:
        raise ValueError("Metric inputs must have the same length.")
    if np.isnan(true_arr).any() or np.isnan(pred_arr).any():
        raise ValueError("Metric inputs must not contain NaN values.")
    return true_arr, pred_arr


def rmse(y_true: MetricInput, y_pred: MetricInput) -> float:
    """Compute root mean squared error.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Root mean squared error.
    """
    true_arr, pred_arr = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))


def mae(y_true: MetricInput, y_pred: MetricInput) -> float:
    """Compute mean absolute error.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Mean absolute error.
    """
    true_arr, pred_arr = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(pred_arr - true_arr)))


def phm08_score(y_true: MetricInput, y_pred: MetricInput) -> float:
    """Compute the asymmetric PHM08 RUL score.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Sum of PHM08 asymmetric penalties.
    """
    true_arr, pred_arr = _as_arrays(y_true, y_pred)
    diff = pred_arr - true_arr
    penalties = np.where(
        diff < 0.0,
        np.exp(-diff / 13.0) - 1.0,
        np.exp(diff / 10.0) - 1.0,
    )
    return float(np.sum(penalties))


def regression_metrics(
    y_true: MetricInput,
    y_pred: MetricInput,
) -> dict[str, float]:
    """Compute all baseline regression metrics.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Mapping with ``rmse``, ``mae``, and ``phm08_score``.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "phm08_score": phm08_score(y_true, y_pred),
    }
```

- [ ] **Step 4: Run tests**

Run:

```powershell
pytest tests/models/test_metrics.py -v
```

Expected: all metrics tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/turbofan/models/metrics.py tests/models/test_metrics.py
git commit -m "feat: add RUL regression metrics"
```

---

## Task 4: Baseline sklearn pipeline factory

**Files:**
- Create: `tests/models/test_baseline.py`
- Create: `src/turbofan/models/baseline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/test_baseline.py`:

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
    """Build model-ready synthetic turbofan rows and labels."""
    rng = np.random.default_rng(42)
    rows = []
    labels = []
    for engine_id in [1, 2, 3]:
        for cycle in range(1, 8):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "op_1": 0.0,
                    "op_2": 0.0,
                    "op_3": 0.0,
                    "s_1": float(cycle) + rng.normal(0.0, 0.1),
                    "s_2": 200.0,
                    "s_3": float(engine_id) + rng.normal(0.0, 0.1),
                }
            )
            labels.append(float(8 - cycle))
    return pd.DataFrame(rows), pd.Series(labels, name="rul")


def test_build_baseline_pipeline_named_steps() -> None:
    """Baseline pipeline exposes features and model steps."""
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
    model = pipe.named_steps["model"]
    assert isinstance(model, Ridge)
    assert model.alpha == 2.5


def test_pipeline_can_fit_and_predict() -> None:
    """Synthetic data can fit and predict without NaNs."""
    X, y = _make_df()
    pipe = build_baseline_pipeline(windows=[3])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert not np.isnan(preds).any()


def test_unknown_model_name_raises() -> None:
    """Unsupported model names fail fast."""
    with pytest.raises(ValueError, match="Unsupported model"):
        build_baseline_pipeline(model_name="random_forest")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/models/test_baseline.py -v
```

Expected: `ModuleNotFoundError: No module named 'turbofan.models.baseline'`.

- [ ] **Step 3: Implement `baseline.py`**

Create `src/turbofan/models/baseline.py`:

```python
"""Baseline sklearn pipeline factory."""
from __future__ import annotations

from typing import Literal

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from turbofan.features.pipeline import build_feature_pipeline


def build_baseline_pipeline(
    model_name: Literal["ridge"] = "ridge",
    alpha: float = 1.0,
    windows: list[int] | None = None,
    op_cols: list[str] | None = None,
) -> Pipeline:
    """Build an unfitted feature-plus-regressor sklearn Pipeline.

    Args:
        model_name: Baseline model identifier. Only ``"ridge"`` is supported.
        alpha: Ridge regularization strength.
        windows: Rolling window sizes for the feature pipeline.
        op_cols: Operational setting columns for normalization.

    Returns:
        Unfitted sklearn Pipeline with ``features`` and ``model`` steps.

    Raises:
        ValueError: If ``model_name`` is unsupported.
    """
    if model_name != "ridge":
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(
        [
            ("features", build_feature_pipeline(windows=windows, op_cols=op_cols)),
            ("model", Ridge(alpha=alpha)),
        ]
    )
```

- [ ] **Step 4: Run tests**

Run:

```powershell
pytest tests/models/test_baseline.py -v
```

Expected: all baseline tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/turbofan/models/baseline.py tests/models/test_baseline.py
git commit -m "feat: add Ridge baseline pipeline"
```

---

## Task 5: Evaluation helpers

**Files:**
- Create: `tests/models/test_evaluate.py`
- Create: `src/turbofan/models/evaluate.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/test_evaluate.py`:

```python
"""Tests for turbofan.models.evaluate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from turbofan.models.evaluate import (
    Predictor,
    add_rul_column,
    align_official_test_labels,
    evaluate_rows,
    select_last_cycle_per_engine,
    split_features_target,
)


class FixedPredictor:
    """Predictor returning preconfigured values."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return predictions matching the requested row count."""
        return np.asarray(self.values[: len(X)], dtype=float)


def test_add_rul_column_uses_capped_rul() -> None:
    """RUL column follows existing capped label semantics."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "s_1": [1.0, 2.0, 3.0],
        }
    )
    result = add_rul_column(df, max_rul=1)
    assert list(result["rul"]) == [1, 1, 0]
    assert "rul" not in df.columns


def test_split_features_target_removes_only_target() -> None:
    """Feature/target split preserves non-target columns."""
    df = pd.DataFrame(
        {"engine_id": [1], "cycle": [1], "s_1": [2.0], "rul": [3.0]}
    )
    X, y = split_features_target(df)
    assert list(X.columns) == ["engine_id", "cycle", "s_1"]
    assert list(y) == [3.0]


def test_select_last_cycle_per_engine() -> None:
    """Final row per engine is selected and sorted by engine_id."""
    df = pd.DataFrame(
        {
            "engine_id": [2, 1, 1, 2],
            "cycle": [1, 1, 3, 4],
            "s_1": [20.0, 10.0, 30.0, 40.0],
        }
    )
    result = select_last_cycle_per_engine(df)
    assert list(result["engine_id"]) == [1, 2]
    assert list(result["cycle"]) == [3, 4]


def test_align_official_test_labels() -> None:
    """Official labels align to one final-cycle row per engine."""
    last_rows = pd.DataFrame({"engine_id": [1, 2], "cycle": [3, 4]})
    labels = pd.Series([50, 60], name="rul")
    aligned = align_official_test_labels(last_rows, labels)
    assert list(aligned) == [50.0, 60.0]
    assert aligned.name == "rul"


def test_evaluate_rows_clips_negative_predictions() -> None:
    """Negative predictions are clipped to zero before metrics."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [1, 2],
            "s_1": [1.0, 2.0],
            "rul": [0.0, 10.0],
        }
    )
    metrics = evaluate_rows(FixedPredictor([-5.0, 10.0]), df)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0


def test_fixed_predictor_satisfies_protocol() -> None:
    """FixedPredictor implements the Predictor protocol."""
    predictor: Predictor = FixedPredictor([1.0])
    assert predictor.predict(pd.DataFrame({"x": [1.0]}))[0] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/models/test_evaluate.py -v
```

Expected: `ModuleNotFoundError: No module named 'turbofan.models.evaluate'`.

- [ ] **Step 3: Implement `evaluate.py`**

Create `src/turbofan/models/evaluate.py`:

```python
"""Evaluation helpers for baseline RUL models."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from turbofan.data.labels import compute_rul_labels
from turbofan.models.metrics import regression_metrics


class Predictor(Protocol):
    """Protocol for fitted estimators that can predict from a DataFrame."""

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Predict target values for feature rows.

        Args:
            X: Feature rows.

        Returns:
            Predicted RUL values.
        """
        ...


def add_rul_column(df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    """Return a copy of training data with a computed ``rul`` column.

    Args:
        df: Training DataFrame with ``engine_id`` and ``cycle``.
        max_rul: Maximum RUL cap.

    Returns:
        Copy of ``df`` with a ``rul`` column.
    """
    result = df.copy()
    result["rul"] = compute_rul_labels(result, max_rul=max_rul)
    return result


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "rul",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from target labels.

    Args:
        df: Labeled DataFrame.
        target_col: Target column name.

    Returns:
        Tuple of feature DataFrame and target Series.
    """
    return df.drop(columns=[target_col]), df[target_col].astype(float)


def evaluate_rows(
    estimator: Predictor,
    df: pd.DataFrame,
    target_col: str = "rul",
) -> dict[str, float]:
    """Evaluate predictions for labeled rows.

    Args:
        estimator: Fitted estimator with a ``predict`` method.
        df: Labeled rows to evaluate.
        target_col: Target column name.

    Returns:
        Regression metrics with non-negative clipped predictions.
    """
    X, y = split_features_target(df, target_col=target_col)
    preds = np.clip(np.asarray(estimator.predict(X), dtype=float), 0.0, None)
    return regression_metrics(y, preds)


def select_last_cycle_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Select the final available row for each engine.

    Args:
        df: Test DataFrame with ``engine_id`` and ``cycle`` columns.

    Returns:
        One row per engine, sorted by ``engine_id`` and reset index.
    """
    sorted_df = df.sort_values(["engine_id", "cycle"])
    idx = sorted_df.groupby("engine_id")["cycle"].idxmax()
    return sorted_df.loc[idx].sort_values("engine_id").reset_index(drop=True)


def align_official_test_labels(
    last_cycle_df: pd.DataFrame,
    rul_labels: pd.Series,
) -> pd.Series:
    """Align official RUL labels to final-cycle test rows.

    Args:
        last_cycle_df: One final-cycle row per test engine.
        rul_labels: Official RUL labels in engine order.

    Returns:
        Float RUL Series aligned to ``last_cycle_df``.

    Raises:
        ValueError: If the number of labels does not match the number of rows.
    """
    if len(last_cycle_df) != len(rul_labels):
        raise ValueError("Official RUL label count must match test engine count.")
    return pd.Series(
        rul_labels.to_numpy(dtype=float),
        index=last_cycle_df.index,
        name="rul",
    )
```

- [ ] **Step 4: Run tests**

Run:

```powershell
pytest tests/models/test_evaluate.py -v
```

Expected: all evaluation tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/turbofan/models/evaluate.py tests/models/test_evaluate.py
git commit -m "feat: add baseline evaluation helpers"
```

---

## Task 6: Local artifact persistence

**Files:**
- Create: `tests/models/test_artifacts.py`
- Create: `src/turbofan/models/artifacts.py`

- [ ] **Step 1: Write failing tests**

Create `tests/models/test_artifacts.py`:

```python
"""Tests for turbofan.models.artifacts."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge

from turbofan.models.artifacts import (
    create_run_dir,
    save_json,
    save_model,
    save_predictions,
)


def test_create_run_dir_uses_timestamp_and_run_name(tmp_path: Path) -> None:
    """Run directories are timestamped under the configured artifact dir."""
    timestamp = datetime(2026, 5, 24, 12, 30, 5, tzinfo=UTC)
    path = create_run_dir(tmp_path, "baseline", timestamp=timestamp)
    assert path == tmp_path / "baseline" / "20260524-123005"
    assert path.exists()


def test_save_model_round_trip(tmp_path: Path) -> None:
    """Saved joblib model can be loaded again."""
    model = Ridge().fit([[0.0], [1.0]], [0.0, 1.0])
    path = save_model(model, tmp_path / "model.joblib")
    loaded = joblib.load(path)
    assert loaded.predict([[2.0]])[0] == model.predict([[2.0]])[0]


def test_save_json_writes_valid_json(tmp_path: Path) -> None:
    """JSON payload is written with stringified Path values."""
    path = save_json(
        {"alpha": 1.0, "artifact_dir": Path("artifacts")},
        tmp_path / "x.json",
    )
    loaded = json.loads(path.read_text())
    assert loaded == {"alpha": 1.0, "artifact_dir": "artifacts"}


def test_save_predictions_writes_csv(tmp_path: Path) -> None:
    """Prediction rows are written to CSV."""
    df = pd.DataFrame({"engine_id": [1], "prediction": [10.0]})
    path = save_predictions(df, tmp_path / "predictions.csv")
    loaded = pd.read_csv(path)
    pd.testing.assert_frame_equal(loaded, df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/models/test_artifacts.py -v
```

Expected: `ModuleNotFoundError: No module named 'turbofan.models.artifacts'`.

- [ ] **Step 3: Implement `artifacts.py`**

Create `src/turbofan/models/artifacts.py`:

```python
"""Local artifact persistence for model training runs."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd
from sklearn.base import BaseEstimator


def create_run_dir(
    artifact_dir: Path,
    run_name: str,
    timestamp: datetime | None = None,
) -> Path:
    """Create and return a timestamped local run directory.

    Args:
        artifact_dir: Root artifact directory.
        run_name: Human-readable run group name.
        timestamp: Optional timestamp for deterministic tests.

    Returns:
        Created run directory path.
    """
    ts = timestamp if timestamp is not None else datetime.now(tz=UTC)
    run_dir = artifact_dir / run_name / ts.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_model(estimator: BaseEstimator, path: Path) -> Path:
    """Persist a fitted sklearn estimator with joblib.

    Args:
        estimator: Fitted sklearn estimator.
        path: Destination file path.

    Returns:
        Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)
    return path


def save_json(payload: Mapping[str, object], path: Path) -> Path:
    """Write a JSON artifact.

    Args:
        payload: JSON-serializable mapping. Path values are stringified.
        path: Destination file path.

    Returns:
        Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def save_predictions(df: pd.DataFrame, path: Path) -> Path:
    """Write prediction rows to CSV.

    Args:
        df: Prediction DataFrame.
        path: Destination file path.

    Returns:
        Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
```

- [ ] **Step 4: Run tests**

Run:

```powershell
pytest tests/models/test_artifacts.py -v
```

Expected: all artifact tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/turbofan/models/artifacts.py tests/models/test_artifacts.py
git commit -m "feat: add local model artifacts"
```

---

## Task 7: Baseline training CLI

**Files:**
- Create: `tests/models/test_train_baseline_cli.py`
- Create: `scripts/train_baseline.py`

- [ ] **Step 1: Write failing CLI smoke test**

Create `tests/models/test_train_baseline_cli.py`:

```python
"""Smoke tests for scripts/train_baseline.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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


def test_train_baseline_cli_writes_artifacts(tmp_path: Path) -> None:
    """CLI trains on synthetic files and writes model artifacts."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=8)
    _write_cmapps_file(raw_dir / "test_FD001.txt", n_engines=2, n_cycles=5)
    (raw_dir / "RUL_FD001.txt").write_text("10\n20\n")

    artifact_dir = tmp_path / "artifacts"
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
                f"  artifact_dir: {artifact_dir.as_posix()}",
            ]
        )
    )

    project_root = Path(__file__).parent.parent.parent
    result = subprocess.run(
        [sys.executable, "scripts/train_baseline.py", "--config", str(cfg_path)],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "validation rmse" in result.stdout
    run_dirs = list((artifact_dir / "baseline").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "validation_predictions.csv").exists()
    assert (run_dir / "official_test_predictions.csv").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "validation" in metrics
    assert "official_test" in metrics
    assert set(metrics["validation"]) == {"rmse", "mae", "phm08_score"}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests/models/test_train_baseline_cli.py -v
```

Expected: failure because `scripts/train_baseline.py` does not exist.

- [ ] **Step 3: Implement `scripts/train_baseline.py`**

Create `scripts/train_baseline.py`:

```python
"""Train a tabular baseline RUL model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from turbofan.config.schema import ProjectConfig, load_config
from turbofan.data.loader import load_raw_test, load_raw_train, load_rul_labels
from turbofan.models.artifacts import (
    create_run_dir,
    save_json,
    save_model,
    save_predictions,
)
from turbofan.models.baseline import build_baseline_pipeline
from turbofan.models.evaluate import (
    add_rul_column,
    align_official_test_labels,
    select_last_cycle_per_engine,
    split_features_target,
)
from turbofan.models.metrics import regression_metrics
from turbofan.models.split import split_by_engine


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML project config.",
    )
    return parser.parse_args()


def _config_to_dict(cfg: ProjectConfig) -> dict[str, object]:
    """Convert config to JSON-friendly dict.

    Args:
        cfg: Project config.

    Returns:
        Dictionary with JSON-friendly values.
    """
    return cfg.model_dump(mode="json")


def _prediction_frame(
    rows: pd.DataFrame,
    y_true: pd.Series,
    y_pred: npt.NDArray[np.float64],
) -> pd.DataFrame:
    """Build a prediction artifact DataFrame.

    Args:
        rows: Feature rows used for prediction.
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        DataFrame with identifiers, targets, and predictions.
    """
    return pd.DataFrame(
        {
            "engine_id": rows["engine_id"].to_numpy(),
            "cycle": rows["cycle"].to_numpy(),
            "rul": y_true.to_numpy(dtype=float),
            "prediction": y_pred,
        }
    )


def _evaluate_official_test(
    cfg: ProjectConfig,
    estimator: Pipeline,
) -> tuple[dict[str, float], pd.DataFrame] | None:
    """Evaluate final-cycle official test labels when files exist.

    Args:
        cfg: Project config.
        estimator: Fitted sklearn estimator.

    Returns:
        Metrics and prediction rows, or None when official files are missing.
    """
    try:
        test_raw = load_raw_test(cfg.data)
        rul_labels = load_rul_labels(cfg.data)
    except FileNotFoundError:
        return None

    last_rows = select_last_cycle_per_engine(test_raw)
    y_true = align_official_test_labels(last_rows, rul_labels)
    y_pred = np.clip(
        np.asarray(estimator.predict(last_rows), dtype=float),
        0.0,
        None,
    )
    metrics = regression_metrics(y_true, y_pred)
    predictions = _prediction_frame(last_rows, y_true, y_pred)
    return metrics, predictions


def main() -> None:
    """Train, evaluate, and persist a baseline model run."""
    args = _parse_args()
    cfg = load_config(args.config)

    train_raw = load_raw_train(cfg.data)
    train_labeled = add_rul_column(train_raw, max_rul=cfg.data.max_rul)
    train_df, val_df = split_by_engine(
        train_labeled,
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
    )

    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)

    estimator = build_baseline_pipeline(
        model_name=cfg.model.name,
        alpha=cfg.model.alpha,
    )
    estimator.fit(X_train, y_train)

    val_pred = np.clip(np.asarray(estimator.predict(X_val), dtype=float), 0.0, None)
    val_metrics = regression_metrics(y_val, val_pred)
    val_predictions = _prediction_frame(X_val, y_val, val_pred)

    run_dir = create_run_dir(cfg.model.artifact_dir, "baseline")
    metrics_payload: dict[str, object] = {"validation": val_metrics}

    official = _evaluate_official_test(cfg, estimator)
    if official is not None:
        official_metrics, official_predictions = official
        metrics_payload["official_test"] = official_metrics
        save_predictions(
            official_predictions,
            run_dir / "official_test_predictions.csv",
        )

    save_model(estimator, run_dir / "model.joblib")
    save_json(metrics_payload, run_dir / "metrics.json")
    save_json(_config_to_dict(cfg), run_dir / "config.json")
    save_predictions(val_predictions, run_dir / "validation_predictions.csv")

    print(f"run_dir: {run_dir}")
    print(f"validation rmse: {val_metrics['rmse']:.6f}")
    print(f"validation mae: {val_metrics['mae']:.6f}")
    print(f"validation phm08_score: {val_metrics['phm08_score']:.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI smoke test**

Run:

```powershell
pytest tests/models/test_train_baseline_cli.py -v
```

Expected: the CLI smoke test passes.

- [ ] **Step 5: Commit**

Run:

```powershell
git add scripts/train_baseline.py tests/models/test_train_baseline_cli.py
git commit -m "feat: add baseline training CLI"
```

---

## Task 8: Full quality gate

**Files:**
- Modify only files needed to fix lint, type, or test failures discovered by this task.

- [ ] **Step 1: Run model tests**

Run:

```powershell
pytest tests/models tests/config/test_schema.py -v
```

Expected: model and config tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
pytest
```

Expected: all tests pass.

- [ ] **Step 3: Run ruff**

Run:

```powershell
ruff check src/ tests/ scripts/
```

Expected: no lint violations.

- [ ] **Step 4: Run mypy**

Run:

```powershell
mypy src/turbofan
```

Expected: no type errors.

- [ ] **Step 5: Fix any failures with focused edits**

If a command fails, make the smallest targeted edit that addresses the failure, then rerun the failed command. Do not refactor unrelated modules during this task.

- [ ] **Step 6: Commit verification fixes if any**

If Task 8 required code changes, commit them:

```powershell
git add src tests scripts pyproject.toml configs/default.yaml .gitignore
git commit -m "fix: satisfy model training quality gates"
```

If no files changed, do not create an empty commit.

---

## Implementation Notes

- Use engine-level splitting only. A row-level split leaks trajectory information and invalidates validation metrics.
- Validation metrics run on all validation rows because training trajectories have labels for every cycle.
- Official test metrics run only on the final row per test engine because `RUL_*.txt` has one label per test engine.
- Clip predictions to zero before metric computation. Do not clip predictions to `max_rul`; values above the cap are useful diagnostics.
- Keep MLflow and neural/sequence models out of this milestone.
