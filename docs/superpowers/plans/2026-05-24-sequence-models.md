# Sequence Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a focused PyTorch GRU baseline for C-MAPSS RUL prediction using fixed-length sliding sequence windows.

**Architecture:** Add sequence preprocessing under `src/turbofan/sequences/` and GRU training helpers under `src/turbofan/models/`. Reuse the existing engine-level split, RUL label creation, regression metrics, official-test final-engine semantics, and local artifact helpers from the Model Training milestone. Keep deployment, MLflow, hyperparameter search, and non-GRU architectures out of this milestone.

**Tech Stack:** Python 3.12, pandas, numpy, PyTorch, pydantic, pytest, ruff, mypy

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
| `pyproject.toml` | Add direct `torch` runtime dependency |
| `configs/default.yaml` | Add default sequence training config |
| `src/turbofan/config/schema.py` | Add `SequenceConfig` and attach it to `ProjectConfig` |
| `tests/config/test_schema.py` | Cover default, custom, and invalid sequence config |
| `src/turbofan/sequences/__init__.py` | Sequence preprocessing package marker |
| `src/turbofan/sequences/normalize.py` | Train-fitted raw feature normalization |
| `src/turbofan/sequences/windowing.py` | Sliding and final fixed-length windows |
| `src/turbofan/sequences/dataset.py` | PyTorch Dataset and DataLoader helpers |
| `tests/sequences/__init__.py` | Sequence test package marker |
| `tests/sequences/test_normalize.py` | Synthetic tests for sequence normalization |
| `tests/sequences/test_windowing.py` | Synthetic tests for sliding/final windows |
| `tests/sequences/test_dataset.py` | Synthetic tests for dataset and dataloaders |
| `src/turbofan/models/gru.py` | GRU RUL regression module |
| `src/turbofan/models/sequence_training.py` | Device, training loop, evaluation helpers |
| `tests/models/test_gru.py` | GRU module tests |
| `tests/models/test_sequence_training.py` | Training helper tests |
| `scripts/train_sequence_gru.py` | CLI orchestration for one GRU sequence run |
| `tests/models/test_train_sequence_gru_cli.py` | CLI smoke tests and official-test semantics |

---

## Task 1: Config, Dependency, and Package Scaffolding

**Files:**
- Modify: `pyproject.toml`
- Modify: `configs/default.yaml`
- Modify: `src/turbofan/config/schema.py`
- Modify: `tests/config/test_schema.py`
- Create: `src/turbofan/sequences/__init__.py`
- Create: `tests/sequences/__init__.py`

- [ ] **Step 1: Write failing sequence config tests**

Append these tests to `tests/config/test_schema.py`:

```python
def test_sequence_config_defaults_when_section_omitted(tmp_path: Path) -> None:
    """Sequence config has stable defaults when omitted from YAML."""
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
    assert cfg.sequence.architecture == "gru"
    assert cfg.sequence.window_size == 30
    assert cfg.sequence.batch_size == 64
    assert cfg.sequence.hidden_size == 64
    assert cfg.sequence.num_layers == 1
    assert cfg.sequence.dropout == 0.0
    assert cfg.sequence.learning_rate == 0.001
    assert cfg.sequence.epochs == 50
    assert cfg.sequence.patience == 8
    assert cfg.sequence.device == "cpu"
    assert cfg.sequence.artifact_dir == Path("artifacts/models")


def test_sequence_config_loads_custom_values(tmp_path: Path) -> None:
    """Sequence config accepts configured GRU training values."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "sequence": {
                "architecture": "gru",
                "window_size": 12,
                "batch_size": 8,
                "hidden_size": 16,
                "num_layers": 2,
                "dropout": 0.1,
                "learning_rate": 0.01,
                "epochs": 5,
                "patience": 2,
                "device": "cpu",
                "artifact_dir": "artifacts/custom-sequence",
            },
        },
    )
    cfg = load_config(cfg_file)
    assert cfg.sequence.window_size == 12
    assert cfg.sequence.batch_size == 8
    assert cfg.sequence.hidden_size == 16
    assert cfg.sequence.num_layers == 2
    assert cfg.sequence.dropout == 0.1
    assert cfg.sequence.learning_rate == 0.01
    assert cfg.sequence.epochs == 5
    assert cfg.sequence.patience == 2
    assert cfg.sequence.device == "cpu"
    assert cfg.sequence.artifact_dir == Path("artifacts/custom-sequence")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("architecture", "lstm"),
        ("window_size", 0),
        ("batch_size", 0),
        ("hidden_size", 0),
        ("num_layers", 0),
        ("dropout", 1.0),
        ("learning_rate", 0.0),
        ("epochs", 0),
        ("patience", 0),
        ("device", "mps"),
    ],
)
def test_invalid_sequence_config_raises(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Invalid sequence config values raise ValidationError."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "sequence": {field: value},
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

Expected: FAIL because `ProjectConfig` has no `sequence` attribute and `SequenceConfig` does not exist.

- [ ] **Step 3: Add torch dependency**

Modify `pyproject.toml` so the runtime dependency list includes `torch`:

```toml
dependencies = [
    "numpy",
    "pandas>=2.0",
    "pydantic>=2.0",
    "pyyaml",
    "scikit-learn",
    "joblib",
    "torch",
]
```

- [ ] **Step 4: Add default sequence config**

Append this section to `configs/default.yaml`:

```yaml
sequence:
  architecture: gru
  window_size: 30
  batch_size: 64
  hidden_size: 64
  num_layers: 1
  dropout: 0.0
  learning_rate: 0.001
  epochs: 50
  patience: 8
  device: cpu
  artifact_dir: artifacts/models
```

- [ ] **Step 5: Implement `SequenceConfig`**

Update `src/turbofan/config/schema.py` imports:

```python
from pathlib import Path
from typing import Literal
```

Add this class after `ModelConfig`:

```python
class SequenceConfig(BaseModel):
    """Configuration for GRU sequence model training.

    Args:
        architecture: Sequence model architecture identifier.
        window_size: Number of cycles per sequence window.
        batch_size: Training batch size.
        hidden_size: GRU hidden state width.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability between GRU layers.
        learning_rate: Adam optimizer learning rate.
        epochs: Maximum training epochs.
        patience: Early-stopping patience in epochs.
        device: Requested torch device.
        artifact_dir: Directory for local sequence run artifacts.
    """

    architecture: Literal["gru"] = "gru"
    window_size: int = Field(default=30, gt=0)
    batch_size: int = Field(default=64, gt=0)
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=1, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    epochs: int = Field(default=50, gt=0)
    patience: int = Field(default=8, gt=0)
    device: Literal["cpu", "cuda"] = "cpu"
    artifact_dir: Path = Path("artifacts/models")
```

Update `ProjectConfig`:

```python
class ProjectConfig(BaseModel):
    """Top-level project configuration.

    Args:
        project_name: Human-readable project name.
        data: Data layer configuration.
        model: Baseline model training configuration.
        sequence: GRU sequence model training configuration.
    """

    project_name: str
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    sequence: SequenceConfig = Field(default_factory=SequenceConfig)
```

- [ ] **Step 6: Add package markers**

Create `src/turbofan/sequences/__init__.py`:

```python
"""Sequence preprocessing utilities for turbofan RUL models."""
```

Create `tests/sequences/__init__.py`:

```python
"""Tests for sequence preprocessing utilities."""
```

- [ ] **Step 7: Run config tests**

Run:

```powershell
pytest tests/config/test_schema.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add pyproject.toml configs/default.yaml src/turbofan/config/schema.py tests/config/test_schema.py src/turbofan/sequences/__init__.py tests/sequences/__init__.py
git commit -m "feat: add sequence model config"
```

---

## Task 2: Sequence Normalizer

**Files:**
- Create: `src/turbofan/sequences/normalize.py`
- Create: `tests/sequences/test_normalize.py`

- [ ] **Step 1: Write failing normalizer tests**

Create `tests/sequences/test_normalize.py`:

```python
"""Tests for sequence feature normalization."""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pytest

from turbofan.sequences.normalize import SequenceNormalizer, default_feature_cols


def _df() -> pd.DataFrame:
    """Build a small raw sequence DataFrame."""
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "op_1": [1.0, 3.0, 5.0, 7.0],
            "op_2": [2.0, 2.0, 2.0, 2.0],
            "op_3": [0.0, 1.0, 0.0, 1.0],
            "s_1": [10.0, 12.0, 20.0, 22.0],
            "s_2": [5.0, 5.0, 5.0, 5.0],
            "rul": [3, 2, 3, 2],
        }
    )


def test_default_feature_cols_include_ops_and_sensors() -> None:
    """Default feature columns are operational settings plus 21 sensors."""
    cols = default_feature_cols()
    assert cols[:3] == ["op_1", "op_2", "op_3"]
    assert cols[3] == "s_1"
    assert cols[-1] == "s_21"
    assert len(cols) == 24


def test_normalizer_fits_and_transforms_training_statistics() -> None:
    """Normalizer uses fitted means and standard deviations."""
    normalizer = SequenceNormalizer(feature_cols=["op_1", "op_2", "s_1"])
    transformed = normalizer.fit_transform(_df())
    assert normalizer.means_.to_dict() == {"op_1": 4.0, "op_2": 2.0, "s_1": 16.0}
    assert normalizer.stds_.to_dict()["op_2"] == 1.0
    assert transformed["engine_id"].tolist() == [1, 1, 2, 2]
    assert transformed["cycle"].tolist() == [1, 2, 1, 2]
    assert transformed["rul"].tolist() == [3, 2, 3, 2]
    assert transformed["op_2"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert abs(float(transformed["op_1"].mean())) < 1e-12


def test_transform_uses_stored_training_statistics() -> None:
    """Validation rows are transformed with training statistics."""
    train = _df().iloc[:2].reset_index(drop=True)
    validation = _df().iloc[2:].reset_index(drop=True)
    normalizer = SequenceNormalizer(feature_cols=["op_1"])
    normalizer.fit(train)
    transformed = normalizer.transform(validation)
    assert transformed["op_1"].tolist() == [3.0, 5.0]


def test_missing_required_feature_raises() -> None:
    """Missing configured feature columns raise KeyError."""
    normalizer = SequenceNormalizer(feature_cols=["missing"])
    with pytest.raises(KeyError):
        normalizer.fit(_df())


def test_transform_before_fit_raises() -> None:
    """Transforming before fit raises RuntimeError."""
    normalizer = SequenceNormalizer(feature_cols=["op_1"])
    with pytest.raises(RuntimeError):
        normalizer.transform(_df())


def test_constructor_accepts_any_sequence_of_columns() -> None:
    """Constructor accepts non-list sequences."""
    cols: Sequence[str] = ("op_1", "s_1")
    normalizer = SequenceNormalizer(feature_cols=cols)
    assert normalizer.feature_cols == ["op_1", "s_1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/sequences/test_normalize.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'turbofan.sequences.normalize'`.

- [ ] **Step 3: Implement normalizer**

Create `src/turbofan/sequences/normalize.py`:

```python
"""Train-fitted normalization for sequence model inputs."""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def default_feature_cols() -> list[str]:
    """Return default raw sequence feature columns.

    Returns:
        Operational setting columns followed by sensor columns.
    """
    return ["op_1", "op_2", "op_3", *[f"s_{i}" for i in range(1, 22)]]


class SequenceNormalizer:
    """Normalize raw sequence feature columns with train-fitted statistics.

    Args:
        feature_cols: Feature columns to normalize. Defaults to operational
            setting columns and all 21 C-MAPSS sensor columns.
    """

    def __init__(self, feature_cols: Sequence[str] | None = None) -> None:
        self.feature_cols = (
            list(feature_cols) if feature_cols is not None else default_feature_cols()
        )

    def fit(self, df: pd.DataFrame) -> SequenceNormalizer:
        """Fit means and standard deviations from training rows.

        Args:
            df: Training DataFrame containing all configured feature columns.

        Returns:
            Fitted normalizer.

        Raises:
            KeyError: If any configured feature column is missing.
        """
        self._validate_columns(df)
        features = df[self.feature_cols].astype(float)
        self.means_ = features.mean()
        stds = features.std(ddof=0)
        self.stds_ = stds.fillna(1.0).replace(0.0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted normalization statistics.

        Args:
            df: DataFrame containing all configured feature columns.

        Returns:
            Copy of ``df`` with normalized feature columns.

        Raises:
            RuntimeError: If called before ``fit``.
            KeyError: If any configured feature column is missing.
        """
        if not hasattr(self, "means_") or not hasattr(self, "stds_"):
            raise RuntimeError("SequenceNormalizer must be fit before transform.")
        self._validate_columns(df)
        result = df.copy()
        result.loc[:, self.feature_cols] = (
            result[self.feature_cols].astype(float) - self.means_
        ) / self.stds_
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit statistics and transform the same DataFrame.

        Args:
            df: Training DataFrame.

        Returns:
            Normalized copy of ``df``.

        Raises:
            KeyError: If any configured feature column is missing.
        """
        return self.fit(df).transform(df)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate required feature columns are present.

        Args:
            df: DataFrame to validate.

        Raises:
            KeyError: If any configured feature column is missing.
        """
        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Missing sequence feature columns: {missing}")
```

- [ ] **Step 4: Run normalizer tests**

Run:

```powershell
pytest tests/sequences/test_normalize.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/turbofan/sequences/normalize.py tests/sequences/test_normalize.py
git commit -m "feat: add sequence normalizer"
```

---

## Task 3: Sequence Windowing

**Files:**
- Create: `src/turbofan/sequences/windowing.py`
- Create: `tests/sequences/test_windowing.py`

- [ ] **Step 1: Write failing windowing tests**

Create `tests/sequences/test_windowing.py`:

```python
"""Tests for fixed-length sequence windowing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.sequences.windowing import (
    WindowedSequences,
    build_final_windows,
    build_sliding_windows,
)


FEATURE_COLS = ["op_1", "s_1"]


def _df() -> pd.DataFrame:
    """Build deliberately unsorted labeled sequence rows."""
    return pd.DataFrame(
        {
            "engine_id": [2, 1, 1, 2, 1, 2, 3],
            "cycle": [2, 1, 3, 1, 2, 3, 1],
            "op_1": [20.0, 1.0, 3.0, 10.0, 2.0, 30.0, 99.0],
            "s_1": [200.0, 10.0, 30.0, 100.0, 20.0, 300.0, 990.0],
            "rul": [1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 5.0],
        }
    )


def test_sliding_windows_are_sorted_and_labeled_by_final_timestep() -> None:
    """Sliding windows sort by engine/cycle and label final timesteps."""
    windows = build_sliding_windows(_df(), FEATURE_COLS, window_size=2)
    assert isinstance(windows, WindowedSequences)
    assert windows.X.shape == (4, 2, 2)
    assert windows.y.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert windows.metadata[["engine_id", "cycle"]].to_dict("records") == [
        {"engine_id": 1, "cycle": 2},
        {"engine_id": 1, "cycle": 3},
        {"engine_id": 2, "cycle": 2},
        {"engine_id": 2, "cycle": 3},
    ]
    np.testing.assert_array_equal(
        windows.X[0],
        np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
    )


def test_final_windows_return_one_window_per_eligible_engine() -> None:
    """Final windows use only the last full window per engine."""
    windows = build_final_windows(_df(), FEATURE_COLS, window_size=2)
    assert windows.X.shape == (2, 2, 2)
    assert windows.y.tolist() == [0.0, 0.0]
    assert windows.metadata[["engine_id", "cycle"]].to_dict("records") == [
        {"engine_id": 1, "cycle": 3},
        {"engine_id": 2, "cycle": 3},
    ]


def test_final_windows_without_target_returns_nan_labels() -> None:
    """Unlabeled final windows are supported for prediction."""
    windows = build_final_windows(
        _df().drop(columns=["rul"]),
        FEATURE_COLS,
        window_size=2,
        target_col=None,
    )
    assert windows.X.shape == (2, 2, 2)
    assert np.isnan(windows.y).all()


def test_short_engines_are_skipped() -> None:
    """Engines shorter than the window size are skipped."""
    windows = build_sliding_windows(_df(), FEATURE_COLS, window_size=3)
    assert windows.X.shape == (2, 3, 2)
    assert windows.metadata["engine_id"].tolist() == [1, 2]


def test_no_eligible_windows_raises() -> None:
    """No eligible windows raises ValueError."""
    with pytest.raises(ValueError, match="No eligible sequence windows"):
        build_sliding_windows(_df(), FEATURE_COLS, window_size=10)


def test_missing_columns_raise() -> None:
    """Missing required columns raise KeyError."""
    with pytest.raises(KeyError):
        build_sliding_windows(_df().drop(columns=["s_1"]), FEATURE_COLS, 2)
    with pytest.raises(KeyError):
        build_sliding_windows(_df().drop(columns=["rul"]), FEATURE_COLS, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/sequences/test_windowing.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'turbofan.sequences.windowing'`.

- [ ] **Step 3: Implement windowing**

Create `src/turbofan/sequences/windowing.py`:

```python
"""Fixed-length sequence window construction."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class WindowedSequences:
    """Container for sequence windows, labels, and row metadata.

    Args:
        X: Window features with shape ``(n_windows, window_size, n_features)``.
        y: Labels with shape ``(n_windows,)``.
        metadata: Final-timestep metadata for each window.
    """

    X: npt.NDArray[np.float32]
    y: npt.NDArray[np.float32]
    metadata: pd.DataFrame


def build_sliding_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window_size: int,
    target_col: str = "rul",
) -> WindowedSequences:
    """Build fixed-length sliding windows per engine.

    Args:
        df: Labeled rows containing ``engine_id`` and ``cycle``.
        feature_cols: Feature columns to include in each timestep.
        window_size: Number of cycles per window.
        target_col: Label column. The final timestep label is used.

    Returns:
        Windowed sequence container.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If ``window_size`` is not positive or no windows are built.
    """
    return _build_windows(
        df=df,
        feature_cols=feature_cols,
        window_size=window_size,
        target_col=target_col,
        final_only=False,
    )


def build_final_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window_size: int,
    target_col: str | None = "rul",
) -> WindowedSequences:
    """Build one final full window per eligible engine.

    Args:
        df: Rows containing ``engine_id`` and ``cycle``.
        feature_cols: Feature columns to include in each timestep.
        window_size: Number of cycles per window.
        target_col: Optional label column. If ``None``, labels are NaN.

    Returns:
        Windowed sequence container.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If ``window_size`` is not positive or no windows are built.
    """
    return _build_windows(
        df=df,
        feature_cols=feature_cols,
        window_size=window_size,
        target_col=target_col,
        final_only=True,
    )


def _build_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window_size: int,
    target_col: str | None,
    final_only: bool,
) -> WindowedSequences:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    required = ["engine_id", "cycle", *feature_cols]
    if target_col is not None:
        required.append(target_col)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing sequence window columns: {missing}")

    sorted_df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    windows: list[npt.NDArray[np.float32]] = []
    labels: list[float] = []
    metadata_rows: list[pd.Series] = []

    for _, group in sorted_df.groupby("engine_id", sort=True):
        if len(group) < window_size:
            continue
        starts = [len(group) - window_size] if final_only else range(
            0, len(group) - window_size + 1
        )
        for start in starts:
            end = start + window_size
            window = group.iloc[start:end]
            windows.append(window[list(feature_cols)].to_numpy(dtype=np.float32))
            final_row = window.iloc[-1]
            labels.append(
                float("nan") if target_col is None else float(final_row[target_col])
            )
            metadata_rows.append(final_row[["engine_id", "cycle"]])

    if not windows:
        raise ValueError("No eligible sequence windows could be built.")

    return WindowedSequences(
        X=np.stack(windows).astype(np.float32),
        y=np.asarray(labels, dtype=np.float32),
        metadata=pd.DataFrame(metadata_rows).reset_index(drop=True),
    )
```

- [ ] **Step 4: Run windowing tests**

Run:

```powershell
pytest tests/sequences/test_windowing.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/turbofan/sequences/windowing.py tests/sequences/test_windowing.py
git commit -m "feat: add sequence windowing"
```

---

## Task 4: PyTorch Sequence Dataset Helpers

**Files:**
- Create: `src/turbofan/sequences/dataset.py`
- Create: `tests/sequences/test_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

Create `tests/sequences/test_dataset.py`:

```python
"""Tests for sequence PyTorch dataset helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from turbofan.sequences.dataset import SequenceDataset, build_sequence_loader
from turbofan.sequences.windowing import WindowedSequences


def _windows() -> WindowedSequences:
    """Build tiny sequence windows."""
    return WindowedSequences(
        X=np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float32,
        ),
        y=np.asarray([10.0, 20.0], dtype=np.float32),
        metadata=pd.DataFrame({"engine_id": [1, 2], "cycle": [2, 2]}),
    )


def test_sequence_dataset_returns_tensors() -> None:
    """Dataset returns feature and target tensors."""
    dataset = SequenceDataset(_windows())
    X, y = dataset[0]
    assert len(dataset) == 2
    assert X.shape == (2, 2)
    assert y.shape == torch.Size([])
    assert X.dtype == torch.float32
    assert y.dtype == torch.float32
    assert float(y) == 10.0


def test_eval_loader_preserves_order() -> None:
    """Evaluation loader does not shuffle rows."""
    loader = build_sequence_loader(_windows(), batch_size=1, shuffle=False)
    targets = [float(y.item()) for _, y in loader]
    assert targets == [10.0, 20.0]


def test_train_loader_uses_configured_batch_size() -> None:
    """Loader respects batch size."""
    loader = build_sequence_loader(_windows(), batch_size=2, shuffle=True)
    batch_X, batch_y = next(iter(loader))
    assert batch_X.shape == (2, 2, 2)
    assert batch_y.shape == (2,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/sequences/test_dataset.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'turbofan.sequences.dataset'`.

- [ ] **Step 3: Implement dataset helpers**

Create `src/turbofan/sequences/dataset.py`:

```python
"""PyTorch Dataset helpers for sequence windows."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from turbofan.sequences.windowing import WindowedSequences


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for RUL sequence windows.

    Args:
        windows: Windowed sequence features and labels.
    """

    def __init__(self, windows: WindowedSequences) -> None:
        self.X = torch.as_tensor(windows.X, dtype=torch.float32)
        self.y = torch.as_tensor(windows.y, dtype=torch.float32)

    def __len__(self) -> int:
        """Return number of windows.

        Returns:
            Dataset length.
        """
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one sequence sample.

        Args:
            index: Dataset index.

        Returns:
            Feature tensor and scalar target tensor.
        """
        return self.X[index], self.y[index]


def build_sequence_loader(
    windows: WindowedSequences,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a DataLoader for sequence windows.

    Args:
        windows: Windowed sequence features and labels.
        batch_size: Batch size.
        shuffle: Whether to shuffle samples.

    Returns:
        PyTorch DataLoader.
    """
    dataset = SequenceDataset(windows)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
```

- [ ] **Step 4: Run dataset tests**

Run:

```powershell
pytest tests/sequences/test_dataset.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/turbofan/sequences/dataset.py tests/sequences/test_dataset.py
git commit -m "feat: add sequence dataset helpers"
```

---

## Task 5: GRU RUL Regressor

**Files:**
- Create: `src/turbofan/models/gru.py`
- Create: `tests/models/test_gru.py`

- [ ] **Step 1: Write failing GRU tests**

Create `tests/models/test_gru.py`:

```python
"""Tests for the GRU RUL regressor."""
from __future__ import annotations

import pytest
import torch

from turbofan.models.gru import GRURULRegressor


def test_gru_forward_returns_one_prediction_per_window() -> None:
    """Forward pass returns shape (batch_size,)."""
    model = GRURULRegressor(input_size=4, hidden_size=8, num_layers=1, dropout=0.0)
    X = torch.ones((3, 5, 4), dtype=torch.float32)
    y = model(X)
    assert y.shape == (3,)


def test_gru_supports_stacked_layers_and_dropout() -> None:
    """Stacked GRU accepts dropout."""
    model = GRURULRegressor(input_size=4, hidden_size=8, num_layers=2, dropout=0.2)
    X = torch.ones((2, 6, 4), dtype=torch.float32)
    y = model(X)
    assert y.shape == (2,)


@pytest.mark.parametrize(
    ("input_size", "hidden_size", "num_layers", "dropout"),
    [
        (0, 8, 1, 0.0),
        (4, 0, 1, 0.0),
        (4, 8, 0, 0.0),
        (4, 8, 1, -0.1),
        (4, 8, 1, 1.0),
    ],
)
def test_invalid_gru_constructor_values_raise(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> None:
    """Invalid model dimensions raise ValueError."""
    with pytest.raises(ValueError):
        GRURULRegressor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/models/test_gru.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'turbofan.models.gru'`.

- [ ] **Step 3: Implement GRU model**

Create `src/turbofan/models/gru.py`:

```python
"""GRU sequence model for RUL regression."""
from __future__ import annotations

import torch
from torch import nn


class GRURULRegressor(nn.Module):
    """GRU-based RUL regressor for fixed-length sensor windows.

    Args:
        input_size: Number of features per timestep.
        hidden_size: GRU hidden state width.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability between GRU layers.

    Raises:
        ValueError: If dimensions or dropout are invalid.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1).")

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Predict RUL for sequence windows.

        Args:
            X: Tensor with shape ``(batch_size, window_size, input_size)``.

        Returns:
            Predicted RUL tensor with shape ``(batch_size,)``.
        """
        _, hidden = self.gru(X)
        last_hidden = hidden[-1]
        return self.regressor(last_hidden).squeeze(-1)
```

- [ ] **Step 4: Run GRU tests**

Run:

```powershell
pytest tests/models/test_gru.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/turbofan/models/gru.py tests/models/test_gru.py
git commit -m "feat: add GRU RUL regressor"
```

---

## Task 6: Sequence Training Helpers

**Files:**
- Create: `src/turbofan/models/sequence_training.py`
- Create: `tests/models/test_sequence_training.py`

- [ ] **Step 1: Write failing training helper tests**

Create `tests/models/test_sequence_training.py`:

```python
"""Tests for GRU sequence training helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from turbofan.config.schema import SequenceConfig
from turbofan.models.gru import GRURULRegressor
from turbofan.models.sequence_training import (
    TrainingResult,
    predict_windows,
    resolve_device,
    train_gru_model,
)
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.windowing import WindowedSequences


def _linear_windows() -> WindowedSequences:
    """Build tiny deterministic windows with labels from final timestep."""
    X = np.asarray(
        [
            [[0.0], [1.0], [2.0]],
            [[1.0], [2.0], [3.0]],
            [[2.0], [3.0], [4.0]],
            [[3.0], [4.0], [5.0]],
        ],
        dtype=np.float32,
    )
    y = np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    metadata = pd.DataFrame({"engine_id": [1, 1, 2, 2], "cycle": [3, 4, 3, 4]})
    return WindowedSequences(X=X, y=y, metadata=metadata)


def test_resolve_device_cpu() -> None:
    """CPU device resolves by default."""
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_cuda_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable CUDA requests raise a clear error."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA was requested"):
        resolve_device("cuda")


def test_predict_windows_returns_one_value_per_window() -> None:
    """Prediction helper returns numpy predictions."""
    windows = _linear_windows()
    loader = build_sequence_loader(windows, batch_size=2, shuffle=False)
    model = GRURULRegressor(input_size=1, hidden_size=4, num_layers=1, dropout=0.0)
    preds = predict_windows(model, loader, torch.device("cpu"))
    assert preds.shape == (4,)
    assert preds.dtype == np.float64


def test_train_gru_model_returns_best_state_and_history() -> None:
    """Training loop returns history and best model metadata."""
    windows = _linear_windows()
    train_loader = build_sequence_loader(windows, batch_size=2, shuffle=True)
    eval_loader = build_sequence_loader(windows, batch_size=2, shuffle=False)
    cfg = SequenceConfig(
        window_size=3,
        batch_size=2,
        hidden_size=4,
        epochs=3,
        patience=2,
        learning_rate=0.01,
    )
    model = GRURULRegressor(input_size=1, hidden_size=4, num_layers=1, dropout=0.0)

    result = train_gru_model(
        model=model,
        train_loader=train_loader,
        validation_final_loader=eval_loader,
        validation_windows_loader=eval_loader,
        config=cfg,
        device=torch.device("cpu"),
        random_seed=42,
    )

    assert isinstance(result, TrainingResult)
    assert result.best_epoch >= 1
    assert result.best_metric >= 0.0
    assert set(result.history.columns) == {
        "epoch",
        "train_loss",
        "validation_final_window_rmse",
        "validation_final_window_mae",
        "validation_final_window_phm08_score",
        "validation_windows_rmse",
        "validation_windows_mae",
        "validation_windows_phm08_score",
    }
    assert len(result.history) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/models/test_sequence_training.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'turbofan.models.sequence_training'`.

- [ ] **Step 3: Implement training helpers**

Create `src/turbofan/models/sequence_training.py`:

```python
"""Training helpers for GRU sequence models."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from turbofan.config.schema import SequenceConfig
from turbofan.models.gru import GRURULRegressor
from turbofan.models.metrics import regression_metrics


@dataclass(frozen=True)
class TrainingResult:
    """Best GRU state and epoch-level training history.

    Args:
        model: Model restored to the best validation state.
        history: Epoch-level metrics.
        best_epoch: Epoch with the lowest primary validation RMSE.
        best_metric: Best primary validation RMSE.
    """

    model: GRURULRegressor
    history: pd.DataFrame
    best_epoch: int
    best_metric: float


def resolve_device(requested: Literal["cpu", "cuda"]) -> torch.device:
    """Resolve configured device to an available torch device.

    Args:
        requested: Configured device name.

    Returns:
        Resolved torch device.

    Raises:
        ValueError: If CUDA is requested but unavailable.
    """
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def train_gru_model(
    model: GRURULRegressor,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    validation_final_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    validation_windows_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    config: SequenceConfig,
    device: torch.device,
    random_seed: int,
) -> TrainingResult:
    """Train a GRU model with early stopping.

    Args:
        model: GRU model to train.
        train_loader: Training window batches.
        validation_final_loader: Primary validation final-window batches.
        validation_windows_loader: Diagnostic validation sliding-window batches.
        config: Sequence training config.
        device: Torch device.
        random_seed: Seed for deterministic training setup.

    Returns:
        Training result with best model and history.
    """
    _seed_everything(random_seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    best_metric = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        final_metrics = _evaluate_loader(model, validation_final_loader, device)
        window_metrics = _evaluate_loader(model, validation_windows_loader, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_final_window_rmse": final_metrics["rmse"],
            "validation_final_window_mae": final_metrics["mae"],
            "validation_final_window_phm08_score": final_metrics["phm08_score"],
            "validation_windows_rmse": window_metrics["rmse"],
            "validation_windows_mae": window_metrics["mae"],
            "validation_windows_phm08_score": window_metrics["phm08_score"],
        }
        rows.append(row)

        primary = final_metrics["rmse"]
        if primary < best_metric:
            best_metric = primary
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        history=pd.DataFrame(rows),
        best_epoch=best_epoch,
        best_metric=best_metric,
    )


def predict_windows(
    model: GRURULRegressor,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> npt.NDArray[np.float64]:
    """Predict RUL values for a sequence DataLoader.

    Args:
        model: Fitted GRU model.
        loader: Sequence DataLoader.
        device: Torch device.

    Returns:
        Prediction array with one value per input window.
    """
    model.eval()
    preds: list[npt.NDArray[np.float64]] = []
    with torch.no_grad():
        for X, _ in loader:
            batch_preds = model(X.to(device)).detach().cpu().numpy()
            preds.append(batch_preds.astype(np.float64))
    return np.concatenate(preds)


def _seed_everything(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def _train_one_epoch(
    model: GRURULRegressor,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for X, y in loader:
        X_device = X.to(device)
        y_device = y.to(device)
        optimizer.zero_grad()
        preds = model(X_device)
        loss = loss_fn(preds, y_device)
        loss.backward()
        optimizer.step()
        batch_size = int(X.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size
    return total_loss / total_rows


def _evaluate_loader(
    model: GRURULRegressor,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    y_true = _loader_targets(loader)
    y_pred = np.clip(predict_windows(model, loader, device), 0.0, None)
    return regression_metrics(y_true, y_pred)


def _loader_targets(
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> npt.NDArray[np.float64]:
    targets: list[npt.NDArray[np.float64]] = []
    for _, y in loader:
        targets.append(y.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(targets)
```

- [ ] **Step 4: Run training helper tests**

Run:

```powershell
pytest tests/models/test_sequence_training.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/turbofan/models/sequence_training.py tests/models/test_sequence_training.py
git commit -m "feat: add GRU training helpers"
```

---

## Task 7: Sequence GRU Training CLI

**Files:**
- Create: `scripts/train_sequence_gru.py`
- Create: `tests/models/test_train_sequence_gru_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/models/test_train_sequence_gru_cli.py`:

```python
"""Smoke tests for scripts/train_sequence_gru.py."""
from __future__ import annotations

import json
import os
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


def _write_config(tmp_path: Path, raw_dir: Path, artifact_dir: Path) -> Path:
    """Write a tiny sequence training config."""
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
                "sequence:",
                "  architecture: gru",
                "  window_size: 3",
                "  batch_size: 2",
                "  hidden_size: 4",
                "  num_layers: 1",
                "  dropout: 0.0",
                "  learning_rate: 0.01",
                "  epochs: 2",
                "  patience: 2",
                "  device: cpu",
                f"  artifact_dir: {artifact_dir.as_posix()}",
            ]
        )
    )
    return cfg_path


def test_train_sequence_gru_cli_writes_artifacts(tmp_path: Path) -> None:
    """CLI trains on synthetic files and writes sequence artifacts."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)
    _write_cmapps_file(raw_dir / "test_FD001.txt", n_engines=2, n_cycles=5)
    (raw_dir / "RUL_FD001.txt").write_text("10\n20\n")
    artifact_dir = tmp_path / "artifacts"
    cfg_path = _write_config(tmp_path, raw_dir, artifact_dir)

    project_root = Path(__file__).parent.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [sys.executable, "scripts/train_sequence_gru.py", "--config", str(cfg_path)],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "validation_final_window rmse" in result.stdout
    run_dirs = list((artifact_dir / "sequence_gru").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "training_history.csv").exists()
    assert (run_dir / "validation_final_window_predictions.csv").exists()
    assert (run_dir / "validation_window_predictions.csv").exists()
    assert (run_dir / "official_test_predictions.csv").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {
        "validation_final_window",
        "validation_windows",
        "official_test",
    }
    assert set(metrics["validation_final_window"]) == {
        "rmse",
        "mae",
        "phm08_score",
    }


def test_train_sequence_gru_cli_skips_missing_official_test(
    tmp_path: Path,
) -> None:
    """CLI trains validation model when official test files are absent."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)
    artifact_dir = tmp_path / "artifacts"
    cfg_path = _write_config(tmp_path, raw_dir, artifact_dir)

    project_root = Path(__file__).parent.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [sys.executable, "scripts/train_sequence_gru.py", "--config", str(cfg_path)],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "official test evaluation skipped" in result.stdout
    run_dir = next((artifact_dir / "sequence_gru").iterdir())
    assert not (run_dir / "official_test_predictions.csv").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {"validation_final_window", "validation_windows"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/models/test_train_sequence_gru_cli.py -v
```

Expected: FAIL because `scripts/train_sequence_gru.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `scripts/train_sequence_gru.py`:

```python
"""Train a GRU sequence RUL model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from turbofan.config.schema import ProjectConfig, load_config
from turbofan.data.loader import load_raw_test, load_raw_train, load_rul_labels
from turbofan.models.artifacts import create_run_dir, save_json, save_predictions
from turbofan.models.evaluate import add_rul_column, align_official_test_labels
from turbofan.models.gru import GRURULRegressor
from turbofan.models.metrics import regression_metrics
from turbofan.models.sequence_training import (
    predict_windows,
    resolve_device,
    train_gru_model,
)
from turbofan.models.split import split_by_engine
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.normalize import SequenceNormalizer, default_feature_cols
from turbofan.sequences.windowing import (
    WindowedSequences,
    build_final_windows,
    build_sliding_windows,
)


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
    windows: WindowedSequences,
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> pd.DataFrame:
    """Build prediction artifact rows.

    Args:
        windows: Window metadata container.
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Prediction artifact DataFrame.
    """
    return pd.DataFrame(
        {
            "engine_id": windows.metadata["engine_id"].to_numpy(),
            "cycle": windows.metadata["cycle"].to_numpy(),
            "rul": y_true,
            "prediction": y_pred,
        }
    )


def _evaluate_windows(
    model: GRURULRegressor,
    windows: WindowedSequences,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate labeled sequence windows.

    Args:
        model: Fitted model.
        windows: Labeled windows.
        batch_size: Evaluation batch size.
        device: Torch device.

    Returns:
        Metrics and prediction rows.
    """
    loader = build_sequence_loader(windows, batch_size=batch_size, shuffle=False)
    y_true = windows.y.astype(np.float64)
    y_pred = np.clip(predict_windows(model, loader, device), 0.0, None)
    return regression_metrics(y_true, y_pred), _prediction_frame(
        windows,
        y_true,
        y_pred,
    )


def _evaluate_official_test(
    cfg: ProjectConfig,
    model: GRURULRegressor,
    normalizer: SequenceNormalizer,
    feature_cols: list[str],
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame] | None:
    """Evaluate official final-window test labels when files exist.

    Args:
        cfg: Project config.
        model: Fitted sequence model.
        normalizer: Fitted sequence normalizer.
        feature_cols: Ordered feature columns.
        device: Torch device.

    Returns:
        Metrics and prediction rows, or None when official files are missing.
    """
    try:
        test_raw = load_raw_test(cfg.data)
        rul_labels = load_rul_labels(cfg.data)
    except FileNotFoundError:
        return None

    test_norm = normalizer.transform(test_raw)
    test_windows = build_final_windows(
        test_norm,
        feature_cols,
        window_size=cfg.sequence.window_size,
        target_col=None,
    )
    y_true = align_official_test_labels(test_windows.metadata, rul_labels)
    loader = build_sequence_loader(
        test_windows,
        batch_size=cfg.sequence.batch_size,
        shuffle=False,
    )
    y_pred = np.clip(predict_windows(model, loader, device), 0.0, None)
    metrics = regression_metrics(y_true, y_pred)
    predictions = _prediction_frame(test_windows, y_true.to_numpy(dtype=np.float64), y_pred)
    return metrics, predictions


def _checkpoint_payload(
    cfg: ProjectConfig,
    model: GRURULRegressor,
    normalizer: SequenceNormalizer,
    feature_cols: list[str],
) -> dict[str, object]:
    """Build checkpoint payload.

    Args:
        cfg: Project config.
        model: Fitted sequence model.
        normalizer: Fitted sequence normalizer.
        feature_cols: Ordered feature columns.

    Returns:
        Torch-serializable checkpoint payload.
    """
    return {
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "sequence_config": cfg.sequence.model_dump(mode="json"),
        "normalizer_means": normalizer.means_.to_dict(),
        "normalizer_stds": normalizer.stds_.to_dict(),
        "fd_subset": cfg.data.fd_subset,
        "random_seed": cfg.data.random_seed,
    }


def main() -> None:
    """Train, evaluate, and persist a GRU sequence model run."""
    args = _parse_args()
    cfg = load_config(args.config)
    if cfg.sequence.architecture != "gru":
        raise ValueError(f"Unsupported sequence architecture: {cfg.sequence.architecture}")

    feature_cols = default_feature_cols()
    device = resolve_device(cfg.sequence.device)
    train_raw = load_raw_train(cfg.data)
    train_labeled = add_rul_column(train_raw, max_rul=cfg.data.max_rul)
    train_df, val_df = split_by_engine(
        train_labeled,
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
    )

    normalizer = SequenceNormalizer(feature_cols=feature_cols)
    train_norm = normalizer.fit_transform(train_df)
    val_norm = normalizer.transform(val_df)

    train_windows = build_sliding_windows(
        train_norm,
        feature_cols,
        window_size=cfg.sequence.window_size,
    )
    validation_final_windows = build_final_windows(
        val_norm,
        feature_cols,
        window_size=cfg.sequence.window_size,
    )
    validation_windows = build_sliding_windows(
        val_norm,
        feature_cols,
        window_size=cfg.sequence.window_size,
    )

    train_loader = build_sequence_loader(
        train_windows,
        batch_size=cfg.sequence.batch_size,
        shuffle=True,
    )
    validation_final_loader = build_sequence_loader(
        validation_final_windows,
        batch_size=cfg.sequence.batch_size,
        shuffle=False,
    )
    validation_windows_loader = build_sequence_loader(
        validation_windows,
        batch_size=cfg.sequence.batch_size,
        shuffle=False,
    )

    model = GRURULRegressor(
        input_size=len(feature_cols),
        hidden_size=cfg.sequence.hidden_size,
        num_layers=cfg.sequence.num_layers,
        dropout=cfg.sequence.dropout,
    )
    result = train_gru_model(
        model=model,
        train_loader=train_loader,
        validation_final_loader=validation_final_loader,
        validation_windows_loader=validation_windows_loader,
        config=cfg.sequence,
        device=device,
        random_seed=cfg.data.random_seed,
    )

    final_metrics, final_predictions = _evaluate_windows(
        result.model,
        validation_final_windows,
        cfg.sequence.batch_size,
        device,
    )
    window_metrics, window_predictions = _evaluate_windows(
        result.model,
        validation_windows,
        cfg.sequence.batch_size,
        device,
    )

    run_dir = create_run_dir(cfg.sequence.artifact_dir, "sequence_gru")
    metrics_payload: dict[str, object] = {
        "validation_final_window": final_metrics,
        "validation_windows": window_metrics,
    }

    official = _evaluate_official_test(cfg, result.model, normalizer, feature_cols, device)
    if official is not None:
        official_metrics, official_predictions = official
        metrics_payload["official_test"] = official_metrics
        save_predictions(official_predictions, run_dir / "official_test_predictions.csv")
    else:
        print("official test evaluation skipped: test or RUL files not found")

    torch.save(
        _checkpoint_payload(cfg, result.model, normalizer, feature_cols),
        run_dir / "model.pt",
    )
    save_json(metrics_payload, run_dir / "metrics.json")
    save_json(_config_to_dict(cfg), run_dir / "config.json")
    result.history.to_csv(run_dir / "training_history.csv", index=False)
    save_predictions(
        final_predictions,
        run_dir / "validation_final_window_predictions.csv",
    )
    save_predictions(
        window_predictions,
        run_dir / "validation_window_predictions.csv",
    )

    print(f"run_dir: {run_dir}")
    print(f"validation_final_window rmse: {final_metrics['rmse']:.6f}")
    print(f"validation_final_window mae: {final_metrics['mae']:.6f}")
    print(
        "validation_final_window phm08_score: "
        f"{final_metrics['phm08_score']:.6f}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
pytest tests/models/test_train_sequence_gru_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/train_sequence_gru.py tests/models/test_train_sequence_gru_cli.py
git commit -m "feat: add sequence GRU training CLI"
```

---

## Task 8: Full Quality Gate and Integration Fixes

**Files:**
- Modify only files touched by Tasks 1-7 if verification finds issues.

- [ ] **Step 1: Run focused sequence tests**

Run:

```powershell
pytest tests/sequences tests/models/test_gru.py tests/models/test_sequence_training.py tests/models/test_train_sequence_gru_cli.py -v
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
pytest
```

Expected: PASS. The previous baseline was 127 tests; the new count should be higher after adding sequence tests.

- [ ] **Step 3: Run lint**

Run:

```powershell
ruff check src/ tests/ scripts/
```

Expected: PASS with no lint violations.

- [ ] **Step 4: Run strict type checks**

Run:

```powershell
mypy src/turbofan
```

Expected: PASS with no type errors.

- [ ] **Step 5: Fix any verification failures narrowly**

If any command fails, inspect the exact failure and patch only the files touched by this milestone. Re-run the failing command until it passes, then re-run all four verification commands from Steps 1-4.

- [ ] **Step 6: Commit final integration fixes if any were needed**

If Step 5 changed files, run:

```powershell
git add pyproject.toml configs/default.yaml src/turbofan tests scripts
git commit -m "fix: stabilize sequence model milestone"
```

If Step 5 made no changes, do not create an empty commit.

---

## Scope Guardrails

- Do not add MLflow, Weights & Biases, model registry integration, or remote tracking.
- Do not add FastAPI, batch scoring, serving contracts, or deployment documentation.
- Do not add LSTM, TCN, transformer, architecture registry, or model-comparison code.
- Do not add Optuna or hyperparameter search.
- Do not add padding or masking for engines shorter than `window_size`; skip those engines as specified.
- Do not replace or refactor `scripts/train_baseline.py` except if a narrow import conflict appears.
- Do not claim real NASA benchmark performance in docs or tests.

---

## Final Verification Checklist

Before handoff, confirm these commands passed in the `mlops` conda environment:

```powershell
ruff check src/ tests/ scripts/
mypy src/turbofan
pytest
```

Also confirm:

- `scripts/train_sequence_gru.py` writes `model.pt`, `metrics.json`, `config.json`, `training_history.csv`, `validation_final_window_predictions.csv`, and `validation_window_predictions.csv`.
- `metrics.json` contains `validation_final_window` and `validation_windows`.
- Official test artifacts are written only when `test_*.txt` and `RUL_*.txt` exist.
- No deployment or MLflow code was introduced.
