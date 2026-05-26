# Target Normalization & Feature Engineering Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize GRU training targets to [0, 1] via `max_rul` and build a feature engineering sweep script that evaluates raw, rolling, and correlation-filtered feature sets.

**Architecture:** Add `max_rul: int` to `_train_one_epoch` (divides targets), `_evaluate_loader` (rescales predictions), `predict_windows` (rescales predictions), and `train_gru_model` (passes through). Create `select_correlated_sensors` for correlation-based feature filtering. Build `sweep_feature_gru.py` mirroring the existing `sweep_sequence_gru.py`.

**Tech Stack:** Python 3.12, PyTorch, pandas, NumPy, scikit-learn

---

## Parallelism

- **Group A** (sequential): Task 1 → Task 2 → Task 3
- **Group B** (independent of Group A): Task 4
- **Group C** (after A and B): Task 5

Tasks 1-3 and Task 4 can run in parallel. Task 5 depends on all others.

---

## Task 1: Target normalization in `sequence_training.py`

**Files:**
- Modify: `src/turbofan/models/sequence_training.py`
- Test: `tests/models/test_sequence_training.py`

- [ ] **Step 1: Write new tests for target normalization**

Add the following three tests and one import to `tests/models/test_sequence_training.py`:

Add `_train_one_epoch` to the existing import block:

```python
from turbofan.models.sequence_training import (
    TrainingResult,
    _evaluate_loader,
    _train_one_epoch,
    predict_windows,
    resolve_device,
    train_gru_model,
)
```

Add `nn` to torch imports:

```python
from torch import nn
```

Add these test functions at the end of the file:

```python
class _ConstantRegressor(torch.nn.Module):
    """Fixed-output regressor for target normalization tests."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = value

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return a constant prediction for every window.

        Args:
            features: Sequence feature batch.

        Returns:
            One constant prediction per window.
        """
        return torch.full(
            (features.shape[0],), self._value, dtype=torch.float32
        )


def test_train_one_epoch_normalized_targets_produce_smaller_loss() -> None:
    """Target normalization by max_rul reduces reported training loss."""
    torch.manual_seed(42)
    model_a = GRURULRegressor(
        input_size=2, hidden_size=4, num_layers=1, dropout=0.0
    )
    torch.manual_seed(42)
    model_b = GRURULRegressor(
        input_size=2, hidden_size=4, num_layers=1, dropout=0.0
    )
    criterion = nn.MSELoss()
    device = torch.device("cpu")

    loss_identity = _train_one_epoch(
        model_a,
        _loader(),
        criterion,
        torch.optim.Adam(model_a.parameters(), lr=0.001),
        device,
        max_rul=1,
    )
    loss_normalized = _train_one_epoch(
        model_b,
        _loader(),
        criterion,
        torch.optim.Adam(model_b.parameters(), lr=0.001),
        device,
        max_rul=125,
    )

    assert loss_normalized < loss_identity


def test_evaluate_loader_rescales_predictions_by_max_rul() -> None:
    """Evaluation multiplies raw model output by max_rul before metrics."""
    targets = torch.tensor([10.0, 10.0], dtype=torch.float32)
    features = torch.zeros((2, 1, 1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(features, targets), batch_size=2)
    model = _ConstantRegressor(0.1)

    metrics = _evaluate_loader(model, loader, torch.device("cpu"), max_rul=100)
    assert metrics["rmse"] == pytest.approx(0.0)

    metrics_identity = _evaluate_loader(
        model, loader, torch.device("cpu"), max_rul=1
    )
    assert metrics_identity["rmse"] == pytest.approx(9.9)


def test_predict_windows_rescales_by_max_rul() -> None:
    """Predictions are multiplied by max_rul before returning."""
    model = GRURULRegressor(
        input_size=2, hidden_size=4, num_layers=1, dropout=0.0
    )

    preds_identity = predict_windows(
        model, _loader(), torch.device("cpu"), max_rul=1
    )
    preds_scaled = predict_windows(
        model, _loader(), torch.device("cpu"), max_rul=10
    )

    np.testing.assert_allclose(preds_scaled, preds_identity * 10, rtol=1e-5)
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/models/test_sequence_training.py::test_train_one_epoch_normalized_targets_produce_smaller_loss tests/models/test_sequence_training.py::test_evaluate_loader_rescales_predictions_by_max_rul tests/models/test_sequence_training.py::test_predict_windows_rescales_by_max_rul -v`

Expected: FAIL (unexpected keyword argument `max_rul`)

- [ ] **Step 3: Implement target normalization in all four functions**

In `src/turbofan/models/sequence_training.py`, make these changes:

**`train_gru_model`** — add `max_rul: int` parameter and pass it through:

```python
def train_gru_model(
    model: GRURULRegressor,
    train_loader: SequenceLoader,
    validation_final_loader: SequenceLoader,
    validation_windows_loader: SequenceLoader,
    config: SequenceConfig,
    device: torch.device,
    random_seed: int,
    max_rul: int,
) -> TrainingResult:
    """Train a GRU RUL regressor with validation metrics and early stopping.

    Args:
        model: Unfitted GRU model.
        train_loader: Mini-batch loader for training windows.
        validation_final_loader: Evaluation loader for final validation windows.
        validation_windows_loader: Evaluation loader for all validation windows.
        config: Sequence model training configuration.
        device: Torch device used for training and evaluation.
        random_seed: Seed for Python, NumPy, and torch random generators.
        max_rul: Maximum RUL cap for target normalization.

    Returns:
        Training result containing the best restored model and metric history.
    """
```

Inside the training loop, pass `max_rul` to both helpers:

```python
        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_rul
        )
        final_metrics = _evaluate_loader(
            model, validation_final_loader, device, max_rul
        )
        window_metrics = _evaluate_loader(
            model, validation_windows_loader, device, max_rul
        )
```

**`predict_windows`** — add `max_rul: int` and rescale:

```python
def predict_windows(
    model: GRURULRegressor,
    loader: SequenceLoader,
    device: torch.device,
    max_rul: int,
) -> npt.NDArray[np.float64]:
    """Predict RUL values for sequence windows.

    Args:
        model: Trained GRU model.
        loader: Loader containing sequence feature batches.
        device: Torch device used for inference.
        max_rul: Maximum RUL cap used to rescale raw model outputs.

    Returns:
        One-dimensional float64 array of predictions in real RUL units.
    """
    predictions, _ = _predict_windows_and_targets(model, loader, device)
    return predictions * max_rul
```

**`_train_one_epoch`** — add `max_rul: int` and normalize targets:

```python
def _train_one_epoch(
    model: GRURULRegressor,
    loader: SequenceLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_rul: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device) / max_rul
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_count += batch_size
    if total_count == 0:
        return 0.0
    return total_loss / total_count
```

**`_evaluate_loader`** — add `max_rul: int` and rescale predictions:

```python
def _evaluate_loader(
    model: GRURULRegressor,
    loader: SequenceLoader,
    device: torch.device,
    max_rul: int,
) -> dict[str, float]:
    predictions, targets = _predict_windows_and_targets(model, loader, device)
    predictions = predictions * max_rul
    predictions = np.clip(predictions, 0.0, None)
    return regression_metrics(targets, predictions)
```

- [ ] **Step 4: Update existing tests to pass `max_rul=1`**

In `tests/models/test_sequence_training.py`, update every existing call site:

`test_evaluate_loader_keeps_predictions_and_targets_paired`:
```python
    metrics = _evaluate_loader(model, loader, torch.device("cpu"), max_rul=1)
```

`test_evaluate_loader_clips_negative_predictions_before_metrics`:
```python
    metrics = _evaluate_loader(model, loader, torch.device("cpu"), max_rul=1)
```

`test_predict_windows_returns_float64_prediction_per_window`:
```python
    predictions = predict_windows(model, _loader(), torch.device("cpu"), max_rul=1)
```

`test_train_gru_model_returns_result_with_expected_history`:
```python
    result = train_gru_model(
        model=model,
        train_loader=_loader(shuffle=True),
        validation_final_loader=_loader(),
        validation_windows_loader=_loader(),
        config=config,
        device=torch.device("cpu"),
        random_seed=7,
        max_rul=1,
    )
```

`test_train_gru_model_restores_best_state_after_early_stopping` — two updates:
```python
    result = train_gru_model(
        model=model,
        train_loader=train_loader,
        validation_final_loader=validation_loader,
        validation_windows_loader=validation_windows_loader,
        config=config,
        device=torch.device("cpu"),
        random_seed=7,
        max_rul=1,
    )
    restored_predictions = predict_windows(
        model,
        validation_windows_loader,
        torch.device("cpu"),
        max_rul=1,
    )
```

- [ ] **Step 5: Run all tests and verify they pass**

Run: `pytest tests/models/test_sequence_training.py -v`

Expected: ALL PASS

- [ ] **Step 6: Lint and type-check**

Run: `ruff check src/turbofan/models/sequence_training.py tests/models/test_sequence_training.py && mypy src/turbofan/models/sequence_training.py`

- [ ] **Step 7: Commit**

```bash
git add src/turbofan/models/sequence_training.py tests/models/test_sequence_training.py
git commit -m "feat(gru): add max_rul target normalization to training pipeline"
```

---

## Task 2: Update `sweep_sequence_gru.py` to pass `max_rul`

**Files:**
- Modify: `scripts/sweep_sequence_gru.py`
- Modify: `tests/models/test_sweep_sequence_gru.py`

- [ ] **Step 1: Pass `max_rul` in `run_gru_sweep`**

In `scripts/sweep_sequence_gru.py`, update the `train_gru_model` call (around line 249):

```python
        result = train_gru_model(
            model=model,
            train_loader=train_loader,
            validation_final_loader=validation_final_loader,
            validation_windows_loader=validation_windows_loader,
            config=spec_cfg,
            device=torch_device,
            random_seed=cfg.data.random_seed,
            max_rul=cfg.data.max_rul,
        )
```

Update the `predict_windows` call (around line 261):

```python
        predictions = np.clip(
            predict_windows(
                result.model,
                validation_windows_loader,
                torch_device,
                max_rul=cfg.data.max_rul,
            ),
            0.0,
            None,
        )
```

- [ ] **Step 2: Update test fakes to accept `max_rul`**

In `tests/models/test_sweep_sequence_gru.py`, update **both** `fake_predict_windows` functions (in `test_gru_sweep_reports_validation_window_metrics` and `test_gru_sweep_appends_training_log_entry_per_completed_config`) to accept the new parameter:

```python
    def fake_predict_windows(
        model: object,
        loader: FakeWindows,
        device: object,
        max_rul: int = 1,
    ) -> np.ndarray:
```

- [ ] **Step 3: Run tests and verify they pass**

Run: `pytest tests/models/test_sweep_sequence_gru.py -v`

Expected: ALL PASS

- [ ] **Step 4: Lint and type-check**

Run: `ruff check scripts/sweep_sequence_gru.py tests/models/test_sweep_sequence_gru.py`

- [ ] **Step 5: Commit**

```bash
git add scripts/sweep_sequence_gru.py tests/models/test_sweep_sequence_gru.py
git commit -m "feat(sweep): pass max_rul through GRU hyperparameter sweep"
```

---

## Task 3: Update `train_sequence_gru.py` to pass `max_rul`

**Files:**
- Modify: `scripts/train_sequence_gru.py`
- Modify: `tests/models/test_train_sequence_gru_cli.py`

- [ ] **Step 1: Update `_evaluate_windows` to accept `max_rul`**

In `scripts/train_sequence_gru.py`, add `max_rul: int` and pass it to `predict_windows`:

```python
def _evaluate_windows(
    model: GRURULRegressor,
    windows: WindowedSequences,
    device: torch.device,
    batch_size: int,
    max_rul: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate labeled sequence windows.

    Args:
        model: Trained GRU model.
        windows: Labeled sequence windows.
        device: Torch device used for inference.
        batch_size: Prediction batch size.
        max_rul: Maximum RUL cap for prediction rescaling.

    Returns:
        Metrics and prediction artifact rows.
    """
    loader = build_sequence_loader(windows, batch_size=batch_size, shuffle=False)
    y_pred = np.clip(
        predict_windows(model, loader, device, max_rul=max_rul), 0.0, None
    )
    y_true = windows.y.astype(np.float64)
    metrics = regression_metrics(y_true, y_pred)
    return metrics, _prediction_frame(windows, y_true, y_pred)
```

- [ ] **Step 2: Update `_evaluate_official_test` to pass `max_rul`**

Use `cfg.data.max_rul` (already available via `cfg` parameter) in the `predict_windows` call:

```python
    y_pred = np.clip(
        predict_windows(model, loader, device, max_rul=cfg.data.max_rul),
        0.0,
        None,
    )
```

- [ ] **Step 3: Update `main()` to pass `max_rul` everywhere**

Pass `max_rul` to `train_gru_model`:

```python
    result = train_gru_model(
        model=model,
        train_loader=train_loader,
        validation_final_loader=validation_final_loader,
        validation_windows_loader=validation_windows_loader,
        config=cfg.sequence,
        device=device,
        random_seed=cfg.data.random_seed,
        max_rul=cfg.data.max_rul,
    )
```

Pass `max_rul` to both `_evaluate_windows` calls:

```python
    final_metrics, final_predictions = _evaluate_windows(
        result.model,
        validation_final_windows,
        device,
        cfg.sequence.batch_size,
        max_rul=cfg.data.max_rul,
    )
    window_metrics, window_predictions = _evaluate_windows(
        result.model,
        validation_windows,
        device,
        cfg.sequence.batch_size,
        max_rul=cfg.data.max_rul,
    )
```

- [ ] **Step 4: Update test fake that has explicit keyword-only signature**

In `tests/models/test_train_sequence_gru_cli.py`, update the `fake_train_gru_model` in `test_train_sequence_gru_cli_seeds_model_initialization` to accept `max_rul`:

```python
    def fake_train_gru_model(
        *,
        model: GRURULRegressor,
        train_loader: object,
        validation_final_loader: object,
        validation_windows_loader: object,
        config: SequenceConfig,
        device: torch.device,
        random_seed: int,
        max_rul: int,
    ) -> TrainingResult:
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `pytest tests/models/test_train_sequence_gru_cli.py -v`

Expected: ALL PASS

- [ ] **Step 6: Lint and type-check**

Run: `ruff check scripts/train_sequence_gru.py tests/models/test_train_sequence_gru_cli.py`

- [ ] **Step 7: Commit**

```bash
git add scripts/train_sequence_gru.py tests/models/test_train_sequence_gru_cli.py
git commit -m "feat(training): pass max_rul through GRU training CLI"
```

---

## Task 4: Correlation-based feature selection (independent — can run in parallel with Tasks 1-3)

**Files:**
- Create: `src/turbofan/sequences/feature_selection.py`
- Create: `tests/sequences/test_feature_selection.py`

- [ ] **Step 1: Write failing tests**

Create `tests/sequences/test_feature_selection.py`:

```python
"""Tests for correlation-based feature selection."""
from __future__ import annotations

import pandas as pd
import pytest

from turbofan.sequences.feature_selection import select_correlated_sensors


def _sample_df() -> pd.DataFrame:
    """Build a DataFrame with known sensor-RUL correlations.

    Returns:
        DataFrame where s_1 has |r|=1.0, s_2 is constant (NaN),
        and s_3 has |r|~0.35 with RUL.
    """
    return pd.DataFrame(
        {
            "engine_id": [1] * 5,
            "cycle": [1, 2, 3, 4, 5],
            "op_1": [4.0, 3.0, 2.0, 1.0, 0.0],
            "s_1": [10.0, 8.0, 6.0, 4.0, 2.0],
            "s_2": [5.0, 5.0, 5.0, 5.0, 5.0],
            "s_3": [3.0, 1.0, 4.0, 1.0, 5.0],
            "rul": [4.0, 3.0, 2.0, 1.0, 0.0],
        }
    )


def test_select_correlated_sensors_returns_above_threshold() -> None:
    """Only sensors with |correlation| >= threshold are returned."""
    result = select_correlated_sensors(_sample_df(), threshold=0.5)

    assert "s_1" in result
    assert "s_2" not in result
    assert "s_3" not in result


def test_select_correlated_sensors_sorts_by_descending_abs_correlation() -> None:
    """Returned sensors are sorted by descending absolute correlation."""
    df = pd.DataFrame(
        {
            "s_1": [10.0, 8.0, 6.0, 4.0, 2.0],
            "s_2": [8.0, 7.0, 5.0, 3.0, 2.0],
            "rul": [4.0, 3.0, 2.0, 1.0, 0.0],
        }
    )

    result = select_correlated_sensors(df, threshold=0.5)

    assert result[0] == "s_1"
    assert result[1] == "s_2"
    assert len(result) == 2


def test_select_correlated_sensors_excludes_non_sensor_columns() -> None:
    """Only columns starting with 's_' are considered."""
    result = select_correlated_sensors(_sample_df(), threshold=0.5)

    assert "op_1" not in result
    assert "engine_id" not in result
    assert "cycle" not in result
    assert "rul" not in result


def test_select_correlated_sensors_raises_when_none_pass() -> None:
    """ValueError is raised when no sensors meet the threshold."""
    df = pd.DataFrame(
        {
            "s_1": [5.0, 5.0, 5.0],
            "s_2": [3.0, 1.0, 4.0],
            "rul": [2.0, 1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="No sensors"):
        select_correlated_sensors(df, threshold=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/sequences/test_feature_selection.py -v`

Expected: FAIL (ImportError — module does not exist)

- [ ] **Step 3: Implement `select_correlated_sensors`**

Create `src/turbofan/sequences/feature_selection.py`:

```python
"""Correlation-based sensor feature selection."""
from __future__ import annotations

import pandas as pd


def select_correlated_sensors(
    df: pd.DataFrame,
    target_col: str = "rul",
    threshold: float = 0.5,
) -> list[str]:
    """Select sensor columns with high absolute correlation to the target.

    Computes the absolute Pearson correlation between each ``s_*`` column
    and ``target_col``. Returns sensor column names where
    ``|r| >= threshold``, sorted by descending ``|r|``.

    Args:
        df: DataFrame containing sensor and target columns.
        target_col: Name of the target column.
        threshold: Minimum absolute correlation to include a sensor.

    Returns:
        Sensor column names sorted by descending absolute correlation.

    Raises:
        ValueError: If no sensors meet the correlation threshold.
    """
    sensor_cols = [col for col in df.columns if col.startswith("s_")]
    correlations = df[sensor_cols].corrwith(df[target_col]).abs()
    selected = correlations[correlations >= threshold].sort_values(ascending=False)
    if selected.empty:
        raise ValueError(
            f"No sensors meet the correlation threshold {threshold}."
        )
    return list(selected.index)
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/sequences/test_feature_selection.py -v`

Expected: ALL PASS

- [ ] **Step 5: Lint and type-check**

Run: `ruff check src/turbofan/sequences/feature_selection.py tests/sequences/test_feature_selection.py && mypy src/turbofan/sequences/feature_selection.py`

- [ ] **Step 6: Commit**

```bash
git add src/turbofan/sequences/feature_selection.py tests/sequences/test_feature_selection.py
git commit -m "feat(sequences): add correlation-based sensor feature selection"
```

---

## Task 5: Feature engineering sweep script (depends on Tasks 1-4)

**Files:**
- Create: `scripts/sweep_feature_gru.py`
- Create: `tests/models/test_sweep_feature_gru.py`

- [ ] **Step 1: Write validation and grid tests**

Create `tests/models/test_sweep_feature_gru.py`:

```python
"""Tests for scripts/sweep_feature_gru.py."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from turbofan.config.schema import DataConfig, ProjectConfig, SequenceConfig


def _load_module(project_root: Path) -> ModuleType:
    """Load the feature sweep script as a module for helper testing.

    Args:
        project_root: Repository root path.

    Returns:
        Imported script module.

    Raises:
        RuntimeError: If the script cannot be imported.
    """
    script_path = project_root / "scripts" / "sweep_feature_gru.py"
    spec = importlib.util.spec_from_file_location("sweep_feature_gru", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    src_path = str(project_root / "src")
    cached_turbofan_modules = {
        name: loaded_module
        for name, loaded_module in sys.modules.items()
        if name == "turbofan" or name.startswith("turbofan.")
    }
    for name in cached_turbofan_modules:
        del sys.modules[name]
    sys.path.insert(0, src_path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(src_path)
        for name in list(sys.modules):
            if name == "turbofan" or name.startswith("turbofan."):
                del sys.modules[name]
        sys.modules.update(cached_turbofan_modules)
    return module


def _write_cmapps_file(path: Path, n_engines: int, n_cycles: int) -> None:
    """Write a small C-MAPSS-style whitespace-delimited file.

    Args:
        path: Destination file path.
        n_engines: Number of synthetic engines.
        n_cycles: Number of cycles per engine.
    """
    lines = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, n_cycles + 1):
            op_cols = [0.0, 0.0, 0.0]
            sensors = [
                float(cycle + sensor_idx + engine_id)
                for sensor_idx in range(1, 22)
            ]
            values = [engine_id, cycle, *op_cols, *sensors]
            lines.append(" ".join(str(value) for value in values))
    path.write_text("\n".join(lines))


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal project config for feature sweep tests.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Created config path.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)
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
                "  batch_size: 8",
                "  hidden_size: 4",
                "  num_layers: 1",
                "  dropout: 0.0",
                "  learning_rate: 0.001",
                "  epochs: 2",
                "  patience: 2",
                "  device: cpu",
            ]
        )
    )
    return cfg_path


def test_feature_sweep_validates_feature_set_names(tmp_path: Path) -> None:
    """Invalid feature set names fail before training."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)

    with pytest.raises(ValueError, match="Invalid feature sets"):
        module.run_feature_sweep(
            cfg_path, ["invalid_set"], [0.5], 10, device="cpu"
        )


def test_feature_sweep_validates_corr_thresholds(tmp_path: Path) -> None:
    """Correlation thresholds outside (0, 1) fail before training."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)

    with pytest.raises(ValueError, match="[Cc]orrelation"):
        module.run_feature_sweep(
            cfg_path, ["raw"], [0.0], 10, device="cpu"
        )
    with pytest.raises(ValueError, match="[Cc]orrelation"):
        module.run_feature_sweep(
            cfg_path, ["raw"], [1.0], 10, device="cpu"
        )


def test_feature_sweep_validates_rolling_window(tmp_path: Path) -> None:
    """Non-positive rolling window fails before training."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)

    with pytest.raises(ValueError, match="[Rr]olling"):
        module.run_feature_sweep(
            cfg_path, ["raw"], [0.5], 0, device="cpu"
        )


def test_feature_sweep_grid_produces_correct_number_of_runs(
    tmp_path: Path,
) -> None:
    """Sweep grid produces 1+1+len(thresholds)+len(thresholds) runs."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)

    grid = module._build_sweep_grid(
        ["raw", "raw_plus_rolling", "top_corr", "top_corr_rolling"],
        [0.3, 0.5, 0.7],
    )

    assert len(grid) == 8
    assert grid[0] == ("raw", None)
    assert grid[1] == ("raw_plus_rolling", None)
    assert grid[2] == ("top_corr", 0.3)
    assert grid[3] == ("top_corr", 0.5)
    assert grid[4] == ("top_corr", 0.7)
    assert grid[5] == ("top_corr_rolling", 0.3)


def test_feature_sweep_returns_expected_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Feature sweep returns a DataFrame with the expected result columns."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)
    monkeypatch.setattr(module, "append_training_log", lambda entry: None)

    results = module.run_feature_sweep(
        config_path=cfg_path,
        feature_sets=["raw"],
        corr_thresholds=[0.5],
        rolling_window=10,
        device="cpu",
    )

    assert len(results) == 1
    assert list(results.columns) == [
        "feature_set",
        "corr_threshold",
        "n_features",
        "best_epoch",
        "rmse",
        "mae",
        "phm08_score",
    ]
    assert results.iloc[0]["feature_set"] == "raw"


def test_feature_sweep_cli_writes_csv(tmp_path: Path) -> None:
    """CLI writes a sorted feature sweep CSV when output is supplied."""
    project_root = Path(__file__).parent.parent.parent
    cfg_path = _write_config(tmp_path)
    output_path = tmp_path / "feature_sweep.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "sweep_feature_gru.py"),
            "--config",
            str(cfg_path),
            "--feature-sets",
            "raw",
            "--corr-thresholds",
            "0.5",
            "--rolling-window",
            "3",
            "--device",
            "cpu",
            "--output",
            str(output_path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "run 1/1" in result.stdout
    results = pd.read_csv(output_path)
    assert len(results) == 1
    assert "feature_set" in results.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/models/test_sweep_feature_gru.py -v`

Expected: FAIL (script does not exist)

- [ ] **Step 3: Implement `sweep_feature_gru.py`**

Create `scripts/sweep_feature_gru.py`:

```python
"""Sweep GRU feature engineering configurations on the validation split."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import numpy as np
import pandas as pd

from turbofan.config.schema import load_config
from turbofan.data.loader import load_raw_train
from turbofan.features.rolling import RollingFeatureExtractor
from turbofan.models.evaluate import add_rul_column
from turbofan.models.gru import GRURULRegressor
from turbofan.models.metrics import regression_metrics
from turbofan.models.sequence_training import (
    predict_windows,
    resolve_device,
    seed_everything,
    train_gru_model,
)
from turbofan.models.split import split_by_engine
from turbofan.models.training_log import append_training_log, build_log_entry
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.feature_selection import select_correlated_sensors
from turbofan.sequences.normalize import SequenceNormalizer
from turbofan.sequences.windowing import build_final_windows, build_sliding_windows

VALID_FEATURE_SETS = frozenset(
    {"raw", "raw_plus_rolling", "top_corr", "top_corr_rolling"}
)

RESULT_COLUMNS = [
    "feature_set",
    "corr_threshold",
    "n_features",
    "best_epoch",
    "rmse",
    "mae",
    "phm08_score",
]


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
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["raw", "raw_plus_rolling", "top_corr", "top_corr_rolling"],
        help="Feature set families to sweep.",
    )
    parser.add_argument(
        "--corr-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="Correlation thresholds for filtered feature sets.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Rolling aggregation window size.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Torch device for training.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for sweep results.",
    )
    return parser.parse_args()


def _validate_inputs(
    feature_sets: list[str],
    corr_thresholds: list[float],
    rolling_window: int,
) -> None:
    """Validate feature sweep inputs.

    Args:
        feature_sets: Feature set families to sweep.
        corr_thresholds: Correlation thresholds for filtered sets.
        rolling_window: Rolling aggregation window size.

    Raises:
        ValueError: If any input is invalid.
    """
    invalid = set(feature_sets) - VALID_FEATURE_SETS
    if invalid:
        raise ValueError(f"Invalid feature sets: {sorted(invalid)}")
    if not corr_thresholds or any(
        t <= 0.0 or t >= 1.0 for t in corr_thresholds
    ):
        raise ValueError("Correlation thresholds must be in (0, 1).")
    if rolling_window <= 0:
        raise ValueError("Rolling window must be positive.")


def _build_sweep_grid(
    feature_sets: list[str],
    corr_thresholds: list[float],
) -> list[tuple[str, float | None]]:
    """Build the sweep grid as (feature_set, threshold_or_None) pairs.

    Args:
        feature_sets: Feature set families to include.
        corr_thresholds: Thresholds for correlation-based feature sets.

    Returns:
        Ordered list of (feature_set, threshold) pairs.
    """
    grid: list[tuple[str, float | None]] = []
    for fs in feature_sets:
        if fs in {"top_corr", "top_corr_rolling"}:
            for threshold in corr_thresholds:
                grid.append((fs, threshold))
        else:
            grid.append((fs, None))
    return grid


def _rolling_feature_cols(
    sensor_cols: list[str],
    rolling_window: int,
) -> list[str]:
    """Compute rolling feature column names for given sensors.

    Args:
        sensor_cols: Base sensor column names.
        rolling_window: Rolling aggregation window size.

    Returns:
        Rolling feature column names in extractor output order.
    """
    cols: list[str] = []
    for sensor in sensor_cols:
        for stat in ["rmean", "rstd", "rmin", "rmax"]:
            cols.append(f"{sensor}_{stat}_{rolling_window}")
    return cols


def _append_incremental_row(
    row: dict[str, object],
    output_path: Path,
    *,
    append: bool,
) -> None:
    """Append one completed result row to an incremental CSV.

    Args:
        row: Completed sweep result row.
        output_path: Destination CSV path.
        append: Whether to append to an existing incremental CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(
        output_path,
        mode="a" if append else "w",
        header=not append,
        index=False,
    )


def _device_name(device: object) -> str:
    """Return a stable display name for a resolved training device.

    Args:
        device: Resolved device object.

    Returns:
        Device type string when available, otherwise ``str(device)``.
    """
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type
    return str(device)


def run_feature_sweep(
    config_path: Path,
    feature_sets: list[str],
    corr_thresholds: list[float],
    rolling_window: int,
    device: str,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Train and evaluate GRU models across feature engineering configurations.

    Args:
        config_path: Project config path.
        feature_sets: Feature set families to evaluate.
        corr_thresholds: Correlation thresholds for filtered feature sets.
        rolling_window: Rolling aggregation window size.
        device: Requested torch device, either ``"cpu"`` or ``"cuda"``.
        output_path: Optional CSV path for incremental and final results.

    Returns:
        Results sorted by validation PHM08 score.

    Raises:
        ValueError: If sweep inputs are invalid.
    """
    _validate_inputs(feature_sets, corr_thresholds, rolling_window)
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'.")

    cfg = load_config(config_path)
    if cfg.sequence.architecture != "gru":
        raise ValueError("Feature sweep requires sequence architecture='gru'.")

    torch_device = resolve_device(cast(Literal["cpu", "cuda"], device))
    all_sensors = [f"s_{i}" for i in range(1, 22)]
    ops = ["op_1", "op_2", "op_3"]

    train_raw = load_raw_train(cfg.data)
    train_labeled = add_rul_column(train_raw, max_rul=cfg.data.max_rul)
    train_df, val_df = split_by_engine(
        train_labeled,
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
    )

    grid = _build_sweep_grid(feature_sets, corr_thresholds)
    total_runs = len(grid)
    rows: list[dict[str, object]] = []

    for run_idx, (feature_set, corr_threshold) in enumerate(grid, 1):
        needs_rolling = feature_set in {"raw_plus_rolling", "top_corr_rolling"}

        if feature_set in {"top_corr", "top_corr_rolling"}:
            assert corr_threshold is not None
            sensor_cols = select_correlated_sensors(
                train_df, target_col="rul", threshold=corr_threshold
            )
        else:
            sensor_cols = all_sensors

        base_feature_cols = ops + sensor_cols

        if needs_rolling:
            extractor = RollingFeatureExtractor(windows=[rolling_window])
            train_for_norm = extractor.fit(train_df).transform(train_df)
            val_for_norm = extractor.transform(val_df)
            rolling_cols = _rolling_feature_cols(sensor_cols, rolling_window)
            feature_cols = base_feature_cols + rolling_cols
        else:
            feature_cols = base_feature_cols
            train_for_norm = train_df
            val_for_norm = val_df

        normalizer = SequenceNormalizer(feature_cols=feature_cols)
        train_normalized = normalizer.fit_transform(train_for_norm)
        val_normalized = normalizer.transform(val_for_norm)

        train_windows = build_sliding_windows(
            train_normalized,
            feature_cols=feature_cols,
            window_size=cfg.sequence.window_size,
        )
        validation_final_windows = build_final_windows(
            val_normalized,
            feature_cols=feature_cols,
            window_size=cfg.sequence.window_size,
        )
        validation_windows = build_sliding_windows(
            val_normalized,
            feature_cols=feature_cols,
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

        seed_everything(cfg.data.random_seed)
        model = GRURULRegressor(
            input_size=len(feature_cols),
            hidden_size=cfg.sequence.hidden_size,
            num_layers=cfg.sequence.num_layers,
            dropout=cfg.sequence.dropout,
        )
        training_start = perf_counter()
        result = train_gru_model(
            model=model,
            train_loader=train_loader,
            validation_final_loader=validation_final_loader,
            validation_windows_loader=validation_windows_loader,
            config=cfg.sequence,
            device=torch_device,
            random_seed=cfg.data.random_seed,
            max_rul=cfg.data.max_rul,
        )
        training_duration_seconds = perf_counter() - training_start

        predictions = np.clip(
            predict_windows(
                result.model,
                validation_windows_loader,
                torch_device,
                max_rul=cfg.data.max_rul,
            ),
            0.0,
            None,
        )
        metrics = regression_metrics(
            validation_windows.y.astype(np.float64),
            predictions,
        )

        row: dict[str, object] = {
            "feature_set": feature_set,
            "corr_threshold": corr_threshold,
            "n_features": len(feature_cols),
            "best_epoch": result.best_epoch,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "phm08_score": metrics["phm08_score"],
        }
        rows.append(row)

        if output_path is not None:
            _append_incremental_row(row, output_path, append=len(rows) > 1)

        log_entry = build_log_entry(
            model_type="gru",
            dataset=cfg.data.fd_subset,
            random_seed=cfg.data.random_seed,
            hyperparameters={
                "window_size": cfg.sequence.window_size,
                "hidden_size": cfg.sequence.hidden_size,
                "learning_rate": cfg.sequence.learning_rate,
                "num_layers": cfg.sequence.num_layers,
                "dropout": cfg.sequence.dropout,
                "batch_size": cfg.sequence.batch_size,
                "epochs": cfg.sequence.epochs,
                "patience": cfg.sequence.patience,
            },
            metrics=metrics,
            training_duration_seconds=training_duration_seconds,
            device=_device_name(torch_device),
            run_dir=None,
            best_epoch=result.best_epoch,
            extra={
                "feature_set": feature_set,
                "corr_threshold": corr_threshold,
                "n_features": len(feature_cols),
                "rolling_window": rolling_window if needs_rolling else None,
            },
        )
        append_training_log(log_entry)

        print(
            f"run {run_idx}/{total_runs}: "
            f"feature_set={feature_set} "
            f"corr_threshold={corr_threshold} "
            f"n_features={len(feature_cols)} "
            f"phm08_score={metrics['phm08_score']:.6f}"
        )

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return results.sort_values("phm08_score").reset_index(drop=True)


def main() -> None:
    """Run the GRU feature engineering sweep CLI."""
    args = _parse_args()
    results = run_feature_sweep(
        config_path=args.config,
        feature_sets=args.feature_sets,
        corr_thresholds=args.corr_thresholds,
        rolling_window=args.rolling_window,
        device=args.device,
        output_path=args.output,
    )
    print(results.to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/models/test_sweep_feature_gru.py -v`

Expected: ALL PASS

- [ ] **Step 5: Lint and type-check**

Run: `ruff check scripts/sweep_feature_gru.py tests/models/test_sweep_feature_gru.py && mypy scripts/sweep_feature_gru.py`

- [ ] **Step 6: Commit**

```bash
git add scripts/sweep_feature_gru.py tests/models/test_sweep_feature_gru.py
git commit -m "feat(sweep): add feature engineering sweep script"
```

---

## Final Verification

After all tasks are complete:

```bash
pytest
ruff check src/ tests/ scripts/
mypy src/turbofan
```

All must pass before the branch is considered complete.
