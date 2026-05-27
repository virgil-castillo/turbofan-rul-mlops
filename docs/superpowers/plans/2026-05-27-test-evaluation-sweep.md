# Test Evaluation Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add official C-MAPSS test set evaluation to both GRU sweep scripts and extract shared test evaluation logic into a reusable module.

**Architecture:** Extract `_evaluate_official_test` and `_align_official_labels_to_eligible_engines` from `scripts/train_sequence_gru.py` into a new `src/turbofan/models/test_evaluation.py` module. The module exposes three public functions: `align_labels_to_eligible_engines` (label alignment), `evaluate_official_test` (file-loading entry point), and `evaluate_test_from_df` (pre-loaded DataFrame entry point). Both sweep scripts call the shared module after each training run and add `test_rmse`, `test_mae`, `test_phm08_score` columns. The training script is refactored to use the shared module.

**Tech Stack:** Python, pandas, numpy, PyTorch, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/turbofan/models/test_evaluation.py` | Create | Shared test evaluation: label alignment, file-loading eval, DataFrame eval |
| `tests/models/test_test_evaluation.py` | Create | Unit tests for the shared module |
| `scripts/train_sequence_gru.py` | Modify | Replace private helpers with shared module calls |
| `tests/models/test_train_sequence_gru_cli.py` | Modify | Update monkeypatches for refactored imports |
| `scripts/sweep_sequence_gru.py` | Modify | Add test evaluation per run, new result columns |
| `tests/models/test_sweep_sequence_gru.py` | Modify | Update column assertions, add test eval coverage |
| `scripts/sweep_feature_gru.py` | Modify | Add test evaluation per run with feature engineering |
| `tests/models/test_sweep_feature_gru.py` | Modify | Update column assertions, add test eval coverage |

---

### Task 1: Create shared test evaluation module — tests

**Files:**
- Create: `tests/models/test_test_evaluation.py`

- [ ] **Step 1: Write test for `align_labels_to_eligible_engines` — happy path**

```python
"""Tests for turbofan.models.test_evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.models.test_evaluation import align_labels_to_eligible_engines


class TestAlignLabelsToEligibleEngines:
    """Tests for align_labels_to_eligible_engines."""

    def test_selects_labels_for_eligible_engines(self) -> None:
        """Labels for eligible engine IDs are selected and aligned."""
        metadata = pd.DataFrame({"engine_id": [1, 3], "cycle": [10, 20]})
        rul_labels = pd.Series([100, 200, 300], name="rul")

        result = align_labels_to_eligible_engines(metadata, rul_labels)

        assert list(result) == [100.0, 300.0]
        assert result.name == "rul"
        assert result.dtype == np.float64
```

- [ ] **Step 2: Write test for `align_labels_to_eligible_engines` — out-of-range engine ID**

Append to the same test class in `tests/models/test_test_evaluation.py`:

```python
    def test_raises_on_out_of_range_engine_id(self) -> None:
        """Engine IDs beyond available labels raise ValueError."""
        metadata = pd.DataFrame({"engine_id": [1, 5], "cycle": [10, 20]})
        rul_labels = pd.Series([100, 200, 300], name="rul")

        with pytest.raises(ValueError, match="eligible test engine"):
            align_labels_to_eligible_engines(metadata, rul_labels)
```

- [ ] **Step 3: Write test for `evaluate_test_from_df` — returns metrics dict**

Add a new test class below the existing one:

```python
import torch

from turbofan.models.gru import GRURULRegressor
from turbofan.models.test_evaluation import evaluate_test_from_df
from turbofan.sequences.normalize import SequenceNormalizer


class TestEvaluateTestFromDf:
    """Tests for evaluate_test_from_df."""

    def test_returns_test_metrics(self) -> None:
        """Returns dict with test_rmse, test_mae, test_phm08_score keys."""
        n_engines = 3
        n_cycles = 5
        feature_cols = ["op_1", "s_1", "s_2"]

        rows = []
        for eid in range(1, n_engines + 1):
            for cycle in range(1, n_cycles + 1):
                rows.append({
                    "engine_id": eid,
                    "cycle": cycle,
                    "op_1": 0.0,
                    "s_1": float(cycle),
                    "s_2": float(cycle * 2),
                })
        test_df = pd.DataFrame(rows)
        rul_labels = pd.Series([10, 20, 30], name="rul")

        normalizer = SequenceNormalizer(feature_cols=feature_cols)
        normalizer.fit_transform(test_df.copy())

        model = GRURULRegressor(
            input_size=len(feature_cols),
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
        )
        device = torch.device("cpu")

        result = evaluate_test_from_df(
            test_df=test_df,
            rul_labels=rul_labels,
            model=model,
            normalizer=normalizer,
            feature_cols=feature_cols,
            device=device,
            window_size=3,
            batch_size=8,
            max_rul=125,
        )

        assert set(result.keys()) == {"test_rmse", "test_mae", "test_phm08_score"}
        assert all(isinstance(v, float) for v in result.values())
        assert all(v >= 0.0 for v in result.values())
```

- [ ] **Step 4: Write test for `evaluate_official_test` — returns None on missing files**

```python
from turbofan.config.schema import DataConfig
from turbofan.models.test_evaluation import evaluate_official_test


class TestEvaluateOfficialTest:
    """Tests for evaluate_official_test."""

    def test_returns_none_when_test_files_missing(self, tmp_path: Path) -> None:
        """Returns None when test or RUL files do not exist."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        data_config = DataConfig(
            raw_dir=raw_dir,
            processed_dir=tmp_path / "processed",
            interim_dir=tmp_path / "interim",
        )
        feature_cols = ["op_1", "s_1"]
        normalizer = SequenceNormalizer(feature_cols=feature_cols)

        rows = []
        for eid in range(1, 3):
            for cycle in range(1, 6):
                rows.append({
                    "engine_id": eid,
                    "cycle": cycle,
                    "op_1": 0.0,
                    "s_1": float(cycle),
                })
        normalizer.fit_transform(pd.DataFrame(rows))

        model = GRURULRegressor(
            input_size=len(feature_cols),
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
        )

        result = evaluate_official_test(
            data_config=data_config,
            model=model,
            normalizer=normalizer,
            feature_cols=feature_cols,
            device=torch.device("cpu"),
            window_size=3,
            batch_size=8,
        )

        assert result is None
```

- [ ] **Step 5: Write test for `evaluate_official_test` — happy path with files**

```python
    def test_returns_metrics_when_files_exist(self, tmp_path: Path) -> None:
        """Returns test metrics dict when test and RUL files exist."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        n_engines = 3
        n_cycles = 5
        lines = []
        for eid in range(1, n_engines + 1):
            for cycle in range(1, n_cycles + 1):
                op_cols = [0.0, 0.0, 0.0]
                sensors = [float(cycle + si) for si in range(1, 22)]
                values = [eid, cycle, *op_cols, *sensors]
                lines.append(" ".join(str(v) for v in values))
        (raw_dir / "test_FD001.txt").write_text("\n".join(lines))
        (raw_dir / "RUL_FD001.txt").write_text("10\n20\n30\n")

        train_lines = list(lines)
        (raw_dir / "train_FD001.txt").write_text("\n".join(train_lines))

        data_config = DataConfig(
            raw_dir=raw_dir,
            processed_dir=tmp_path / "processed",
            interim_dir=tmp_path / "interim",
            fd_subset="FD001",
            max_rul=125,
        )
        feature_cols = [
            "op_1", "op_2", "op_3",
            *[f"s_{i}" for i in range(1, 22)],
        ]
        normalizer = SequenceNormalizer(feature_cols=feature_cols)

        from turbofan.data.loader import load_raw_train
        train_raw = load_raw_train(data_config)
        normalizer.fit_transform(train_raw)

        model = GRURULRegressor(
            input_size=len(feature_cols),
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
        )

        result = evaluate_official_test(
            data_config=data_config,
            model=model,
            normalizer=normalizer,
            feature_cols=feature_cols,
            device=torch.device("cpu"),
            window_size=3,
            batch_size=8,
        )

        assert result is not None
        assert set(result.keys()) == {"test_rmse", "test_mae", "test_phm08_score"}
        assert all(isinstance(v, float) for v in result.values())
```

- [ ] **Step 6: Add missing import at top of test file**

The file needs `from pathlib import Path` at the top. Ensure the full import block is:

```python
"""Tests for turbofan.models.test_evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from turbofan.config.schema import DataConfig
from turbofan.models.gru import GRURULRegressor
from turbofan.models.test_evaluation import (
    align_labels_to_eligible_engines,
    evaluate_official_test,
    evaluate_test_from_df,
)
from turbofan.sequences.normalize import SequenceNormalizer
```

- [ ] **Step 7: Run tests to confirm they fail**

Run: `pytest tests/models/test_test_evaluation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'turbofan.models.test_evaluation'`

- [ ] **Step 8: Commit test file**

```bash
git add tests/models/test_test_evaluation.py
git commit -m "test: add failing tests for shared test evaluation module"
```

---

### Task 2: Implement shared test evaluation module

**Files:**
- Create: `src/turbofan/models/test_evaluation.py`

- [ ] **Step 1: Implement the full module**

```python
"""Shared official test set evaluation for sequence models."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from turbofan.config.schema import DataConfig
from turbofan.data.loader import load_raw_test, load_rul_labels
from turbofan.models.evaluate import align_official_test_labels
from turbofan.models.gru import GRURULRegressor
from turbofan.models.metrics import regression_metrics
from turbofan.models.sequence_training import predict_windows
from turbofan.sequences.dataset import build_sequence_loader
from turbofan.sequences.normalize import SequenceNormalizer
from turbofan.sequences.windowing import build_final_windows


def align_labels_to_eligible_engines(
    metadata: pd.DataFrame,
    rul_labels: pd.Series,
) -> pd.Series:
    """Align official RUL labels to eligible sequence test engines.

    C-MAPSS official RUL labels are ordered by engine ID, while final
    sequence windows can skip engines shorter than the window size. This
    selects labels for the eligible engine IDs before applying the
    standard count check.

    Args:
        metadata: Final-window metadata containing eligible ``engine_id``
            rows.
        rul_labels: Official RUL labels in full test engine order.

    Returns:
        Float RUL Series aligned to ``metadata``.

    Raises:
        ValueError: If an eligible engine ID cannot be mapped to a label
            row.
    """
    engine_ids = metadata["engine_id"].to_numpy(dtype=np.int64)
    label_positions = engine_ids - 1
    if np.any(label_positions < 0) or np.any(label_positions >= len(rul_labels)):
        raise ValueError(
            "Official RUL labels must include a row for every eligible test engine."
        )

    eligible_labels = rul_labels.iloc[label_positions].reset_index(drop=True)
    return align_official_test_labels(
        metadata.reset_index(drop=True), eligible_labels
    )


def evaluate_test_from_df(
    test_df: pd.DataFrame,
    rul_labels: pd.Series,
    model: GRURULRegressor,
    normalizer: SequenceNormalizer,
    feature_cols: list[str],
    device: torch.device,
    window_size: int,
    batch_size: int,
    max_rul: int,
) -> dict[str, float]:
    """Evaluate a trained model on pre-loaded test data.

    Normalizes the test DataFrame, builds final windows, predicts,
    aligns labels, and computes regression metrics. Use this when
    the test DataFrame has already been loaded and optionally
    feature-engineered (e.g. rolling features applied).

    Args:
        test_df: Raw or feature-engineered test DataFrame.
        rul_labels: Official RUL labels in full test engine order.
        model: Trained GRU model.
        normalizer: Fitted normalizer (trained on training data).
        feature_cols: Feature columns matching the model's input.
        device: Torch device for inference.
        window_size: Sequence window size.
        batch_size: Inference batch size.
        max_rul: Maximum RUL cap for prediction rescaling.

    Returns:
        Dict with ``test_rmse``, ``test_mae``, ``test_phm08_score``.
    """
    test_normalized = normalizer.transform(test_df)
    test_windows = build_final_windows(
        test_normalized,
        feature_cols=feature_cols,
        window_size=window_size,
        target_col=None,
    )
    loader = build_sequence_loader(
        test_windows, batch_size=batch_size, shuffle=False
    )
    y_pred = np.clip(
        predict_windows(model, loader, device, max_rul=max_rul), 0.0, None
    )
    y_true = align_labels_to_eligible_engines(test_windows.metadata, rul_labels)
    metrics = regression_metrics(y_true, y_pred)
    return {
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_phm08_score": metrics["phm08_score"],
    }


def evaluate_official_test(
    data_config: DataConfig,
    model: GRURULRegressor,
    normalizer: SequenceNormalizer,
    feature_cols: list[str],
    device: torch.device,
    window_size: int,
    batch_size: int,
) -> dict[str, float] | None:
    """Evaluate a trained model on official C-MAPSS test files.

    Loads the test data and RUL labels from paths derived from
    ``data_config``, then delegates to :func:`evaluate_test_from_df`.

    Args:
        data_config: Data config for file paths and ``max_rul``.
        model: Trained GRU model.
        normalizer: Fitted normalizer (trained on training data).
        feature_cols: Feature columns matching the model's input.
        device: Torch device for inference.
        window_size: Sequence window size.
        batch_size: Inference batch size.

    Returns:
        Dict with ``test_rmse``, ``test_mae``, ``test_phm08_score``,
        or ``None`` when test files are missing.
    """
    try:
        test_raw = load_raw_test(data_config)
        rul_labels = load_rul_labels(data_config)
    except FileNotFoundError:
        return None

    return evaluate_test_from_df(
        test_df=test_raw,
        rul_labels=rul_labels,
        model=model,
        normalizer=normalizer,
        feature_cols=feature_cols,
        device=device,
        window_size=window_size,
        batch_size=batch_size,
        max_rul=data_config.max_rul,
    )
```

- [ ] **Step 2: Run tests to confirm they pass**

Run: `pytest tests/models/test_test_evaluation.py -v`
Expected: All 5 tests PASS

- [ ] **Step 3: Run lint and type checks**

Run: `ruff check src/turbofan/models/test_evaluation.py tests/models/test_test_evaluation.py`
Run: `mypy src/turbofan/models/test_evaluation.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add src/turbofan/models/test_evaluation.py
git commit -m "feat: add shared test evaluation module"
```

---

### Task 3: Refactor `train_sequence_gru.py` to use shared module

The training script keeps its `_evaluate_official_test` private function (it returns prediction DataFrames, which is unique to artifact saving). The refactor only extracts the label alignment logic by deleting `_align_official_labels_to_eligible_engines` and importing the shared version.

**Files:**
- Modify: `scripts/train_sequence_gru.py`

- [ ] **Step 1: Update imports**

Remove `align_official_test_labels` from the `turbofan.models.evaluate` import (line 16):
```python
# Before:
from turbofan.models.evaluate import add_rul_column, align_official_test_labels

# After:
from turbofan.models.evaluate import add_rul_column
```

Add the shared module import after the evaluate import:
```python
from turbofan.models.test_evaluation import align_labels_to_eligible_engines
```

- [ ] **Step 2: Delete `_align_official_labels_to_eligible_engines`**

Remove the entire function at lines 137-166.

- [ ] **Step 3: Update `_evaluate_official_test` to use the shared function**

Change the call inside `_evaluate_official_test` (line 209):
```python
# Before:
    y_true = _align_official_labels_to_eligible_engines(
        test_windows.metadata,
        rul_labels,
    )

# After:
    y_true = align_labels_to_eligible_engines(
        test_windows.metadata,
        rul_labels,
    )
```

- [ ] **Step 4: Run all existing tests to confirm nothing broke**

Run: `pytest tests/models/test_train_sequence_gru_cli.py -v`
Expected: All tests PASS

Run: `pytest tests/models/test_test_evaluation.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run lint and type checks**

Run: `ruff check scripts/train_sequence_gru.py`
Run: `mypy src/turbofan`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add scripts/train_sequence_gru.py
git commit -m "refactor: use shared align_labels_to_eligible_engines in train script"
```

---

### Task 4: Add test evaluation to `sweep_sequence_gru.py`

**Files:**
- Modify: `scripts/sweep_sequence_gru.py`
- Modify: `tests/models/test_sweep_sequence_gru.py`

- [ ] **Step 1: Write failing test — result columns include test metrics**

Add to `tests/models/test_sweep_sequence_gru.py`. Find the test `test_gru_sweep_returns_expected_rows` and note it asserts `list(results.columns)`. We need to update its column expectation and add a new test for test metrics. But first, write a new targeted test:

```python
def test_gru_sweep_includes_test_metric_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GRU sweep result includes test_rmse, test_mae, test_phm08_score columns."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)
    monkeypatch.setattr(module, "append_training_log", lambda entry: None)

    results = module.run_gru_sweep(
        config_path=cfg_path,
        window_sizes=[3],
        hidden_sizes=[2],
        learning_rates=[1e-3],
        device="cpu",
    )

    assert "test_rmse" in results.columns
    assert "test_mae" in results.columns
    assert "test_phm08_score" in results.columns
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `pytest tests/models/test_sweep_sequence_gru.py::test_gru_sweep_includes_test_metric_columns -v`
Expected: FAIL — columns don't include test metrics yet

- [ ] **Step 3: Update `_write_config` to include test files**

The existing `_write_config` only creates `train_FD001.txt`. For test evaluation to work, it needs `test_FD001.txt` and `RUL_FD001.txt`. Update `_write_config` in the test file:

```python
def _write_config(tmp_path: Path) -> Path:
    """Write a minimal project config for GRU sweep tests.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Created config path.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)
    _write_cmapps_file(raw_dir / "test_FD001.txt", n_engines=4, n_cycles=6)
    (raw_dir / "RUL_FD001.txt").write_text("10\n20\n30\n40\n")
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
```

- [ ] **Step 4: Implement test evaluation in `sweep_sequence_gru.py`**

Add to imports:
```python
from turbofan.models.test_evaluation import evaluate_official_test
```

Update `RESULT_COLUMNS`:
```python
RESULT_COLUMNS = [
    "window_size",
    "hidden_size",
    "learning_rate",
    "best_epoch",
    "rmse",
    "mae",
    "phm08_score",
    "test_rmse",
    "test_mae",
    "test_phm08_score",
]
```

In `run_gru_sweep`, after loading config and resolving device (around line 184), add test data loading attempt:

```python
    test_metrics_available = True
    try:
        from turbofan.data.loader import load_raw_test, load_rul_labels
        test_raw = load_raw_test(cfg.data)
        rul_labels = load_rul_labels(cfg.data)
    except FileNotFoundError:
        test_metrics_available = False
        test_raw = None
        rul_labels = None
```

Wait — this is getting complicated. The cleaner approach: use `evaluate_official_test` per run (it handles `FileNotFoundError` internally and returns `None`). The small cost is reloading test files per run, but correctness is more important than optimization in a sweep script.

Inside the per-run loop, after the validation metrics block (after line 283), add:

```python
        test_result = evaluate_official_test(
            data_config=cfg.data,
            model=result.model,
            normalizer=normalizer,
            feature_cols=feature_cols,
            device=torch_device,
            window_size=window_size,
            batch_size=spec_cfg.batch_size,
        )
        if test_result is not None:
            row["test_rmse"] = test_result["test_rmse"]
            row["test_mae"] = test_result["test_mae"]
            row["test_phm08_score"] = test_result["test_phm08_score"]
        else:
            row["test_rmse"] = float("nan")
            row["test_mae"] = float("nan")
            row["test_phm08_score"] = float("nan")
```

Update the log entry to include test metrics in `extra`. The current `build_log_entry` call has no `extra` parameter. Add it:

```python
        log_entry = build_log_entry(
            model_type="gru",
            dataset=cfg.data.fd_subset,
            random_seed=cfg.data.random_seed,
            hyperparameters={
                "window_size": window_size,
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
                "num_layers": spec_cfg.num_layers,
                "dropout": spec_cfg.dropout,
                "batch_size": spec_cfg.batch_size,
                "epochs": spec_cfg.epochs,
                "patience": spec_cfg.patience,
            },
            metrics=metrics,
            training_duration_seconds=training_duration_seconds,
            device=_device_name(torch_device),
            run_dir=None,
            best_epoch=result.best_epoch,
            extra=test_result if test_result is not None else {},
        )
```

Update the print statement to include test score:

```python
        test_score_str = (
            f" test_phm08={test_result['test_phm08_score']:.6f}"
            if test_result is not None
            else " test=N/A"
        )
        print(
            f"run {run_idx}/{total_runs}: "
            f"window_size={window_size} hidden_size={hidden_size} "
            f"learning_rate={learning_rate:g} "
            f"phm08_score={metrics['phm08_score']:.6f}"
            f"{test_score_str}"
        )
```

- [ ] **Step 5: Update existing test column assertions**

In `test_gru_sweep_returns_expected_rows`, update the expected columns:

```python
    assert list(results.columns) == [
        "window_size",
        "hidden_size",
        "learning_rate",
        "best_epoch",
        "rmse",
        "mae",
        "phm08_score",
        "test_rmse",
        "test_mae",
        "test_phm08_score",
    ]
```

- [ ] **Step 6: Update the monkeypatch-based tests**

The tests `test_gru_sweep_reports_validation_window_metrics` and `test_gru_sweep_appends_training_log_entry_per_completed_config` use monkeypatching and never create test files, so `evaluate_official_test` will return `None`. We need to monkeypatch the new import. Add to both tests:

```python
    monkeypatch.setattr(module, "evaluate_official_test", lambda **kwargs: None)
```

For the training log test, update the expected `build_calls` to include the new `extra` parameter. Each entry should now have `"extra": {}`.

- [ ] **Step 7: Run all sweep sequence tests**

Run: `pytest tests/models/test_sweep_sequence_gru.py -v`
Expected: All tests PASS

- [ ] **Step 8: Run lint and type checks**

Run: `ruff check scripts/sweep_sequence_gru.py tests/models/test_sweep_sequence_gru.py`
Run: `mypy src/turbofan`
Expected: No errors

- [ ] **Step 9: Commit**

```bash
git add scripts/sweep_sequence_gru.py tests/models/test_sweep_sequence_gru.py
git commit -m "feat: add test evaluation to sequence GRU sweep"
```

---

### Task 5: Add test evaluation to `sweep_feature_gru.py`

**Files:**
- Modify: `scripts/sweep_feature_gru.py`
- Modify: `tests/models/test_sweep_feature_gru.py`

- [ ] **Step 1: Write failing test — result columns include test metrics**

Add to `tests/models/test_sweep_feature_gru.py`:

```python
def test_feature_sweep_includes_test_metric_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Feature sweep result includes test_rmse, test_mae, test_phm08_score."""
    project_root = Path(__file__).parent.parent.parent
    module = _load_module(project_root)
    cfg_path = _write_config(tmp_path)
    monkeypatch.setattr(module, "append_training_log", lambda entry: None)

    results = module.run_feature_sweep(
        config_path=cfg_path,
        feature_sets=["raw"],
        corr_thresholds=[0.5],
        rolling_window=3,
        device="cpu",
    )

    assert "test_rmse" in results.columns
    assert "test_mae" in results.columns
    assert "test_phm08_score" in results.columns
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `pytest tests/models/test_sweep_feature_gru.py::test_feature_sweep_includes_test_metric_columns -v`
Expected: FAIL — columns don't include test metrics yet

- [ ] **Step 3: Update `_write_config` in the test file to include test files**

Same pattern as Task 4 — update `_write_config` in `tests/models/test_sweep_feature_gru.py`:

```python
def _write_config(tmp_path: Path) -> Path:
    """Write a minimal project config for feature GRU sweep tests.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Created config path.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)
    _write_cmapps_file(raw_dir / "test_FD001.txt", n_engines=4, n_cycles=6)
    (raw_dir / "RUL_FD001.txt").write_text("10\n20\n30\n40\n")
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
```

- [ ] **Step 4: Implement test evaluation in `sweep_feature_gru.py`**

This is the more complex case because the feature engineering (rolling features, correlation filtering) must also be applied to test data.

Add to imports:
```python
from turbofan.data.loader import load_raw_test, load_rul_labels
from turbofan.models.test_evaluation import evaluate_test_from_df
```

Update `RESULT_COLUMNS`:
```python
RESULT_COLUMNS = [
    "feature_set",
    "corr_threshold",
    "n_features",
    "best_epoch",
    "rmse",
    "mae",
    "phm08_score",
    "test_rmse",
    "test_mae",
    "test_phm08_score",
]
```

In `run_feature_sweep`, before the run loop (after loading train data and building the grid, around line 249), load test data once:

```python
    try:
        test_raw = load_raw_test(cfg.data)
        rul_labels = load_rul_labels(cfg.data)
    except FileNotFoundError:
        test_raw = None
        rul_labels = None
```

Inside the per-run loop, after the validation metrics block (after line 359), add test evaluation. The test data needs the same feature engineering as the run:

```python
        if test_raw is not None and rul_labels is not None:
            current_test_df = test_raw
            if use_rolling:
                current_test_df = extractor.transform(test_raw)
            test_result = evaluate_test_from_df(
                test_df=current_test_df,
                rul_labels=rul_labels,
                model=result.model,
                normalizer=normalizer,
                feature_cols=feature_cols,
                device=torch_device,
                window_size=window_size,
                batch_size=cfg.sequence.batch_size,
                max_rul=cfg.data.max_rul,
            )
            row["test_rmse"] = test_result["test_rmse"]
            row["test_mae"] = test_result["test_mae"]
            row["test_phm08_score"] = test_result["test_phm08_score"]
        else:
            test_result = None
            row["test_rmse"] = float("nan")
            row["test_mae"] = float("nan")
            row["test_phm08_score"] = float("nan")
```

Update the training log `extra` dict to include test metrics. The current `extra` dict has feature set info. Merge test metrics into it:

```python
        extra_dict: dict[str, object] = {
            "feature_set": feature_set,
            "corr_threshold": threshold_val,
            "n_features": len(feature_cols),
            "rolling_window": rolling_window,
        }
        if test_result is not None:
            extra_dict.update(test_result)

        log_entry = build_log_entry(
            ...
            extra=extra_dict,
        )
```

Update the print statement:

```python
        test_score_str = (
            f" test_phm08={test_result['test_phm08_score']:.6f}"
            if test_result is not None
            else " test=N/A"
        )
        print(
            f"run {run_idx}/{total_runs}: "
            f"feature_set={feature_set} "
            f"corr_threshold={corr_threshold} "
            f"n_features={len(feature_cols)} "
            f"phm08_score={metrics['phm08_score']:.6f}"
            f"{test_score_str}"
        )
```

- [ ] **Step 5: Update existing test column assertions**

In `test_feature_sweep_returns_expected_columns`, update the expected columns:

```python
    assert list(results.columns) == [
        "feature_set",
        "corr_threshold",
        "n_features",
        "best_epoch",
        "rmse",
        "mae",
        "phm08_score",
        "test_rmse",
        "test_mae",
        "test_phm08_score",
    ]
```

Also update `test_feature_sweep_cli_writes_csv` to expect the new columns:

```python
    assert list(results.columns) == [
        "feature_set",
        "corr_threshold",
        "n_features",
        "best_epoch",
        "rmse",
        "mae",
        "phm08_score",
        "test_rmse",
        "test_mae",
        "test_phm08_score",
    ]
```

- [ ] **Step 6: Run all feature sweep tests**

Run: `pytest tests/models/test_sweep_feature_gru.py -v`
Expected: All tests PASS

- [ ] **Step 7: Run lint and type checks**

Run: `ruff check scripts/sweep_feature_gru.py tests/models/test_sweep_feature_gru.py`
Run: `mypy src/turbofan`
Expected: No errors

- [ ] **Step 8: Commit**

```bash
git add scripts/sweep_feature_gru.py tests/models/test_sweep_feature_gru.py
git commit -m "feat: add test evaluation to feature GRU sweep"
```

---

### Task 6: Final integration verification

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS

- [ ] **Step 2: Run full lint and type check**

Run: `ruff check src/ tests/ scripts/`
Run: `mypy src/turbofan`
Expected: No errors

- [ ] **Step 3: Commit if any fixups were needed**

Only if previous steps required changes:
```bash
git add -A
git commit -m "fix: lint and type check fixups for test evaluation sweep"
```
