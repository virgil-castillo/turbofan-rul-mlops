# Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the installable `turbofan` Python package with a Pydantic v2 config system, stdlib logging, a raw C-MAPSS data loader with piecewise-linear RUL label computation, and a NASA data download script.

**Architecture:** The `src/turbofan/` package uses three domain subpackages — `config/`, `utils/`, `data/` — each with a single clear responsibility. Config is loaded once from YAML and passed explicitly to all functions. The data layer separates I/O (`loader.py`) from pure computation (`labels.py`) so label logic can be unit-tested without touching the filesystem.

**Tech Stack:** Python 3.12, pandas, numpy, pydantic v2, pyyaml, pytest, ruff, mypy, pandas-stubs

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Package metadata, tool config (ruff, mypy, pytest) |
| `configs/default.yaml` | Default project configuration values |
| `src/turbofan/__init__.py` | Package version |
| `src/turbofan/config/__init__.py` | Subpackage marker |
| `src/turbofan/config/schema.py` | `DataConfig`, `ProjectConfig`, `load_config()` |
| `src/turbofan/utils/__init__.py` | Subpackage marker |
| `src/turbofan/utils/logging.py` | `setup_logging()`, `get_logger()` |
| `src/turbofan/data/__init__.py` | Subpackage marker |
| `src/turbofan/data/labels.py` | `compute_rul_labels()` — pure RUL math |
| `src/turbofan/data/loader.py` | `load_raw_train()`, `load_raw_test()`, `load_rul_labels()` |
| `scripts/download_data.py` | `--kaggle` download, `--check` validation |
| `tests/conftest.py` | `sample_train_df`, `tmp_data_dir` fixtures |
| `tests/config/test_schema.py` | Config schema unit tests |
| `tests/utils/test_logging.py` | Logging utility tests |
| `tests/data/test_labels.py` | RUL label computation unit tests |
| `tests/data/test_loader.py` | Data loader I/O tests |

---

## Task 1: pyproject.toml and package scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/turbofan/__init__.py`
- Create: `src/turbofan/config/__init__.py`
- Create: `src/turbofan/utils/__init__.py`
- Create: `src/turbofan/data/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "turbofan"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "numpy",
    "pandas>=2.0",
    "pydantic>=2.0",
    "pyyaml",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "ruff",
    "mypy",
    "types-pyyaml",
    "pandas-stubs",
    "kaggle",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "ANN"]

[tool.mypy]
strict = true
python_version = "3.12"
mypy_path = "src"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `src/turbofan/__init__.py`**

```python
"""Turbofan engine Remaining Useful Life prediction package."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create the three subpackage `__init__.py` files**

`src/turbofan/config/__init__.py`:
```python
"""Configuration schema and loader for the turbofan package."""
```

`src/turbofan/utils/__init__.py`:
```python
"""Utility helpers for the turbofan package."""
```

`src/turbofan/data/__init__.py`:
```python
"""Data loading and label computation for C-MAPSS turbofan dataset."""
```

- [ ] **Step 4: Install the package in editable mode**

Run:
```bash
pip install -e ".[dev]"
```

Expected: output ending with `Successfully installed turbofan-0.1.0`

- [ ] **Step 5: Verify the import works**

Run:
```bash
python -c "import turbofan; print(turbofan.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/
git commit -m "feat: add installable turbofan package scaffold"
```

---

## Task 2: Directory skeleton and .gitignore update

**Files:**
- Modify: `.gitignore`
- Create: `data/raw/.gitkeep`, `data/interim/.gitkeep`, `data/processed/.gitkeep`
- Create: `outputs/models/.gitkeep`, `outputs/figures/.gitkeep`, `outputs/logs/.gitkeep`
- Create: `configs/` (empty dir for now), `scripts/` (empty dir for now)

- [ ] **Step 1: Extend `.gitignore`**

Append to the existing `.gitignore`:
```
# Data directories — populated by download script
data/raw/*
data/interim/*
data/processed/*
!data/raw/.gitkeep
!data/interim/.gitkeep
!data/processed/.gitkeep

# Outputs
outputs/*
!outputs/models/.gitkeep
!outputs/figures/.gitkeep
!outputs/logs/.gitkeep

# MLflow
mlruns/
mlartifacts/
```

- [ ] **Step 2: Create tracked placeholder files**

Create empty `.gitkeep` files at:
- `data/raw/.gitkeep`
- `data/interim/.gitkeep`
- `data/processed/.gitkeep`
- `outputs/models/.gitkeep`
- `outputs/figures/.gitkeep`
- `outputs/logs/.gitkeep`

Each file is empty. These allow git to track the directory structure while the actual data and output files remain git-ignored.

- [ ] **Step 3: Create empty `configs/` and `scripts/` directories**

Create `configs/.gitkeep` and `scripts/.gitkeep` (both empty). These will be replaced with real files in later tasks and can be deleted once those files exist.

- [ ] **Step 4: Commit**

```bash
git add .gitignore data/ outputs/ configs/ scripts/
git commit -m "chore: add directory skeleton and gitignore rules for data and outputs"
```

---

## Task 3: Config schema (TDD)

**Files:**
- Create: `tests/config/test_schema.py`
- Create: `src/turbofan/config/schema.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/config/test_schema.py`:

```python
"""Tests for turbofan.config.schema."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from turbofan.config.schema import DataConfig, ProjectConfig, load_config


def _write_config(tmp_path: Path, data: dict) -> Path:  # type: ignore[type-arg]
    """Write a config dict to a YAML file and return the path."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data))
    return path


def test_load_config_valid(tmp_path: Path) -> None:
    """Valid YAML round-trips through load_config without error."""
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
    assert cfg.project_name == "test-project"
    assert cfg.data.fd_subset == "FD001"
    assert cfg.data.max_rul == 125
    assert cfg.data.random_seed == 42


def test_path_fields_are_path_objects(tmp_path: Path) -> None:
    """DataConfig path fields are Path objects, not strings."""
    cfg = DataConfig(
        raw_dir="data/raw",  # type: ignore[arg-type]
        processed_dir="data/processed",  # type: ignore[arg-type]
        interim_dir="data/interim",  # type: ignore[arg-type]
    )
    assert isinstance(cfg.raw_dir, Path)
    assert isinstance(cfg.processed_dir, Path)
    assert isinstance(cfg.interim_dir, Path)


def test_invalid_fd_subset_raises(tmp_path: Path) -> None:
    """Invalid fd_subset value raises ValidationError."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
                "fd_subset": "FD999",
            },
        },
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_missing_project_name_raises(tmp_path: Path) -> None:
    """Missing required project_name raises ValidationError."""
    cfg_file = _write_config(
        tmp_path,
        {
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
        },
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_missing_data_section_raises(tmp_path: Path) -> None:
    """Missing required data section raises ValidationError."""
    cfg_file = _write_config(tmp_path, {"project_name": "test"})
    with pytest.raises(ValidationError):
        load_config(cfg_file)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/config/test_schema.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'turbofan.config.schema'`

- [ ] **Step 3: Implement `src/turbofan/config/schema.py`**

```python
"""Configuration schema for the turbofan package."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


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
    max_rul: int = 125
    test_size: float = 0.2
    random_seed: int = 42


class ProjectConfig(BaseModel):
    """Top-level project configuration.

    Args:
        project_name: Human-readable project name.
        data: Data layer configuration.
    """

    project_name: str
    data: DataConfig


def load_config(path: Path) -> ProjectConfig:
    """Load and validate project configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated ProjectConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the config is invalid.
    """
    raw = yaml.safe_load(path.read_text())
    return ProjectConfig.model_validate(raw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/config/test_schema.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/config/test_schema.py src/turbofan/config/schema.py
git commit -m "feat: add Pydantic v2 config schema with YAML loader"
```

---

## Task 4: Logging utilities (TDD)

**Files:**
- Create: `tests/utils/test_logging.py`
- Create: `src/turbofan/utils/logging.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/utils/test_logging.py`:

```python
"""Tests for turbofan.utils.logging."""
from __future__ import annotations

import logging

from turbofan.utils.logging import get_logger, setup_logging


def test_get_logger_returns_logger() -> None:
    """get_logger returns a logging.Logger with the given name."""
    logger = get_logger("turbofan.test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "turbofan.test_module"


def test_get_logger_different_names_are_distinct() -> None:
    """Two loggers with different names are different objects."""
    logger_a = get_logger("turbofan.module_a")
    logger_b = get_logger("turbofan.module_b")
    assert logger_a is not logger_b


def test_setup_logging_does_not_raise() -> None:
    """setup_logging completes without raising for valid level strings."""
    setup_logging("DEBUG")
    setup_logging("INFO")
    setup_logging("WARNING")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/utils/test_logging.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'turbofan.utils.logging'`

- [ ] **Step 3: Implement `src/turbofan/utils/logging.py`**

```python
"""Logging configuration for the turbofan package."""
from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with timestamped console output.

    Args:
        level: Logging level string (e.g., "INFO", "DEBUG", "WARNING").
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a module.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)
```

Note: `force=True` in `basicConfig` allows reconfiguration in tests. Without it, subsequent `basicConfig` calls are no-ops once handlers are attached.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/utils/test_logging.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/utils/test_logging.py src/turbofan/utils/logging.py
git commit -m "feat: add stdlib logging helpers setup_logging and get_logger"
```

---

## Task 5: Default config YAML

**Files:**
- Create: `configs/default.yaml`
- Modify: `tests/config/test_schema.py` (add one integration test)

- [ ] **Step 1: Create `configs/default.yaml`**

```yaml
project_name: turbofan-rul-mlops

data:
  raw_dir: data/raw
  processed_dir: data/processed
  interim_dir: data/interim
  fd_subset: FD001
  max_rul: 125
  test_size: 0.2
  random_seed: 42
```

- [ ] **Step 2: Add an integration test that loads the real default config**

Append to `tests/config/test_schema.py`:

```python
def test_load_default_config() -> None:
    """The committed default.yaml loads and validates without error."""
    project_root = Path(__file__).parent.parent.parent
    cfg_path = project_root / "configs" / "default.yaml"
    cfg = load_config(cfg_path)
    assert cfg.project_name == "turbofan-rul-mlops"
    assert cfg.data.fd_subset == "FD001"
    assert cfg.data.max_rul == 125
```

- [ ] **Step 3: Run the new test**

Run:
```bash
pytest tests/config/test_schema.py::test_load_default_config -v
```

Expected: `1 passed`

- [ ] **Step 4: Commit**

```bash
git add configs/default.yaml tests/config/test_schema.py
git commit -m "feat: add default.yaml project config and integration test"
```

---

## Task 6: RUL label computation (TDD)

**Files:**
- Create: `tests/data/test_labels.py`
- Create: `src/turbofan/data/labels.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_labels.py`:

```python
"""Tests for turbofan.data.labels."""
from __future__ import annotations

import pandas as pd

from turbofan.data.labels import compute_rul_labels


def _make_df(engine_cycles: dict[int, int]) -> pd.DataFrame:
    """Build a minimal DataFrame from {engine_id: n_cycles}."""
    rows = []
    for engine_id, n_cycles in engine_cycles.items():
        for cycle in range(1, n_cycles + 1):
            rows.append({"engine_id": engine_id, "cycle": cycle})
    return pd.DataFrame(rows)


def test_rul_is_zero_at_last_cycle() -> None:
    """RUL is 0 at the final cycle of each engine."""
    df = _make_df({1: 10})
    rul = compute_rul_labels(df, max_rul=125)
    last_cycle_rul = rul[df["cycle"] == 10].iloc[0]
    assert last_cycle_rul == 0


def test_rul_capped_at_max_rul() -> None:
    """RUL never exceeds max_rul for any cycle."""
    df = _make_df({1: 200})
    rul = compute_rul_labels(df, max_rul=125)
    assert (rul <= 125).all()


def test_rul_equals_max_rul_for_early_cycles() -> None:
    """Early cycles where cycles_remaining > max_rul receive max_rul."""
    df = _make_df({1: 200})
    rul = compute_rul_labels(df, max_rul=125)
    first_cycle_rul = rul[df["cycle"] == 1].iloc[0]
    assert first_cycle_rul == 125


def test_rul_custom_max_rul() -> None:
    """Custom max_rul value is respected."""
    df = _make_df({1: 50})
    rul = compute_rul_labels(df, max_rul=30)
    first_cycle_rul = rul[df["cycle"] == 1].iloc[0]
    assert first_cycle_rul == 30
    last_cycle_rul = rul[df["cycle"] == 50].iloc[0]
    assert last_cycle_rul == 0


def test_rul_multiple_engines_independent() -> None:
    """Engines with different lifespans each reach RUL=0 at their own last cycle."""
    df = _make_df({1: 10, 2: 20})
    rul = compute_rul_labels(df, max_rul=125)
    engine1_last = rul[(df["engine_id"] == 1) & (df["cycle"] == 10)].iloc[0]
    engine2_last = rul[(df["engine_id"] == 2) & (df["cycle"] == 20)].iloc[0]
    assert engine1_last == 0
    assert engine2_last == 0


def test_rul_index_aligned_to_input() -> None:
    """Returned Series has the same index as the input DataFrame."""
    df = _make_df({1: 5})
    rul = compute_rul_labels(df, max_rul=125)
    assert list(rul.index) == list(df.index)


def test_rul_series_name() -> None:
    """Returned Series is named 'rul'."""
    df = _make_df({1: 5})
    rul = compute_rul_labels(df, max_rul=125)
    assert rul.name == "rul"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/data/test_labels.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'turbofan.data.labels'`

- [ ] **Step 3: Implement `src/turbofan/data/labels.py`**

```python
"""RUL label computation for C-MAPSS turbofan training data."""
from __future__ import annotations

import pandas as pd


def compute_rul_labels(df: pd.DataFrame, max_rul: int = 125) -> pd.Series:  # type: ignore[type-arg]
    """Compute piecewise-linear RUL labels for training data.

    For each engine: RUL = min(max_rul, max_cycle - current_cycle).
    Returns a Series aligned to df.index.

    Args:
        df: DataFrame with columns ``engine_id`` and ``cycle``.
        max_rul: Maximum RUL cap. Default 125 matches FD001 literature.

    Returns:
        pd.Series of integer RUL values with the same index as ``df``,
        named ``"rul"``.
    """
    max_cycles: pd.Series = df.groupby("engine_id")["cycle"].transform("max")  # type: ignore[assignment]
    rul: pd.Series = (max_cycles - df["cycle"]).clip(upper=max_rul)  # type: ignore[assignment]
    return rul.rename("rul")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/data/test_labels.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_labels.py src/turbofan/data/labels.py
git commit -m "feat: add piecewise-linear RUL label computation"
```

---

## Task 7: Data loader (TDD)

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/data/test_loader.py`
- Create: `src/turbofan/data/loader.py`

- [ ] **Step 1: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures for the turbofan test suite."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from turbofan.config.schema import DataConfig

SENSOR_COLS = {f"s_{i}": float(i) for i in range(1, 22)}


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """Minimal valid train DataFrame: 3 engines with varied cycle lengths.

    Engine 1: 10 cycles, Engine 2: 7 cycles, Engine 3: 15 cycles.
    """
    rows = []
    for engine_id, n_cycles in [(1, 10), (2, 7), (3, 15)]:
        for cycle in range(1, n_cycles + 1):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "op_1": 0.0,
                    "op_2": 0.0,
                    "op_3": 0.0,
                    **SENSOR_COLS,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_data_dir(tmp_path: Path, sample_train_df: pd.DataFrame) -> Path:
    """Temp directory with correctly-named stub C-MAPSS .txt files for FD001."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    train_path = raw_dir / "train_FD001.txt"
    sample_train_df.to_csv(train_path, sep=" ", header=False, index=False)

    test_path = raw_dir / "test_FD001.txt"
    sample_train_df.to_csv(test_path, sep=" ", header=False, index=False)

    rul_path = raw_dir / "RUL_FD001.txt"
    rul_path.write_text("112\n98\n42\n")

    return raw_dir


@pytest.fixture
def data_cfg(tmp_data_dir: Path) -> DataConfig:
    """DataConfig pointing at the tmp_data_dir stub files."""
    return DataConfig(
        raw_dir=tmp_data_dir,
        processed_dir=tmp_data_dir / "processed",
        interim_dir=tmp_data_dir / "interim",
        fd_subset="FD001",
    )
```

- [ ] **Step 2: Write the failing tests**

Create `tests/data/test_loader.py`:

```python
"""Tests for turbofan.data.loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from turbofan.config.schema import DataConfig
from turbofan.data.loader import COLUMN_NAMES, load_raw_test, load_raw_train, load_rul_labels


def test_load_raw_train_column_names(data_cfg: DataConfig) -> None:
    """load_raw_train assigns the correct 26 column names."""
    df = load_raw_train(data_cfg)
    assert list(df.columns) == COLUMN_NAMES


def test_load_raw_train_returns_nonempty_dataframe(data_cfg: DataConfig) -> None:
    """load_raw_train returns a non-empty DataFrame."""
    df = load_raw_train(data_cfg)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_raw_train_engine_id_column(data_cfg: DataConfig) -> None:
    """engine_id column contains the expected engine IDs."""
    df = load_raw_train(data_cfg)
    assert set(df["engine_id"].unique()) == {1, 2, 3}


def test_load_raw_test_column_names(data_cfg: DataConfig) -> None:
    """load_raw_test assigns the correct 26 column names."""
    df = load_raw_test(data_cfg)
    assert list(df.columns) == COLUMN_NAMES


def test_load_raw_train_missing_file_raises(tmp_path: Path) -> None:
    """load_raw_train raises FileNotFoundError with download hint when file is missing."""
    cfg = DataConfig(
        raw_dir=tmp_path / "nonexistent",
        processed_dir=tmp_path,
        interim_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_raw_train(cfg)


def test_load_raw_test_missing_file_raises(tmp_path: Path) -> None:
    """load_raw_test raises FileNotFoundError with download hint when file is missing."""
    cfg = DataConfig(
        raw_dir=tmp_path / "nonexistent",
        processed_dir=tmp_path,
        interim_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_raw_test(cfg)


def test_load_rul_labels_returns_series(data_cfg: DataConfig) -> None:
    """load_rul_labels returns a pd.Series."""
    rul = load_rul_labels(data_cfg)
    assert isinstance(rul, pd.Series)


def test_load_rul_labels_length(data_cfg: DataConfig) -> None:
    """load_rul_labels returns one value per test engine (3 in stub file)."""
    rul = load_rul_labels(data_cfg)
    assert len(rul) == 3


def test_load_rul_labels_missing_file_raises(tmp_path: Path) -> None:
    """load_rul_labels raises FileNotFoundError with download hint when file is missing."""
    cfg = DataConfig(
        raw_dir=tmp_path / "nonexistent",
        processed_dir=tmp_path,
        interim_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_rul_labels(cfg)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:
```bash
pytest tests/data/test_loader.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'turbofan.data.loader'`

- [ ] **Step 4: Implement `src/turbofan/data/loader.py`**

```python
"""Raw data loaders for the NASA C-MAPSS turbofan dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from turbofan.config.schema import DataConfig

COLUMN_NAMES: list[str] = [
    "engine_id",
    "cycle",
    "op_1",
    "op_2",
    "op_3",
    *[f"s_{i}" for i in range(1, 22)],
]

_DOWNLOAD_HINT: str = (
    "Run `python scripts/download_data.py --kaggle` to download the dataset, "
    "or `python scripts/download_data.py --check` to verify files are present."
)


def _load_txt(path: Path) -> pd.DataFrame:
    """Read a space-delimited C-MAPSS file and assign column names.

    Args:
        path: Path to the .txt file.

    Returns:
        DataFrame with COLUMN_NAMES columns.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}\n{_DOWNLOAD_HINT}"
        )
    df: pd.DataFrame = pd.read_csv(
        path, sep=r"\s+", header=None, index_col=False
    )
    df = df.iloc[:, : len(COLUMN_NAMES)]
    df.columns = pd.Index(COLUMN_NAMES)
    return df


def load_raw_train(cfg: DataConfig) -> pd.DataFrame:
    """Load raw training data for the configured FD subset.

    Args:
        cfg: DataConfig with ``raw_dir`` and ``fd_subset``.

    Returns:
        DataFrame with columns: engine_id, cycle, op_1–op_3, s_1–s_21.

    Raises:
        FileNotFoundError: If the training file is not found.
    """
    return _load_txt(cfg.raw_dir / f"train_{cfg.fd_subset}.txt")


def load_raw_test(cfg: DataConfig) -> pd.DataFrame:
    """Load raw test data for the configured FD subset.

    Args:
        cfg: DataConfig with ``raw_dir`` and ``fd_subset``.

    Returns:
        DataFrame with columns: engine_id, cycle, op_1–op_3, s_1–s_21.

    Raises:
        FileNotFoundError: If the test file is not found.
    """
    return _load_txt(cfg.raw_dir / f"test_{cfg.fd_subset}.txt")


def load_rul_labels(cfg: DataConfig) -> pd.Series:  # type: ignore[type-arg]
    """Load ground-truth RUL values for test engines.

    Args:
        cfg: DataConfig with ``raw_dir`` and ``fd_subset``.

    Returns:
        pd.Series of integer RUL values, one per test engine, named ``"rul"``.

    Raises:
        FileNotFoundError: If the RUL file is not found.
    """
    path = cfg.raw_dir / f"RUL_{cfg.fd_subset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected RUL file not found: {path}\n{_DOWNLOAD_HINT}"
        )
    return pd.read_csv(path, header=None).iloc[:, 0].rename("rul")
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
pytest tests/data/test_loader.py -v
```

Expected: `9 passed`

- [ ] **Step 6: Run the full test suite**

Run:
```bash
pytest -v
```

Expected: all tests pass (3 + 5 + 7 + 9 + 1 = 25 or more passing)

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py tests/data/test_loader.py src/turbofan/data/loader.py
git commit -m "feat: add C-MAPSS raw data loader with column assignment and error hints"
```

---

## Task 8: Download script

**Files:**
- Create: `scripts/download_data.py`

- [ ] **Step 1: Implement `scripts/download_data.py`**

```python
"""Download or verify the NASA C-MAPSS turbofan dataset.

Usage:
    python scripts/download_data.py --kaggle   # download via Kaggle API
    python scripts/download_data.py --check    # verify files are present
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPECTED_FILES: list[str] = [
    f"{split}_FD00{i}.txt"
    for split in ("train", "test", "RUL")
    for i in range(1, 5)
]

RAW_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "raw"
KAGGLE_DATASET: str = "behrad3d/nasa-cmaps"
MANUAL_URL: str = "https://www.kaggle.com/datasets/behrad3d/nasa-cmaps"


def check(raw_dir: Path = RAW_DIR) -> bool:
    """Verify all expected data files are present in raw_dir.

    Args:
        raw_dir: Directory to check.

    Returns:
        True if all 12 expected files are present, False otherwise.
    """
    all_present = True
    for fname in EXPECTED_FILES:
        path = raw_dir / fname
        if path.exists():
            size_kb = path.stat().st_size // 1024
            print(f"  [OK] {fname} ({size_kb} KB)")
        else:
            print(f"  [MISSING] {fname}")
            all_present = False
    return all_present


def download_kaggle(raw_dir: Path = RAW_DIR) -> None:
    """Download the dataset via the Kaggle API.

    Prints manual instructions and exits with code 1 if the Kaggle API
    key is not configured.

    Args:
        raw_dir: Directory to download files into.
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("Kaggle API key not found at ~/.kaggle/kaggle.json")
        print("\nManual download instructions:")
        print(f"  1. Go to {MANUAL_URL}")
        print("  2. Click 'Download' to get the dataset zip")
        print(f"  3. Extract all .txt files into: {raw_dir}/")
        sys.exit(1)

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} into {raw_dir} ...")
    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "--unzip",
            "-p",
            str(raw_dir),
        ],
        check=False,
    )
    if result.returncode != 0:
        print("\nDownload failed. Check your Kaggle API credentials and connection.")
        sys.exit(1)
    print("Download complete.\n")


def main() -> None:
    """Entry point for the download script."""
    parser = argparse.ArgumentParser(
        description="Manage the NASA C-MAPSS turbofan dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--kaggle",
        action="store_true",
        help="Download dataset via Kaggle API",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Verify all expected files are present",
    )
    args = parser.parse_args()

    if args.check:
        print(f"Checking data files in {RAW_DIR} ...\n")
        all_present = check()
        sys.exit(0 if all_present else 1)

    if args.kaggle:
        download_kaggle()
        print("Verifying downloaded files ...\n")
        check()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the --check flag runs without error**

Run:
```bash
python scripts/download_data.py --check
```

Expected: output listing 12 `[MISSING]` entries and exit code 1 (data not yet downloaded — this is correct behaviour).

- [ ] **Step 3: Commit**

```bash
git add scripts/download_data.py
git commit -m "feat: add download_data.py with Kaggle API support and file verification"
```

---

## Task 9: Lint and type-check pass

**Files:** no new files — fix any issues found

- [ ] **Step 1: Run ruff**

Run:
```bash
ruff check src/ tests/ scripts/
```

Expected: `All checks passed.`

If there are errors, fix them before continuing. Common fixes:
- Missing `ANN` annotations: add return type or argument type
- `I` import order: run `ruff check --fix src/ tests/ scripts/` to auto-fix

- [ ] **Step 2: Run mypy**

Run:
```bash
mypy src/turbofan
```

Expected: `Success: no issues found in N source files`

If mypy reports errors on pandas operations (e.g., `error: Returning Any from function ...`), add targeted `# type: ignore[return-value]` comments at the call site. Do not suppress entire files.

- [ ] **Step 3: Run the full test suite one final time**

Run:
```bash
pytest -v --tb=short
```

Expected: all tests pass, no warnings about unresolved fixtures.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "chore: fix lint and type-check issues, all tests green"
```

---

## Task 10: Push to remote

- [ ] **Step 1: Push all commits**

Run:
```bash
git push
```

Expected: all commits pushed to `origin/master`.

---

## Self-Review Checklist

**Spec coverage:**
- [x] Installable `src/turbofan/` package — Task 1
- [x] `pyproject.toml` with ruff/mypy/pytest config — Task 1
- [x] `DataConfig` + `ProjectConfig` Pydantic v2 models — Task 3
- [x] `load_config()` from YAML — Task 3
- [x] `configs/default.yaml` — Task 5
- [x] `setup_logging()` + `get_logger()` — Task 4
- [x] `compute_rul_labels()` piecewise-linear — Task 6
- [x] `load_raw_train()`, `load_raw_test()`, `load_rul_labels()` — Task 7
- [x] `FileNotFoundError` with download hint — Task 7
- [x] `COLUMN_NAMES` constant — Task 7
- [x] `scripts/download_data.py` with `--kaggle` and `--check` — Task 8
- [x] `data/raw/`, `data/interim/`, `data/processed/` git-ignored — Task 2
- [x] `.gitkeep` files preserving directory structure — Task 2
- [x] All tests use fixtures, no real filesystem dependency — Tasks 3–7
- [x] Ruff + mypy pass — Task 9

**Type consistency:**
- `DataConfig` defined in Task 3, used in Tasks 5, 7 ✓
- `COLUMN_NAMES: list[str]` defined in Task 7, referenced in test ✓
- `compute_rul_labels(df: pd.DataFrame, max_rul: int) -> pd.Series` consistent across Task 6 ✓
- `load_raw_train(cfg: DataConfig) -> pd.DataFrame` consistent across Tasks 7 ✓
