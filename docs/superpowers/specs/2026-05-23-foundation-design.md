# Foundation: Sub-project 1 Design Spec

**Date:** 2026-05-23
**Sub-project:** 1 of 6
**Status:** Approved

---

## Goal

Establish the installable `turbofan` Python package with domain subpackages, a type-validated config system, stdlib logging, a raw data loader with piecewise-linear RUL label computation, and a NASA C-MAPSS download script.

## Architecture

The package uses the `src` layout (`src/turbofan/`) installed via `pip install -e .`. All code lives in three subpackages — `data/`, `config/`, `utils/` — each with a single clear responsibility. Config is loaded once at the entry point from a YAML file and passed explicitly to all functions; no global singletons. The data layer separates I/O (`loader.py`) from pure computation (`labels.py`) so label logic can be unit-tested without touching the filesystem.

## Tech Stack

Python 3.12, pandas, numpy, pydantic v2, pyyaml, pydantic-settings, pytest, ruff, mypy

---

## File Structure

```
turbofan-rul-mlops/
├── pyproject.toml                  # package metadata, ruff/mypy/pytest config
├── environment.yml                 # conda environment (existing)
├── .gitignore                      # extend to cover data/, outputs/
├── configs/
│   └── default.yaml                # data paths, max_rul, fd_subset, seed
├── data/
│   ├── raw/                        # git-ignored; populated by download script
│   ├── interim/                    # git-ignored; intermediate outputs
│   └── processed/                  # git-ignored; final features
├── src/
│   └── turbofan/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py           # load_raw_train(), load_raw_test(), load_rul_labels()
│       │   └── labels.py           # compute_rul_labels() — piecewise-linear RUL
│       ├── config/
│       │   ├── __init__.py
│       │   └── schema.py           # DataConfig, ProjectConfig (Pydantic v2)
│       └── utils/
│           ├── __init__.py
│           └── logging.py          # setup_logging(), get_logger()
├── scripts/
│   └── download_data.py            # --kaggle download or --check validation
├── tests/
│   ├── conftest.py                 # shared fixtures
│   ├── data/
│   │   ├── test_loader.py
│   │   └── test_labels.py
│   └── config/
│       └── test_schema.py
└── docs/
    └── superpowers/
        └── specs/                  # design docs
```

No stub directories for `features/`, `models/`, etc. — those appear in later sub-projects.

---

## Component Designs

### 1. `pyproject.toml`

Defines the `turbofan` package using the `src` layout. Configures:
- `[project]`: name, version, Python `>=3.12`, dependencies (pandas, numpy, pydantic, pyyaml)
- `[project.optional-dependencies]`: `dev` (pytest, pytest-cov, ruff, mypy, kaggle)
- `[tool.ruff]`: rules `E`, `F`, `W`, `I`, `UP`, `ANN`; line length 88
- `[tool.mypy]`: `strict = true`
- `[tool.pytest.ini_options]`: `testpaths = ["tests"]`

### 2. Config system (`src/turbofan/config/schema.py`)

Two nested Pydantic v2 `BaseModel` classes:

```python
class DataConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path
    interim_dir: Path
    fd_subset: Literal["FD001", "FD002", "FD003", "FD004"] = "FD001"
    max_rul: int = 125
    test_size: float = 0.2
    random_seed: int = 42

class ProjectConfig(BaseModel):
    project_name: str
    data: DataConfig
```

One loader function:
```python
def load_config(path: Path) -> ProjectConfig:
    raw = yaml.safe_load(path.read_text())
    return ProjectConfig.model_validate(raw)
```

Config is loaded once at the entry point and passed explicitly — no global singletons, no `os.environ` fishing inside library code.

### 3. `configs/default.yaml`

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

Paths are relative to the project root. The package assumes all scripts are run from the project root; loaders construct paths as `cfg.raw_dir / filename` with no automatic resolution in the validator.

### 4. Data loader (`src/turbofan/data/loader.py`)

C-MAPSS files are space-delimited with no header, 26 columns:

```
engine_id | cycle | op_1 | op_2 | op_3 | s_1 … s_21
```

Column names defined as a module-level constant (`COLUMN_NAMES: list[str]`).

Three public functions:

```python
def load_raw_train(cfg: DataConfig) -> pd.DataFrame:
    """Load train_{fd_subset}.txt with assigned column names."""

def load_raw_test(cfg: DataConfig) -> pd.DataFrame:
    """Load test_{fd_subset}.txt with assigned column names."""

def load_rul_labels(cfg: DataConfig) -> pd.Series:
    """Load RUL_{fd_subset}.txt — ground-truth RUL for test engines."""
```

Each raises `FileNotFoundError` with a message pointing to `scripts/download_data.py` when the file is missing.

### 5. RUL label computation (`src/turbofan/data/labels.py`)

Pure function with no I/O — separated from `loader.py` so it can be unit-tested with literal values and reused in the feature engineering sub-project:

```python
def compute_rul_labels(df: pd.DataFrame, max_rul: int = 125) -> pd.Series:
    """
    Compute piecewise-linear RUL labels for training data.

    For each engine: RUL = min(max_rul, max_cycle - current_cycle).
    Returns a Series aligned to df.index.

    Args:
        df: DataFrame with columns engine_id and cycle.
        max_rul: Maximum RUL cap (default 125 for FD001).

    Returns:
        pd.Series of integer RUL values, same index as df.
    """
```

### 6. Logging (`src/turbofan/utils/logging.py`)

Standard library `logging`. Two functions:

```python
def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with timestamped console output."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a module."""
    return logging.getLogger(name)
```

Usage pattern: `logger = get_logger(__name__)` at module top level.

File logging to `outputs/logs/` is deferred to the training sub-project where there is a reason to persist run logs.

### 7. Download script (`scripts/download_data.py`)

Two modes:

- `--kaggle`: uses `kaggle datasets download -d behrad3d/nasa-cmaps` if `~/.kaggle/kaggle.json` exists; extracts to `data/raw/`.
- `--check`: validates all expected files exist (`train_FD001.txt` … `RUL_FD004.txt`) and prints file sizes.

If Kaggle API is not configured, `--kaggle` prints manual download instructions and exits with code 1. `kaggle` is an optional dev dependency — not required to import or use the package.

---

## Data Management

- `data/raw/`, `data/interim/`, `data/processed/` are all git-ignored.
- Raw files are never modified — all transformations write to `interim/` or `processed/`.
- The download script is the single source of truth for getting data onto disk.

## RUL Label Convention

Piecewise-linear with cap:

```
RUL(engine, cycle) = min(max_rul, total_cycles(engine) - cycle)
```

Default `max_rul = 125` (FD001 literature standard). Configurable via `DataConfig.max_rul`.

---

## Testing Strategy

All tests use synthetic fixtures — no dependency on real downloaded data.

**`tests/conftest.py`**
- `sample_train_df`: minimal valid train DataFrame with 3 engines and varied cycle lengths
- `tmp_data_dir`: `tmp_path`-based directory with correctly-named stub `.txt` files

**`tests/data/test_labels.py`** (pure unit tests)
- RUL is capped at `max_rul` for early cycles
- RUL reaches 0 at the last cycle of each engine
- Engines with different lifespans are handled correctly
- Custom `max_rul` values are respected

**`tests/data/test_loader.py`** (I/O tests against `tmp_data_dir`)
- Correct column names are assigned
- Returns DataFrame with expected dtypes
- Raises `FileNotFoundError` with helpful message when file is missing

**`tests/config/test_schema.py`**
- Valid YAML round-trips through `load_config` without error
- Invalid `fd_subset` raises `ValidationError`
- Missing required fields raise `ValidationError`
- Path fields resolve to `Path` objects

---

## What This Sub-project Does NOT Include

The following are explicitly deferred to later sub-projects:

- Feature engineering (`features/` subpackage) — Sub-project 3
- EDA notebooks and visualization utilities — Sub-project 2
- Any model code — Sub-projects 4 and 5
- MLflow integration — Sub-project 4
- FastAPI inference endpoint — Sub-project 6
- Docker — Sub-project 6
- File logging to `outputs/logs/` — Sub-project 4

---

## Sub-project Sequence

| # | Sub-project | Depends on |
|---|-------------|------------|
| 1 | **Foundation** ← you are here | — |
| 2 | EDA | 1 |
| 3 | Feature Engineering | 1 |
| 4 | Baseline Models | 1, 3 |
| 5 | Sequence Models | 1, 3 |
| 6 | Inference & Deployment | 1, 4 or 5 |
