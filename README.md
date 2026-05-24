# Turbofan RUL Prediction — MLOps

End-to-end remaining useful life (RUL) prediction system for turbofan engines using the NASA C-MAPSS dataset. Covers data loading, exploratory analysis, feature engineering, baseline modeling, sequence modeling, and inference deployment.

## What this does

Predicts how many operating cycles a turbofan engine has left before failure, given multivariate sensor readings and operational settings. The system supports all four C-MAPSS subsets (FD001-FD004), each with different numbers of fault modes and operating conditions.

## Current capabilities

| Component | Status | Description |
|-----------|--------|-------------|
| Data loading | Done | Raw C-MAPSS loader with piecewise-linear RUL labels (capped at 125) |
| EDA | Done | Reusable analysis utilities (quality, sensor stats, degradation) + structured notebook |
| Feature engineering | Done | sklearn pipeline: drops constant sensors, computes rolling stats (windows 5/10/20), normalizes by operating condition |
| Baseline model | Done | Ridge regression trained on engineered features, engine-level validation split, PHM08 scoring |
| Sequence models | Planned | LSTM/transformer models operating on time-series windows |
| Inference & deployment | Planned | Serving API and batch prediction |

## Project structure

```
turbofan-rul-mlops/
├── configs/
│   └── default.yaml                 # Project config (data paths, model params)
├── data/
│   └── raw/                         # C-MAPSS .txt files (git-ignored, download via script)
├── notebooks/
│   └── 01_eda_fd001.ipynb           # EDA notebook (outputs stripped via nbstripout)
├── scripts/
│   ├── download_data.py             # Download dataset from Kaggle
│   └── train_baseline.py            # Train and evaluate baseline model
├── src/turbofan/
│   ├── config/schema.py             # Pydantic config with validation
│   ├── data/
│   │   ├── loader.py                # Raw data I/O
│   │   └── labels.py                # RUL label computation
│   ├── eda/
│   │   ├── quality.py               # Missing values, constant sensors, dtypes
│   │   ├── sensors.py               # Sensor stats, correlation, noise estimation
│   │   └── degradation.py           # RUL curves, trends, sensor selection
│   ├── features/
│   │   ├── sensor_dropper.py        # Drop zero-variance sensors (sklearn transformer)
│   │   ├── rolling.py               # Multi-window rolling stats (sklearn transformer)
│   │   ├── normalizer.py            # Per-condition z-score (sklearn transformer)
│   │   └── pipeline.py              # Pipeline factory
│   ├── models/
│   │   ├── split.py                 # Engine-level train/validation split
│   │   ├── metrics.py               # RMSE, MAE, PHM08 score
│   │   ├── baseline.py              # Ridge baseline pipeline factory
│   │   ├── evaluate.py              # Validation and official test evaluation
│   │   └── artifacts.py             # Local experiment artifact persistence
│   └── utils/logging.py             # Stdlib logging setup
├── tests/                           # 127 tests, all synthetic data (no download needed)
├── artifacts/                       # Model outputs (git-ignored)
└── pyproject.toml                   # Package metadata, tool config
```

## Setup

```bash
# Create conda environment
conda create -n mlops python=3.12
conda activate mlops

# Install package in editable mode with dev tools
pip install -e ".[dev]"

# Install nbstripout git filter (strips notebook outputs on commit)
nbstripout --install
```

## Download data

```bash
# Requires Kaggle API credentials (~/.kaggle/kaggle.json)
python scripts/download_data.py --kaggle

# Verify files are present
python scripts/download_data.py --check
```

This downloads the NASA C-MAPSS dataset (FD001-FD004) into `data/raw/`.

## Train baseline model

```bash
python scripts/train_baseline.py --config configs/default.yaml
```

Outputs a fitted Ridge model, metrics JSON, and prediction CSVs to `artifacts/models/baseline/<timestamp>/`. Evaluates on a held-out engine split and, if test data is present, on the official C-MAPSS test set.

## Run tests

```bash
pytest                          # all 127 tests
pytest tests/models/ -v         # just model tests
pytest -k "test_rolling" -v     # single test by pattern
```

## Lint and type-check

```bash
ruff check src/ tests/          # lint (E, F, W, I, UP, ANN rules)
mypy src/turbofan               # strict type checking
```

## Configuration

All settings live in `configs/default.yaml`:

```yaml
project_name: turbofan-rul-mlops

data:
  raw_dir: data/raw
  fd_subset: FD001          # FD001, FD002, FD003, or FD004
  max_rul: 125              # RUL cap for piecewise-linear labels
  test_size: 0.2            # Validation split fraction
  random_seed: 42

model:
  name: ridge
  alpha: 1.0                # Ridge regularization strength
  artifact_dir: artifacts/models
```
