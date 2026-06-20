# Turbofan Remaining Useful Life Prediction

This repository implements an end-to-end machine learning workflow for estimating
turbofan engine Remaining Useful Life (RUL) from NASA C-MAPSS run-to-failure
data. It covers data preparation, exploratory analysis, feature engineering,
baseline and sequence models, experiment tracking, model registry operations,
batch inference, API serving, Docker deployment, and tests.

## Problem

Remaining Useful Life prediction estimates how many operating cycles remain
before an engine reaches failure. In the C-MAPSS dataset, each engine is observed
over multiple cycles with operating settings and sensor readings. The modeling
task is to predict RUL for unseen engines from their multivariate time series.

This project works across all four C-MAPSS subsets:

| Subset | Fault modes | Operating conditions | Notes |
|---|---:|---:|---|
| FD001 | 1 | 1 | Single-condition baseline case |
| FD002 | 1 | 6 | Multi-condition operating regime |
| FD003 | 2 | 1 | Multiple fault modes, single condition |
| FD004 | 2 | 6 | Multiple fault modes and operating regimes |

RUL labels are capped at 125 cycles with a piecewise-linear scheme, reflecting
the usual assumption that early-life degradation is not directly observable.
Validation uses an engine-level split. Official test evaluation uses one
prediction per held-out test engine at its final observed cycle.

## Repository Contents

- Dataset download and validation CLI for the C-MAPSS files.
- EDA utilities and notebooks for all four subsets.
- Pydantic-validated YAML configuration with recursive `_base_` composition.
- Per-subset configs for FD001-FD004, including model-specific feature settings.
- Shared preprocessing pipeline for sensor dropping, operating-mode
  normalization, sensor selection, and feature engineering.
- Ridge regression baseline training and evaluation.
- GRU and LSTM sequence model support through a shared RNN architecture registry.
- Hyperparameter and feature-engineering sweep tooling.
- Analysis reports linked to benchmark CSV snapshots under `results/baselines/`.
- MLflow tracking and local model registry integration.
- Registered-model promotion and listing CLIs.
- Batch prediction, FastAPI serving, and Docker serving paths.
- Unit and smoke tests for data, config, features, models, inference, tracking,
  CLI behavior, and experiment harnesses.

## Results and Findings

The tables below compare the selected Ridge, GRU, and LSTM configurations for
each C-MAPSS subset. Ridge uses the best feature configuration from the Ridge
feature sweep; GRU and LSTM use the winning feature/sequence cells from the
sequence feature-family screen. The committed benchmark snapshot lives under
`results/baselines/latest_official_eval_*.csv`. The numbers can be regenerated
by `turbofan-regenerate-baselines`, which trains each configuration and evaluates it on
the official C-MAPSS test set, writing
`outputs/results/latest_official_eval_per_run.csv` (one row per
model/subset/seed) and `outputs/results/latest_official_eval_summary.csv` by
default. Pass `--output-dir results/baselines` only when intentionally
refreshing the committed snapshot. GRU and LSTM are reported as `mean ± sd` over
five model seeds (42-46); Ridge is deterministic apart from the split/normalizer
seed and is reported at the single production seed (42).

Validation RMSE:

| Subset | Ridge | GRU | LSTM |
|---|---:|---:|---:|
| FD001 | 20.72 | 10.39 ± 0.33 | 9.72 ± 0.41 |
| FD002 | 19.35 | 13.08 ± 0.15 | 13.19 ± 0.23 |
| FD003 | 17.07 | 10.46 ± 0.22 | 10.53 ± 0.33 |
| FD004 | 18.48 | 14.96 ± 0.10 | 15.04 ± 0.20 |

Official test set:

| Subset | Ridge RMSE | Ridge PHM08 | GRU RMSE | GRU PHM08 | LSTM RMSE | LSTM PHM08 |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 21.58 | 1,316 | 14.29 ± 0.11 | 330 ± 13 | 14.99 ± 0.20 | 334 ± 23 |
| FD002 | 31.30 | 17,700 | 24.34 ± 0.40 | 4,992 ± 501 | 25.05 ± 0.60 | 5,470 ± 907 |
| FD003 | 23.01 | 2,491 | 14.22 ± 0.17 | 414 ± 22 | 13.98 ± 0.30 | 332 ± 29 |
| FD004 | 32.88 | 9,643 | 26.00 ± 0.41 | 4,603 ± 507 | 26.11 ± 0.56 | 4,531 ± 399 |

Main findings:

- Sequence models substantially outperform Ridge on both validation and
  official-test evaluation.
- GRU is the best official-test RMSE model on FD001, FD002, and FD004; LSTM
  slightly leads on FD003.
- Ridge's selected feature set is consistently `raw+rolling_mean`, with short to
  medium rolling windows.
- GRU prefers `raw+rolling_slope` on FD001/FD002, `raw` on FD003, and
  `raw+rolling_mean` on FD004.
- LSTM prefers `raw+rolling_slope` on FD001/FD002, `raw+rolling_delta` on FD003,
  and `raw+rolling_mean` on FD004.

## Analysis Reports

The detailed experiment design, tables, interpretation, and limitations are in the project reports:

| Report | Purpose |
|---|---|
| [Ridge feature sweep](docs/feature_sweep_ridge_report.md) | Feature-family and rolling-window analysis for Ridge |
| [Ridge vs GRU comparison](docs/feature_sweep_ridge_vs_gru.md) | Cross-model feature-engineering comparison and official test results |
| [GRU capacity sweep](docs/gru_capacity_sweep_report.md) | Two-stage temporal-context and capacity selection for GRU |
| [Sequence feature-family screen](docs/feature_family_screen_report.md) | GRU/LSTM feature-family screen with seed-noise replication |

Archived reports are kept under [docs/archive](docs/archive/README.md) when a
newer analysis supersedes the original result.

## System Design

The project is organized around reproducible, configuration-driven experiments:

- **Configuration:** Shared defaults live in `configs/default.yaml`; subset
  configs override only the values that differ. GRU and LSTM configs are explicit
  version-controlled files, for example `configs/subsets/fd001_gru.yaml` and
  `configs/subsets/fd001_lstm.yaml`.
- **Preprocessing contract:** All models use the same sklearn-compatible
  pipeline: `SensorDropper -> OperatingModeNormalizer -> SensorColumnSelector ->
  FeatureEngineer`.
- **Feature selection:** Feature families are selected by config and can differ
  per model through `features.ridge`, `features.gru`, and `features.lstm`.
- **Modeling:** Ridge provides a linear baseline. GRU and LSTM share the same
  sequence training and inference path, with `sequence.architecture` selecting
  the recurrent cell.
- **Experiment tracking:** Production runs and sweeps log params, metrics,
  artifacts, and per-epoch curves to a local MLflow SQLite store.
- **Model registry:** Serving resolves models by registered name and alias, not
  by a local artifact path. Promotion and rollback are handled by registry alias
  updates.
- **Serving:** Batch prediction, FastAPI, and Docker all use the same registry
  resolution path.

## Project Structure

```text
configs/
  default.yaml                        # Shared base configuration
  subsets/                            # Per-subset and per-architecture configs
data/                                 # Raw, interim, and processed dataset files
docs/                                 # Analysis reports and design notes
notebooks/                            # EDA notebooks for FD001-FD004
outputs/                              # Git-ignored generated run outputs
results/                              # Reviewed benchmark snapshots
  baselines/                          # CSV/JSONL data used by reports
mlflow.db                             # Local MLflow run + registry store (git-ignored)
artifacts/                            # Per-run records; model bytes live in MLflow
jobs/slurm/                           # SLURM job scripts for HPC experiments
scripts/                              # Local helper scripts
src/turbofan/
  cli/                                # Command-line entrypoints
  config/                             # Pydantic config schema
  data/                               # C-MAPSS loading, parsing, labels
  eda/                                # EDA utilities
  experiments/                        # Sweep and screen harnesses
  features/                           # Preprocessing and feature engineering
  models/                             # Baseline, sequence models, metrics
  predictions/                        # Pure RUL compute from trained models
  preprocessing/                      # Normalization components
  sequences/                          # Windowing, datasets, feature selection
  serving/                            # Schemas, FastAPI service, pyfunc adapter
  utils/                              # Shared utilities
tests/                                # Unit and smoke tests
```

## Quickstart

Use Python 3.12. To install with pip:

```bash
pip install -e ".[dev]"
```

Or create the conda environment:

```bash
conda env create -f environment.yml
conda activate mlops
pip install -e ".[dev]"
```

Download and verify the C-MAPSS dataset:

```bash
# Requires Kaggle API credentials (~/.kaggle/kaggle.json)
turbofan-download-data --kaggle

# Verify all expected files are present
turbofan-download-data --check
```

This downloads all four C-MAPSS subsets into `data/raw/`.

## Common Workflows

Train the Ridge baseline:

```bash
turbofan-train-baseline --config configs/subsets/fd001.yaml
```

Train a sequence model. The architecture is read from
`sequence.architecture` in the config:

```bash
turbofan-train-sequence --config configs/subsets/fd001_gru.yaml
turbofan-train-sequence --config configs/subsets/fd001_lstm.yaml
```

Browse MLflow runs and registered models:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Promote or inspect registered models:

```bash
turbofan-promote turbofan-gru-fd001 3 --to production
turbofan-models
```

Run batch prediction with the production model alias:

```bash
turbofan-predict \
  --model turbofan-gru-fd001 \
  --input data.csv \
  --output predictions.csv \
  --metadata-output metadata.json
```

When `--data-dir data/raw --subset FD001` are provided and official labels
align with the prediction count, the prediction CLI also reports RMSE, MAE, and
PHM08 score.

## CLI Commands

All commands are installed as entry points via `pyproject.toml`:

| Command | Purpose |
|---|---|
| `turbofan-download-data` | Download C-MAPSS data from Kaggle or verify files |
| `turbofan-train-baseline` | Train the Ridge regression baseline |
| `turbofan-train-sequence` | Train GRU or LSTM based on `sequence.architecture` |
| `turbofan-feature-screen` | Run the sequence feature-family screen |
| `turbofan-regenerate-baselines` | Train and officially evaluate the selected configs across seeds into the eval CSVs |
| `turbofan-predict` | Run batch prediction and optional official-label evaluation |
| `turbofan-serve-api` | Start the FastAPI inference server |
| `turbofan-promote` | Promote a registered model version to an alias |
| `turbofan-models` | List registered models, versions, aliases, and metrics |

CLI diagnostics go to stderr through leveled stdlib logging. Result values such
as run directories and validation metrics are printed to stdout. Production
training commands also attach `logs/run.log` to the MLflow run.

## Configuration

`configs/default.yaml` holds shared settings. Subset configs use `_base_`
composition and override only subset-specific values:

```yaml
# configs/subsets/fd002.yaml
_base_: ../default.yaml

data:
  fd_subset: FD002

features:
  sensor_cols_to_drop:
    - s_1
    - s_5
    - s_10
    - s_18
    - s_19
  n_modes: 6
  ridge:
    feature_families: [raw, rolling_mean]
    windows: [4]
  gru:
    feature_families: [rolling_mean]
    windows: [15]
```

Each subset also has explicit GRU and LSTM configs:

```bash
configs/subsets/fd001_gru.yaml
configs/subsets/fd001_lstm.yaml
configs/subsets/fd002_gru.yaml
configs/subsets/fd002_lstm.yaml
configs/subsets/fd003_gru.yaml
configs/subsets/fd003_lstm.yaml
configs/subsets/fd004_gru.yaml
configs/subsets/fd004_lstm.yaml
```

The `sensor_cols_to_drop` lists come from the EDA notebooks. Model-specific
feature blocks override the shared feature settings only where needed.

## Modeling

| Model | Role |
|---|---|
| Ridge Regression | Linear baseline over engineered tabular features |
| GRU | Recurrent sequence model for temporal degradation patterns |
| LSTM | Alternative recurrent architecture on the same sequence path |

The sequence path uses a single `SequenceRULRegressor` for the encoder,
regression head, packed-sequence handling, training loop, and inference contract.
Only the recurrent cell changes between GRU and LSTM.

Sequence models share one training-hyperparameter surface:
`window_size`, `hidden_size`, `num_layers`, `dropout`, `learning_rate`,
`weight_decay`, `batch_size`, `epochs`, and `patience`. Inter-layer `dropout`
only applies when `num_layers > 1`; for a single recurrent layer, `weight_decay`
is the explicit regularizer.

Feature families are computed per engine without crossing engine boundaries.
Supported families include `raw`, `rolling_mean`, `lag`, `rolling_std`,
`rolling_min`, `rolling_max`, `rolling_slope`, and `rolling_delta`.

## Evaluation

| Metric | Purpose |
|---|---|
| RMSE | Primary ranking metric; emphasizes larger errors |
| MAE | Average absolute prediction error |
| PHM08 score | Asymmetric prognostics score; penalizes late predictions more heavily |

Validation and sweep ranking use RMSE and MAE. PHM08 is computed only on the
official test set, where each test engine contributes one prediction. Per-run
metrics and prediction CSVs are written under
`artifacts/models/<model_type>/<timestamp>/`; cross-run summaries are written
under `outputs/results/` by default. Reviewed benchmark snapshots used by the
reports live under `results/baselines/`.

## Inference and Serving

Batch prediction, FastAPI serving, and Docker serving resolve registered models
by name from MLflow:

```bash
turbofan-serve-api --model turbofan-gru-fd001 --host 127.0.0.1 --port 8000
```

The server exposes:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Return loaded model metadata and status |
| `/predict` | POST | Return RUL predictions for engine sensor records |

The served model can be selected with `--model` and `--alias`, or through
`TURBOFAN_MODEL_NAME` and `TURBOFAN_MODEL_ALIAS`. Set `MLFLOW_TRACKING_URI` when
using a non-default registry store.

### Docker

Build and start the API:

```bash
docker compose up --build
```

By default this serves `turbofan-gru-fd001@production`. Override the model or
registry store with environment variables:

```bash
TURBOFAN_MODEL_NAME=turbofan-ridge-fd001 \
MLFLOW_STORE_DIR=./mlflow_store docker compose up
```

Check health:

```bash
curl http://localhost:8000/health
```

Query the API with C-MAPSS test data:

```bash
python scripts/query_api.py
python scripts/query_api.py --subset FD002
```

The helper reads the matching `data/raw/test_FD00X.txt`, posts engine records to
`/predict`, and prints a per-engine prediction table with RMSE and MAE when
labels are available.

## Reproducibility

The workflow is designed to be reproducible through:

- Scripted data acquisition and file checks.
- Version-controlled YAML configs.
- Deterministic random seeds where supported.
- Engine-level validation splits.
- MLflow run tracking and registry versions.
- Saved metrics, prediction CSVs, and sweep summaries.
- Documented command-line workflows.

Neural-network results can still vary slightly across hardware and dependency
versions.

## Tests and Quality Checks

Run the full test suite:

```bash
pytest
```

Run linting and type checking:

```bash
ruff check src/ tests/
mypy src/turbofan
```

Tests use synthetic fixtures and cover data loading, labels, feature
engineering, normalization, model training, sequence windowing, metrics,
inference schemas, CLI commands, MLflow tracking, registry behavior, and
experiment harnesses.

## Current Status

Completed work includes the full FD001-FD004 workflow, EDA, Ridge baseline,
GRU sequence model, LSTM architecture support, feature-engineering sweeps,
selected production Ridge/GRU runs, MLflow tracking and registry operations,
batch prediction, FastAPI serving, Docker serving, and tests.

The next step is Bayesian modeling of the feature-engineering results: estimate
feature-family, architecture, subset, window, sequence-length, and seed effects
jointly before spending more compute on a narrower confirmation sweep. Current
follow-up areas also include production LSTM benchmarking and transformer-based
sequence modeling. See [docs/roadmap.md](docs/roadmap.md) for detailed status
and design decisions.

## License

MIT
