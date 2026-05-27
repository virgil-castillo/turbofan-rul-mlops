# Turbofan Remaining Useful Life Prediction

This repository is a reproducible ML project for estimating turbofan engine Remaining Useful Life (RUL) using NASA C-MAPSS data. The current implementation targets FD001; support for FD002–FD004 is planned.

## What this repo contains

- Configuration-driven training experiments with Pydantic-validated YAML configs
- Ridge regression baseline modeling
- GRU-based sequence modeling for temporal degradation patterns
- Evaluation with RMSE, MAE, and PHM08 score
- Hyperparameter sweep experiments (Ridge alpha, GRU architecture, feature selection)
- Exploratory data analysis utilities and notebook
- Saved model artifacts and experiment outputs
- Tests for core project components
- Experimental batch prediction, FastAPI serving, and Docker serving interfaces

## Current Scope

| Dataset | Status |
|---|---|
| FD001 | Implemented |
| FD002 | Planned |
| FD003 | Planned |
| FD004 | Planned |

## Quickstart

Install the project dependencies:

```bash
pip install -e ".[dev]"
```

Or use the conda environment:

```bash
conda env create -f environment.yml
conda activate mlops
pip install -e ".[dev]"
```

Download and prepare the C-MAPSS dataset:

```bash
# Requires Kaggle API credentials (~/.kaggle/kaggle.json)
turbofan-download-data --kaggle

# Verify all expected files are present
turbofan-download-data --check
```

This downloads all four C-MAPSS subsets (FD001–FD004) into `data/raw/`. The current training workflow targets FD001.

Train the Ridge baseline:

```bash
turbofan-train-baseline --config configs/default.yaml
```

Train the GRU sequence model:

```bash
turbofan-train-sequence-gru --config configs/default.yaml
```

Model training creates a timestamped run directory under `artifacts/models/<model_type>/<timestamp>/`. Each run directory is self-contained and includes the model checkpoint, config snapshot, manifest, metrics, training history, and prediction CSVs.

Cross-run experiment summaries and the global append-only training log are written to `results/`.

## Project Structure

```text
configs/                              # Experiment configuration files
data/                                 # Raw, interim, and processed dataset files
docs/                                 # Analysis reports and documentation
notebooks/                            # Exploratory analysis notebooks
results/                              # Cross-run experiment summaries and global training logs
artifacts/                            # Per-run model artifacts, metrics, configs, manifests, histories, and predictions (git-ignored)
src/turbofan/
  cli/                                # Command-line entrypoints
  config/                             # Pydantic configuration schema
  data/                               # Dataset loading, parsing, and label logic
  eda/                                # Exploratory data analysis utilities
  features/                           # Feature engineering pipeline
  models/                             # Model definitions, training, and evaluation
  sequences/                          # Sequence windowing and DataLoader builders
  inference/                          # Prediction, model-loading, and serving
  experiments/                        # Hyperparameter sweeps and comparisons
  utils/                              # Shared utilities
jobs/slurm/                           # SLURM job scripts for HPC experiments
tests/                                # Unit and smoke tests
```

## CLI Commands

All commands are installed as entry points via `pyproject.toml`:

| Command | Purpose |
|---|---|
| `turbofan-download-data` | Download C-MAPSS data from Kaggle or verify files exist |
| `turbofan-train-baseline` | Train Ridge regression baseline |
| `turbofan-train-sequence-gru` | Train GRU sequence model |
| `turbofan-sweep-baseline-alpha` | Sweep Ridge regularization strength |
| `turbofan-compare-baseline-features` | Compare feature sets (raw, rolling, engineered) |
| `turbofan-sweep-gru` | Sweep GRU hyperparameters |
| `turbofan-sweep-feature-gru` | Sweep GRU with feature selection variants |
| `turbofan-predict` | Experimental batch prediction and optional official-label evaluation from a saved artifact |
| `turbofan-serve-api` | Experimental FastAPI inference server |

## Configuration

All settings live in `configs/default.yaml` and are validated by a Pydantic schema:

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
  alpha: 100.0
  feature_set: rolling
  artifact_dir: artifacts/models

sequence:
  architecture: gru
  window_size: 45
  hidden_size: 64
  num_layers: 1
  dropout: 0.0
  learning_rate: 0.001
  epochs: 50
  patience: 8
  artifact_dir: artifacts/models
```

## Dataset

The project uses NASA's C-MAPSS turbofan degradation dataset. Each subset contains multivariate time-series data where engines are observed over multiple operating cycles. The goal is to estimate how many cycles remain before failure from operating settings and sensor measurements.

| Subset | Fault modes | Operating conditions |
|---|---|---|
| FD001 | 1 | 1 |
| FD002 | 1 | 6 |
| FD003 | 2 | 1 |
| FD004 | 2 | 6 |

RUL labels are capped at 125 cycles using a piecewise-linear scheme, reflecting the assumption that degradation is not detectable in early engine life.

## Modeling Approach

The project includes both baseline and neural sequence modeling approaches.

| Model | Description |
|---|---|
| Ridge Regression | Linear baseline trained on engineered tabular features (rolling statistics, normalized by operating condition) |
| GRU | Recurrent sequence model trained on sliding windows of sensor readings |

The feature engineering pipeline drops constant sensors, computes rolling statistics over configurable windows, and normalizes features by operating condition. The GRU model operates on fixed-length sliding windows with its own normalization path.

## Evaluation

Models are evaluated using:

| Metric | Purpose |
|---|---|
| RMSE | Penalizes larger prediction errors |
| MAE | Measures average absolute prediction error |
| PHM08 score | Asymmetric scoring function from the prognostics community — penalizes late predictions more heavily than early ones |

GRU training reports RMSE, MAE, and PHM08 on an engine-level validation split. When the official FD001 test files are present, training also evaluates against `test_FD001.txt` using labels from `RUL_FD001.txt`.

Per-run metrics and prediction CSVs are saved in the run directory under `artifacts/models/sequence_gru/<timestamp>/`. Cross-run summaries are saved under `results/`.

## Inference and Serving

Batch prediction and FastAPI serving have been validated end-to-end against trained model artifacts. Docker serving is implemented but not yet tested.

Run batch prediction with a saved model artifact:

```bash
turbofan-predict \
  --artifact artifacts/models/baseline/<timestamp>/model_manifest.json \
  --input data.csv \
  --output predictions.csv \
  --metadata-output metadata.json
```

When `--data-dir data/raw --subset FD001` are provided and the matching
`RUL_FD001.txt` labels align with the prediction count, the CLI also reports
RMSE, MAE, and PHM08 score and writes them to the metadata JSON.

Start the FastAPI server locally:

```bash
turbofan-serve-api --host 127.0.0.1 --port 8000
```

The server exposes two endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Returns loaded model metadata and status |
| `/predict` | POST | Accepts engine sensor records and returns RUL predictions |

The model artifact is configured via the `TURBOFAN_MODEL_ARTIFACT` environment variable or passed at app creation time.

### Docker

A Dockerfile is included for containerized deployment of the inference server:

```bash
docker build -t turbofan-serve .

docker run -p 8000:8000 \
  -v /path/to/artifact:/models \
  turbofan-serve
```

The container expects a model manifest at `/models/model_manifest.json`.

## Reproducibility

The project is designed to make experiments reproducible through:

- Scripted data download
- Configuration-driven training
- Deterministic random seeds
- Saved model artifacts with manifests
- Saved evaluation outputs
- Documented command-line workflows

Exact numerical results may vary slightly across hardware and dependency versions, especially for neural-network training.

## Tests

Run the test suite with:

```bash
pytest
```

Tests use synthetic data fixtures and cover data loading, feature engineering, model training, sequence windowing, inference schemas, CLI commands, and experiment sweeps. Test coverage will continue to expand as support for additional C-MAPSS subsets is added.

## Lint and Type-Check

```bash
ruff check src/ tests/          # lint (E, F, W, I, UP, ANN rules)
mypy src/turbofan               # strict type checking
```

## Roadmap

- [x] FD001 data download and preparation
- [x] FD001 baseline modeling (Ridge)
- [x] GRU sequence model
- [x] Hyperparameter sweep experiments
- [x] RMSE, MAE, and PHM08 evaluation
- [x] Validate batch prediction end-to-end
- [x] Validate FastAPI serving end-to-end
- [x] Fix known inference bugs (GRU rescaling, Ridge prediction scope, predict CLI evaluation)
- [x] Remove official test-set evaluation from GRU sweeps
- [ ] Validate Docker serving end-to-end
- [ ] FD002–FD004 support
- [ ] Cross-dataset benchmark table
- [ ] Additional models (LSTM, Transformer)
- [ ] Advanced feature engineering
- [ ] MLOps infrastructure (experiment tracking, CI/CD)

See [docs/roadmap.md](docs/roadmap.md) for detailed priorities and design decisions.


## License

MIT
