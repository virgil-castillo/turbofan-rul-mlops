# Turbofan Remaining Useful Life Prediction

This repository is a reproducible ML project for estimating turbofan engine Remaining Useful Life (RUL) using NASA C-MAPSS data. EDA, per-subset configuration, and model training are complete for all four C-MAPSS subsets (FD001–FD004).

## What this repo contains

- Configuration-driven training experiments with Pydantic-validated YAML configs
- Ridge regression baseline modeling
- GRU-based sequence modeling for temporal degradation patterns
- Evaluation with RMSE, MAE, and PHM08 score
- Hyperparameter sweep experiments (Ridge alpha, GRU architecture, feature selection)
- Unified feature-engineering sweep running Ridge and GRU on identical features across all four C-MAPSS subsets
- Cross-model feature-engineering analysis (Ridge vs GRU) with findings reports grounded in the sweep data and EDA
- Exploratory data analysis utilities and notebooks
- Saved model artifacts and experiment outputs
- Tests for core project components
- Batch prediction, FastAPI serving, and Docker serving interfaces

## Current Scope

| Dataset | EDA | Config | Training |
|---|---|---|---|
| FD001 | Done | Done | Done |
| FD002 | Done | Done | Done |
| FD003 | Done | Done | Done |
| FD004 | Done | Done | Done |

## Key findings

Production models trained on each subset's best feature config (from the sweep),
evaluated on both the engine-level validation split and the official C-MAPSS test set:

| Subset | Ridge val RMSE | Ridge test RMSE | GRU val RMSE | GRU test RMSE |
|--------|---:|---:|---:|---:|
| FD001 | 20.72 | 21.58 | 10.91 | 15.81 |
| FD002 | 19.35 | 31.31 | 13.17 | 24.63 |
| FD003 | 17.07 | 23.01 | 10.57 | 14.76 |
| FD004 | 18.47 | 32.88 | 14.73 | 23.67 |

- The GRU roughly **halves** validation RMSE versus the Ridge baseline on the same features; the gap narrows on official test but GRU still leads by 27–38%.
- The optimal rolling-mean window is **opposite** for the two models — Ridge favors
  short windows (2–4), the GRU favors long ones (~15) — because Ridge is memoryless
  while the GRU already models the sequence.
- Lag-difference features are counterproductive: additive-neutral for Ridge, and they
  prevent the GRU from converging.
- Multi-condition subsets (FD002, FD004) show larger val→test gaps for both models.

Full sweep analysis:
[Ridge](docs/feature_sweep_ridge_report.md) ·
[GRU](docs/feature_sweep_gru_report.md) ·
[Ridge vs GRU](docs/feature_sweep_ridge_vs_gru.md)

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

This downloads all four C-MAPSS subsets (FD001–FD004) into `data/raw/`.

Train the Ridge baseline:

```bash
turbofan-train-baseline --config configs/subsets/fd001.yaml
```

Train the GRU sequence model:

```bash
turbofan-train-sequence-gru --config configs/subsets/fd001.yaml
```

Model training creates a timestamped run directory under `artifacts/models/<model_type>/<timestamp>/`. Each run directory is self-contained and includes the model checkpoint, config snapshot, manifest, metrics, training history, and prediction CSVs.

Cross-run experiment summaries and the global append-only training log are written to `results/`.

## Project Structure

```text
configs/
  default.yaml                        # Shared base configuration
  subsets/                            # Per-subset overrides (fd001–fd004)
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
| `turbofan-sweep-features` | Unified Ridge/GRU feature-engineering sweep through the shared preprocessing pipeline |
| `turbofan-sweep-gru-temporal` | Stage 1: cross GRU sequence window size with the rolling-feature grid per subset |
| `turbofan-sweep-gru-capacity` | Stage 2: cross GRU hidden size and learning rate on the top Stage 1 configs |
| `turbofan-predict` | Batch prediction and optional official-label evaluation from a saved artifact |
| `turbofan-serve-api` | FastAPI inference server |

## Configuration

`configs/default.yaml` holds all shared settings. Per-subset configs in `configs/subsets/` override only what differs using a `_base_` composition key:

```yaml
# configs/subsets/fd002.yaml
_base_: ../default.yaml

data:
  fd_subset: FD002

features:
  sensor_cols_to_drop:    # EDA-derived drop list for FD002
    - s_1
    - s_5
    - s_10
    - s_18
    - s_19
  n_modes: 6              # 6 operating conditions
  ridge:                  # Ridge's best feature config (from the sweep)
    feature_set: raw_plus_rolling_mean
    windows: [4]
  gru:                    # GRU's best feature config (from the sweep)
    feature_set: rolling_mean
    windows: [15]
```

Pass a subset config directly to any training CLI:

```bash
turbofan-train-baseline --config configs/subsets/fd002.yaml
turbofan-train-sequence-gru --config configs/subsets/fd002.yaml
```

The `sensor_cols_to_drop` lists are derived from the EDA notebooks (`notebooks/eda_fd00{1-4}.ipynb`) and represent the authoritative drop decision for each subset.

The optional `features.ridge` and `features.gru` blocks set per-model feature engineering: the top-level `feature_set`/`windows` are shared fallbacks, and each model's training CLI resolves its own block (inheriting the shared values for anything left unset). The values committed per subset are each model's best configuration from the feature sweep.

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
| Ridge Regression | Linear baseline trained on engineered tabular features |
| GRU | Recurrent sequence model trained on sliding windows of sensor readings |

Both models share the same preprocessing contract: a 4-step sklearn Pipeline (`SensorDropper → OperatingModeNormalizer → SensorColumnSelector → FeatureEngineer`). The `feature_set` config key selects which engineered features both models receive — `raw`, `rolling_mean`, `rolling_stats`, `raw_plus_rolling_mean`, `raw_plus_rolling_stats`, `lag`, or `raw_plus_lag`. Rolling and lag features are computed per engine without crossing engine boundaries; `lag` is a normalized lag-difference `(x[t] - x[t-N]) / rolling_mean(x, N)`. Feature settings can differ per model via the `features.ridge` / `features.gru` config blocks.

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

Batch prediction, FastAPI serving, and Docker serving have been validated end-to-end against trained model artifacts.

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

A `docker-compose.yml` is included for containerized deployment. It mounts the model run directory into the container at `/models/`, where the server expects `model_manifest.json`.

Build and start the server:

```bash
docker compose up --build
```

By default this mounts the most recently trained GRU run. To point at a different run:

```bash
MODEL_RUN_DIR=./artifacts/models/sequence_gru/<timestamp> docker compose up
```

Verify the container is healthy and the model loaded:

```bash
curl http://localhost:8000/health
```

To send test data and compare predictions against true RUL labels:

```bash
python scripts/query_api.py                        # FD001 (default)
python scripts/query_api.py --subset FD002         # other subsets
```

This reads `data/raw/test_FD001.txt`, POSTs all engine records to `/predict`, and prints a per-engine prediction table with RMSE and MAE. The reported RMSE should match `official_test rmse` from the training run.

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
- [x] Validate Docker serving end-to-end
- [x] Operating-mode normalization (OperatingModeNormalizer, self-contained GRU artifacts)
- [x] Config-driven sensor dropping (EDA-derived explicit drop list replaces runtime std-threshold)
- [x] EDA notebooks for all four subsets with correlation-based sensor filter
- [x] Per-subset configs with `_base_` composition (sensor drop lists and n_modes from EDA)
- [x] Unified feature pipeline — Ridge and GRU share the same 4-step preprocessing contract; `feature_set` is config-driven
- [x] Unified feature-engineering sweep CLI (`turbofan-sweep-features`) with Ridge vs GRU analysis across all four subsets
- [x] Train and persist baseline and GRU production artifacts on FD002–FD004
- [x] Cross-dataset benchmark table from persisted models
- [x] Left-zero-pad short engines in GRU windowing (packed sequences; no engine skipped)
- [x] Two-stage GRU temporal-context and capacity sweep CLIs (`turbofan-sweep-gru-temporal`, `turbofan-sweep-gru-capacity`) with SLURM drivers
- [ ] Additional models (LSTM, Transformer)
- [ ] Advanced feature engineering
- [ ] MLOps infrastructure (experiment tracking, CI/CD)

See [docs/roadmap.md](docs/roadmap.md) for detailed priorities and design decisions.


## License

MIT
