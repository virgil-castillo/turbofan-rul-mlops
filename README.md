# Turbofan Remaining Useful Life Prediction

This repository is a reproducible ML project for estimating turbofan engine Remaining Useful Life (RUL) using NASA C-MAPSS data. EDA, per-subset configuration, and model training are complete for all four C-MAPSS subsets (FD001–FD004).

## What this repo contains

- Configuration-driven training experiments with Pydantic-validated YAML configs
- Ridge regression baseline modeling
- Recurrent sequence modeling (GRU and LSTM) for temporal degradation patterns, selected by config through a shared architecture registry
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

Production models evaluated on both the engine-level validation split and the
official C-MAPSS test set. Both models were retrained on CPU (2026-06-01) from
their per-subset selected configs — GRU from the Stage 2 capacity sweep, Ridge
from its best feature-sweep config. The Ridge numbers reproduce the prior deployed
values exactly (Ridge is deterministic given the same config and seed):

| Subset | Ridge val RMSE | Ridge test RMSE | GRU val RMSE | GRU test RMSE |
|--------|---:|---:|---:|---:|
| FD001 | 20.72 | 21.58 | 10.19 | 15.40 |
| FD002 | 19.35 | 31.31 | 12.78 | 25.08 |
| FD003 | 17.07 | 23.01 | 10.07 | 14.16 |
| FD004 | 18.47 | 32.88 | 14.73 | 25.58 |

- The GRU roughly **halves** validation RMSE versus the Ridge baseline on the same features; the gap narrows on official test but GRU still leads by 20–38%.
- The optimal rolling-mean window is **opposite** for the two models — Ridge favors
  short windows (2–4), the GRU favors long ones (~15) — because Ridge is memoryless
  while the GRU already models the sequence.
- Lag-difference features are counterproductive: additive-neutral for Ridge, and they
  prevent the GRU from converging.
- Multi-condition subsets (FD002, FD004) show larger val→test gaps for both models.

LSTM is now supported as a sequence architecture alongside GRU (selected by
`sequence.architecture`), but no LSTM has been trained yet — LSTM rows are
**pending training** and a cross-model GRU-vs-LSTM benchmark is deferred. The
table above reports only the models that have actually been trained.

Full sweep analysis:
[Ridge feature sweep](docs/feature_sweep_ridge_report.md) ·
[Ridge vs GRU](docs/feature_sweep_ridge_vs_gru.md) ·
[GRU two-stage capacity sweep](docs/gru_capacity_sweep_report.md)
(the earlier fixed-capacity [GRU feature sweep](docs/archive/feature_sweep_gru_report.md) is archived)

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

Train a sequence model (GRU or LSTM). The architecture is read from
`sequence.architecture` (`gru` or `lstm`) in the config and built through a
shared RNN architecture registry, so one entrypoint trains either:

```bash
turbofan-train-sequence --config configs/subsets/fd001_gru.yaml    # GRU
turbofan-train-sequence --config configs/subsets/fd001_lstm.yaml   # LSTM
```

Each subset ships an explicit `fd00X_gru.yaml` and `fd00X_lstm.yaml` config (both
`_base_`-referencing the shared `fd00X.yaml`), so the architecture is always
selected by a named, version-controlled config rather than relying on the
default. The bare `fd00X.yaml` still loads (defaulting to GRU via
`sequence.architecture`), but prefer the explicit variant when training.

The `turbofan-train-sequence-gru` console script is retained as a
backward-compatible alias for the same entrypoint (it also honors
`sequence.architecture`, defaulting to GRU):

```bash
turbofan-train-sequence-gru --config configs/subsets/fd001_gru.yaml
```

Model training logs and registers the fitted model in the local MLflow store and
writes a timestamped run directory under `artifacts/models/<model_type>/<timestamp>/`
holding lightweight run records (config snapshot, metrics, training history, and
prediction CSVs). The MLflow registry — not the run directory — is the
authoritative model store: each production run auto-registers a new version of
`turbofan-<model_type>-<subset>` (e.g. `turbofan-gru-fd001`, `turbofan-lstm-fd001`)
linked to its run. LSTM models register under their own `turbofan-lstm-<subset>`
name with an independent `@production` alias, so GRU and LSTM versions are
promoted and served separately. Model bytes are no longer written to the run
directory.

Cross-run experiment summaries (sweep result CSVs) are written to `results/`. Run
metadata — params, metrics, and per-epoch GRU training curves — is logged to the
same local MLflow store (`mlflow.db`, SQLite) under two experiments,
`turbofan-training` (production runs) and `turbofan-sweeps` (hyperparameter
sweeps). Browse runs and registered models with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Promote a registered version to production (immediate, no approval gate), and
list registered models with their production alias and key metric:

```bash
turbofan-promote turbofan-gru-fd001 3 --to production   # roll back by promoting an earlier version
turbofan-models
```

The CLIs keep diagnostics separate from results: progress and lifecycle
diagnostics go to **stderr** via leveled `logging` (control verbosity with
`--log-level DEBUG|INFO|WARNING|ERROR`, or the `LOG_LEVEL` env var), while
genuine results (run directory, validation metrics) print to **stdout**. The two
production training CLIs additionally capture a per-run `run.log` and attach it
as an MLflow artifact under `logs/run.log`.

## Project Structure

```text
configs/
  default.yaml                        # Shared base configuration
  subsets/                            # Per-subset overrides (fd001–fd004)
data/                                 # Raw, interim, and processed dataset files
docs/                                 # Analysis reports and documentation
notebooks/                            # Exploratory analysis notebooks
results/                              # Cross-run experiment summary CSVs (sweep results)
mlflow.db                             # Local MLflow run + registry store (SQLite, git-ignored)
artifacts/                            # Per-run records: metrics, configs, histories, and prediction CSVs (model bytes live in MLflow; git-ignored)
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
| `turbofan-train-sequence` | Train a sequence model (GRU or LSTM), architecture from `sequence.architecture` in config |
| `turbofan-train-sequence-gru` | Backward-compatible alias for `turbofan-train-sequence` (also honors `sequence.architecture`, defaults to GRU) |
| `turbofan-predict` | Batch prediction and optional official-label evaluation, resolving the production model by name |
| `turbofan-serve-api` | FastAPI inference server |
| `turbofan-promote` | Promote a registered model version to an alias (e.g. `@production`); rollback by promoting an earlier version |
| `turbofan-models` | List registered models, versions, the `@production` alias, key metric, and run link |

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
turbofan-train-sequence --config configs/subsets/fd002_gru.yaml     # GRU
turbofan-train-sequence --config configs/subsets/fd002_lstm.yaml    # LSTM
```

Each subset ships dedicated `configs/subsets/fd00{1-4}_gru.yaml` and
`configs/subsets/fd00{1-4}_lstm.yaml` configs. These inherit everything from
their `fd00{1-4}.yaml` base via `_base_` composition and set only
`sequence.architecture`, so GRU and LSTM training for each subset are both
selected by an explicit, version-controlled config. (`_base_` references resolve
recursively, so an `_gru`/`_lstm` config extending a subset config that itself
extends `default.yaml` is fully composed.)

The `sensor_cols_to_drop` lists are derived from the EDA notebooks (`notebooks/eda_fd00{1-4}.ipynb`) and represent the authoritative drop decision for each subset.

The optional `features.ridge`, `features.gru`, and `features.lstm` blocks set per-model feature engineering: the top-level `feature_set`/`windows` are shared fallbacks, and each model's training CLI resolves its own block (inheriting the shared values for anything left unset). The committed Ridge and GRU values are each model's best configuration from the feature sweep; the `features.lstm` block is currently seeded from the GRU-tuned values (a dedicated LSTM feature sweep is a possible follow-up).

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
| LSTM | Recurrent sequence model trained on the same sliding windows; shares the GRU's hyperparameter surface and training/inference path, differing only in the recurrent cell |

The two recurrent architectures (GRU, LSTM) are interchangeable through a shared
RNN architecture registry: a single `SequenceRULRegressor` owns the encoder,
regression head, and packed-sequence path, and the recurrent layer is selected by
`sequence.architecture` (`gru` or `lstm`). They train through the same
`turbofan-train-sequence` entrypoint and serve through the same inference path.

Both architectures share one training-hyperparameter surface in `SequenceConfig`
(`window_size`, `hidden_size`, `num_layers`, `dropout`, `learning_rate`,
`weight_decay`, `batch_size`, `epochs`, `patience`). `weight_decay` (Adam L2,
default `0.0`) is the regularizer to reach for when overfitting on the smaller
subsets — note that inter-layer `dropout` only takes effect when
`num_layers > 1`, so at the default single layer `weight_decay` is the sole
explicit regularizer.

All models share the same preprocessing contract: a 4-step sklearn Pipeline (`SensorDropper → OperatingModeNormalizer → SensorColumnSelector → FeatureEngineer`). The `feature_set` config key selects which engineered features the models receive — `raw`, `rolling_mean`, `rolling_stats`, `raw_plus_rolling_mean`, `raw_plus_rolling_stats`, `lag`, or `raw_plus_lag`. Rolling and lag features are computed per engine without crossing engine boundaries; `lag` is a normalized lag-difference `(x[t] - x[t-N]) / rolling_mean(x, N)`. Feature settings can differ per model via the `features.ridge` / `features.gru` / `features.lstm` config blocks.

## Evaluation

Models are evaluated using:

| Metric | Purpose |
|---|---|
| RMSE | Penalizes larger prediction errors |
| MAE | Measures average absolute prediction error |
| PHM08 score | Asymmetric scoring function from the prognostics community — penalizes late predictions more heavily than early ones. Computed on the official test set only |

Validation and sweep ranking use RMSE and MAE; the PHM08 score is computed only
on the official test set. Sequence training (GRU or LSTM) reports RMSE and MAE on
an engine-level validation split. When the official test files for the configured subset are
present (e.g. `test_FD001.txt` with `RUL_FD001.txt`), training also evaluates
against them and reports RMSE, MAE, and the PHM08 score.

Per-run metrics and prediction CSVs are saved in the run directory under `artifacts/models/<model_type>/<timestamp>/`. Cross-run summaries are saved under `results/`.

## Inference and Serving

Batch prediction, FastAPI serving, and Docker serving resolve the production
model **by name** from the MLflow registry — there is no path-based artifact or
manifest to point at.

Run batch prediction against a registered model (resolves `@production` by
default):

```bash
turbofan-predict \
  --model turbofan-gru-fd001 \
  --input data.csv \
  --output predictions.csv \
  --metadata-output metadata.json
```

The same command serves an LSTM model by pointing `--model` at its
`turbofan-lstm-<subset>` registered name — LSTM models are served through the
identical predict/serving path. Pass `--alias <alias>` to resolve a
non-production alias, or pass an explicit `models:/<name>@<alias>` URI to
`--model`. When `--data-dir data/raw --subset
FD001` are provided and the matching `RUL_FD001.txt` labels align with the
prediction count, the CLI also reports RMSE, MAE, and PHM08 score and writes them
to the metadata JSON.

Start the FastAPI server locally (resolving `models:/<name>@production`):

```bash
turbofan-serve-api --model turbofan-gru-fd001 --host 127.0.0.1 --port 8000
```

The server exposes two endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Returns loaded model metadata and status |
| `/predict` | POST | Accepts engine sensor records and returns RUL predictions |

The served model is selected by `--model`/`--alias`, or by the
`TURBOFAN_MODEL_NAME` (and optional `TURBOFAN_MODEL_ALIAS`, default `production`)
environment variables. Point `MLFLOW_TRACKING_URI` at the registry store.

### Docker

A `docker-compose.yml` is included for containerized deployment. It mounts the
MLflow registry store (the SQLite db and its artifacts) into the container at
`/models/` and resolves the model by name.

Build and start the server:

```bash
docker compose up --build
```

By default this serves `turbofan-gru-fd001@production`. To point at a different
model or store:

```bash
TURBOFAN_MODEL_NAME=turbofan-ridge-fd001 \
MLFLOW_STORE_DIR=./mlflow_store docker compose up
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

## Reproducibility

The project is designed to make experiments reproducible through:

- Scripted data download
- Configuration-driven training
- Deterministic random seeds
- Versioned models in the MLflow registry, each linked to its run
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
- [x] Unified feature-engineering sweep with Ridge vs GRU analysis across all four subsets
- [x] Train and persist baseline and GRU production artifacts on FD002–FD004
- [x] Cross-dataset benchmark table from persisted models
- [x] Left-zero-pad short engines in GRU windowing (packed sequences; no engine skipped)
- [x] Two-stage GRU temporal-context and capacity sweep (sequence window vs. hidden-size/LR cross); short engines left-zero-padded
- [x] MLflow experiment tracking — local SQLite store; Ridge and GRU runs log params, metrics, and per-epoch curves; replaces JSONL audit log
- [x] Structured logging — leveled stdlib logging to stderr; `run.log` captured as MLflow artifact; results stay on stdout
- [x] Model registry — MLflow registry over local store; `turbofan-promote` and `turbofan-models` CLIs; name-based resolution replaces path/manifest layer
- [x] LSTM sequence model (shared RNN architecture registry; selected by `sequence.architecture`) — training real LSTM artifacts and the GRU-vs-LSTM benchmark deferred
- [ ] Transformer-based sequence model
- [ ] Advanced feature engineering

See [docs/roadmap.md](docs/roadmap.md) for detailed priorities and design decisions.


## License

MIT
