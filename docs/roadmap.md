# Roadmap

This file is a chronological map of repository progress. Use it to understand
what changed, what contracts now matter, and what work remains.

## Completed Work

### Foundation (2026-05-23)

- Created the repository structure, config schema, data download CLI, raw C-MAPSS
  loader, and piecewise-linear RUL labels.
- Added the EDA notebook and reusable analysis utilities.
- Added nbstripout for cleaner notebook commits.

### Feature Engineering (2026-05-23)

- Added an sklearn feature pipeline with sensor dropping, rolling statistics, and
  operating-condition normalization.
- Made normalization train-fitted to avoid validation and test leakage.

### Baseline Modeling (2026-05-24)

- Added Ridge regression with an engine-level validation split.
- Added alpha and feature-set comparison experiments.
- Added SLURM scripts for HPC execution.

### Sequence Modeling (2026-05-24)

- Added GRU training on fixed-length per-engine windows.
- Ensured windowing never crosses engine boundaries.
- Added GRU hyperparameter and sensor-subset sweeps.
- Removed official test-set evaluation from sweep selection.

### Inference and Serving (2026-05-25)

- Added batch prediction and FastAPI serving.
- Added a model manifest contract for Ridge and GRU artifacts.
- Added Docker and docker-compose support for the inference server.
- Added API smoke testing with `scripts/query_api.py`.
- Aligned prediction semantics to one final prediction per engine.

### Inference Fixes (2026-05-27)

- Fixed GRU inference to rescale normalized predictions by `max_rul`.
- Fixed Ridge inference to return the last-cycle prediction per engine.
- Added official-label evaluation to `turbofan-predict` when labels are present.
- Stopped sweep experiments from ranking models on official test labels.

### Docker Serving Validation (2026-05-27)

- Validated batch prediction against real Ridge and GRU artifacts.
- Validated FastAPI serving with real artifacts and test data.
- Validated containerized serving with a mounted artifact.

### Operating-Mode Normalization (2026-05-28)

- Replaced exact operating-setting grouping with `OperatingModeNormalizer`.
- Added config-driven `n_modes` support for single-mode and multi-mode subsets.
- Migrated Ridge, GRU training, sweeps, and inference to the new normalizer.
- Removed stale normalizer implementations and tests.

### Config-Driven Sensor Dropping (2026-05-28)

- Replaced threshold-based sensor dropping with explicit config lists.
- Renamed config fields and propagated the new contract through pipelines, CLIs,
  and experiments.
- Updated default and subset configs with EDA-derived drop lists.

### EDA Pipeline and Subset Configs (2026-05-28)

- Rebuilt EDA notebooks for FD001 through FD004.
- Added correlation-based sensor filtering and low-variance checks.
- Added `_base_` config composition.
- Made subset configs the training-time source of truth for sensor drops and
  operating-mode counts.

### Unified Feature Pipeline (2026-05-29)

- Added `FeatureEngineer` for config-driven raw, rolling, lag, and related feature
  families.
- Unified Ridge and GRU preprocessing through `build_feature_pipeline`.
- Moved feature parameters into `FeatureConfig`.
- Removed dead pre-refactor feature and normalizer code.

### Unified Feature-Engineering Sweep (2026-05-29)

- Added one sweep harness for Ridge and GRU using the shared feature pipeline.
- Added comparable sweep outputs across all four C-MAPSS subsets.
- Added cross-model analysis reports grounded in sweep results and EDA.
- Wired best known per-model feature settings into subset configs.

### Multi-Dataset Training

- Trained and persisted production Ridge and GRU artifacts for FD002, FD003, and
  FD004.
- Confirmed feature engineering handles six operating modes for FD002 and FD004.
- Added cross-dataset benchmark coverage from persisted production models.

### GRU Temporal-Context and Capacity Sweep (2026-05-31)

- Added a two-stage GRU sweep for sequence window size, rolling context, hidden
  size, and learning rate.
- Added left-zero-padding for engines shorter than the configured window.
- Added packed-sequence handling across training, evaluation, and inference.
- Added the selected-retrain SLURM driver.

### Model Registry (2026-06-02)

- Added MLflow Model Registry integration through `turbofan.registry`.
- Packaged Ridge and sequence models as MLflow pyfunc models.
- Added registration during production training and manual promotion by alias.
- Added `turbofan-promote`, `turbofan-models`, registry-based prediction, and
  registry-based serving.
- Retired path-based model manifests as the source of model resolution.

### LSTM Support (2026-06-03)

- Added LSTM through the shared sequence architecture registry.
- Unified GRU and LSTM training, inference, and registry pyfunc paths.
- Added the `turbofan-train-sequence` CLI and retained the GRU-specific alias.
- Added per-subset LSTM configs seeded from GRU-tuned feature settings.

### MLOps Infrastructure

- Added MLflow experiment tracking with a local SQLite store.
- Added CI for lint, type-check, and tests.
- Added structured logging across the main entrypoints.
- Attached training logs as MLflow artifacts.

## Current Repository Contracts

- Public prediction input is raw C-MAPSS format: `engine_id`, `cycle`, `op_1`
  through `op_3`, and `s_1` through `s_21`.
- Artifacts own model-specific preprocessing. Do not pass pre-normalized or
  pre-featurized data into public inference paths.
- Ridge and sequence inference return one final prediction per engine.
- Sensor dropping is explicit in config and derived from EDA, not recomputed at
  fit time.
- `n_modes` is explicit in config. FD002 and FD004 use multi-mode normalization.
- Subset configs inherit from `configs/default.yaml` through `_base_`.
- Short sequence inputs are left-zero-padded and processed with packed sequences.
- MLflow Model Registry is the source of model resolution for production serving.

## Pending Work

- [ ] Add Transformer-based time-series RUL modeling with temporal/sensor
  attention.
- [ ] Add tabular model families: XGBoost and Random Forest.
- [ ] Add convolutional sequence models: 1D CNN and TCN.
- [ ] Run feature follow-ups: stack the winning window-20 families
  (`rolling_mean`, `rolling_slope`, `rolling_delta`) and test smoothing or
  learned health-index proxy features.
- [ ] Rerun the Ridge feature sweep before changing Ridge configs; existing
  Ridge sweep outputs are stale.
- [ ] Speed up the test suite, especially subprocess-heavy CLI and sweep tests.
