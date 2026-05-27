# Roadmap

Detailed plan for what's been built and what's next. The README has a summary checklist; this document has the reasoning and priorities.

## Completed

### Foundation (2026-05-23)

- Repository structure, Pydantic config schema, data download CLI, raw C-MAPSS loader, piecewise-linear RUL labels
- EDA notebook and reusable analysis utilities (quality, sensors, degradation)
- nbstripout git filter for clean notebook commits

### Feature Engineering (2026-05-23)

- sklearn pipeline: constant sensor removal, rolling statistics (configurable windows), operating-condition normalization
- Train-fitted normalizers to prevent data leakage — always fit on training rows only

### Baseline Modeling (2026-05-24)

- Ridge regression with engine-level validation split
- Alpha sweep and feature set comparison experiments (raw, rolling, engineered)
- SLURM job scripts for HPC execution

### Sequence Modeling (2026-05-24)

- GRU trained on fixed-length sliding windows with stored normalizer stats
- Windowing never crosses engine boundaries
- GRU hyperparameter sweep (window size, hidden size, learning rate)
- Feature selection sweep (GRU with different sensor subsets)
- Official C-MAPSS test set evaluation when test files are present

### Inference and Serving (2026-05-25)

- Batch prediction CLI with CSV/JSON input, strict and partial modes
- FastAPI server (`/health`, `/predict`) loading one artifact at startup
- Model manifest standard (`model_manifest.json`) abstracting Ridge vs GRU
- Dockerfile for containerized inference server

## Next — Fix Known Inference Bugs

Batch prediction and FastAPI serving have been validated end-to-end (2026-05-27). The following bugs were found:

- [ ] GRU inference predictor does not rescale predictions by `max_rul` — outputs are 0–1 instead of actual RUL cycles
- [ ] Ridge batch prediction returns one prediction per input row instead of one per engine (last cycle only)
- [ ] `turbofan-predict` CLI does not evaluate predictions against official test labels when available
- [ ] Sweep experiments evaluate on official test set — should be removed to avoid data leakage risk

Remaining validation:

- [x] Run `turbofan-predict` against real baseline and GRU artifacts
- [x] Run `turbofan-serve-api` with a real artifact, hit `/predict` with test data
- [ ] Build and run the Docker container with a mounted artifact

## Next — Multi-Dataset Support (FD002–FD004)

The config already accepts `fd_subset: FD002` etc., but no training runs or validation have been done beyond FD001.

- [ ] Train baseline and GRU on FD002, FD003, FD004
- [ ] Verify feature engineering handles multiple operating conditions (FD002/FD004 have 6)
- [ ] Verify sequence windowing handles multiple fault modes (FD003/FD004 have 2)
- [ ] Cross-dataset benchmark table comparing all subsets

## Future — Additional Models

The original plan included Random Forest, XGBoost, LSTM, and Transformer models. Current priority is depth on the existing models before breadth.

- [ ] LSTM (closest to GRU — reuses the sequence infrastructure)
- [ ] Transformer-based sequence model
- [ ] Tree-based baselines (XGBoost, Random Forest) if they add value to the comparison

## Future — Advanced Feature Engineering

Current features: constant sensor removal, rolling statistics, operating-condition normalization. The original plan listed several more:

- [ ] Lag features
- [ ] Degradation slope estimation
- [ ] Trend indicators
- [ ] Frequency-domain features (FFT/spectral)
- [ ] Composite health indicators

These are lower priority — the current feature set already supports meaningful model comparison.

## Future — MLOps Infrastructure

Deliberately deferred until the modeling contract stabilizes:

- [ ] Experiment tracking (MLflow or similar)
- [ ] CI/CD (GitHub Actions for lint, type-check, tests)
- [ ] Structured logging (replace print statements with proper logging)
- [ ] Model registry / formal versioning beyond timestamp directories

## Design Decisions Worth Preserving

**Prediction semantics should be one per engine.** Both Ridge and GRU should return one RUL prediction per engine (at the last observed cycle). The GRU predictor already does this via final-window inference. The Ridge predictor currently returns one prediction per input row — this is a known bug to fix.

**Inference schema contract.** Public input is always canonical raw C-MAPSS format (engine_id, cycle, op_1–3, s_1–21). Model-specific preprocessing is owned by the artifact. No pre-normalized or pre-featurized input is accepted.

**Short engines are skipped, not padded.** Engines shorter than the GRU window size are currently skipped during training. Padding/masking is deferred.

**Single-layer GRU by design.** Kept simple to avoid grid explosion. Multi-layer and alternative architectures (LSTM, TCN) are deferred.
