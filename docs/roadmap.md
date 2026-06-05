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
- GRU sweeps evaluate validation windows only; official test-set evaluation was
  removed from sweep selection to avoid leakage-prone model ranking

### Inference and Serving (2026-05-25)

- Batch prediction CLI with CSV/JSON input, strict and partial modes
- FastAPI server (`/health`, `/predict`) loading one artifact at startup
- Model manifest standard (`model_manifest.json`) abstracting Ridge vs GRU
- Dockerfile and docker-compose.yml for containerized inference server
- `scripts/query_api.py` for end-to-end API smoke testing against official test labels
- Ridge inference returns one last-cycle prediction per engine
- GRU inference rescales normalized model output by `max_rul`
- `turbofan-predict` can evaluate against official RUL labels when `--data-dir`
  and `--subset` are provided

## Recently Fixed — Inference Bugs (2026-05-27)

Batch prediction and FastAPI serving have been validated end-to-end (2026-05-27). The following bugs were fixed:

- [x] GRU inference predictor rescales predictions by `max_rul`
- [x] Ridge batch prediction returns one prediction per engine at the last cycle
- [x] `turbofan-predict` CLI evaluates predictions against official test labels when available and aligned
- [x] Sweep experiments no longer evaluate on the official test set

## Completed — Docker Serving Validation (2026-05-27)

- [x] Run `turbofan-predict` against real baseline and GRU artifacts
- [x] Run `turbofan-serve-api` with a real artifact, hit `/predict` with test data
- [x] Build and run the Docker container with a mounted artifact
- [x] Smoke-test containerized API with `scripts/query_api.py` — RMSE matches training run

## Completed — Operating-Mode Normalization (2026-05-28)

- [x] New `turbofan.preprocessing.normalization.OperatingModeNormalizer` — sklearn-compatible, fits per-mode KMeans clusters on training operating-setting rows, normalizes sensor features per mode
- [x] `n_modes` added to `FeatureConfig` (default `1`; set to `6` for FD002/FD004)
- [x] Baseline pipeline: replaced `OperationalNormalizer` with `OperatingModeNormalizer`; exact operating-setting tuple grouping eliminated
- [x] GRU training: replaced global `SequenceNormalizer` with `OperatingModeNormalizer`; normalizer state serialized into artifact as `normalizer_type: operating_mode` payload
- [x] GRU sweeps: updated to `OperatingModeNormalizer`
- [x] GRU inference: reconstructs normalizer from payload; legacy flat-stat checkpoints rejected with a clear retrain error
- [x] Stale `OperationalNormalizer` and `SequenceNormalizer` tests removed

## Completed — Config-Driven Sensor Dropping (2026-05-28)

- [x] `SensorDropper` rewritten: takes an explicit `drop: list[str]` from config; no statistics computed from training data; `check_is_fitted` guard added
- [x] `FeatureConfig.sensor_cols_to_drop` replaces `sensor_std_threshold` and `sensor_keep`
- [x] Rename propagated through pipeline factory, baseline builder, CLI, and experiments
- [x] `configs/default.yaml` updated; FD002 users set `sensor_cols_to_drop: [s_16, s_1, s_5, s_18, s_19]`

## Completed — EDA Pipeline and Subset Configs (2026-05-28)

- [x] Rebuilt EDA notebooks for all four subsets (`notebooks/eda_fd00{1-4}.ipynb`) with a coherent flow:
  - Single-condition subsets (FD001/FD003): constant sensor removal → correlation filter → low-variance check → distributions → degradation trajectories
  - Multi-condition subsets (FD002/FD004): same but operating-mode normalization runs before the correlation filter; before/after correlation and distribution plots included
- [x] Correlation-based sensor filter replaces the std-threshold approach: sensors are dropped if `|Pearson r with RUL| < 0.1` (post-normalization for FD002/FD004)
- [x] Low-variance check added after correlation filter to flag sensors like `s_16` (binary) that pass correlation but warrant caution in feature engineering
- [x] `_base_` config composition added to `load_config`: subset configs reference `default.yaml` and override only `fd_subset`, `sensor_cols_to_drop`, and `n_modes`
- [x] `configs/subsets/fd001–fd004.yaml` created with EDA-derived drop lists as the authoritative sensor drop decision per subset
- [x] `compute_coefficient_of_variation` removed from `quality.py` (CV did not help distinguish informative low-std sensors)
- [x] Tests added for `_base_` composition (deep merge, scalar replace, nested merge, list replace, key stripping)

## Completed — Unified Feature Pipeline (2026-05-29)

Ridge and GRU previously received fundamentally different data: Ridge used rolling statistics on a filtered sensor set; GRU used raw values on all 21 sensors with its own normalization path. This refactor makes them share a single preprocessing contract.

- [x] `FeatureEngineer` sklearn transformer - config-driven feature families (`raw`, `rolling_mean`, `lag`, `rolling_std`, `rolling_min`, `rolling_max`, `rolling_slope`, `rolling_delta`); families compose by concatenating output columns in config order
- [x] 5-step shared `build_feature_pipeline`: `SensorDropper → OperatingModeNormalizer → SensorColumnSelector → FeatureEngineer → StandardScaler`
- [x] `FeatureConfig` gains `feature_families`, `windows`, `lag_steps`; `ModelConfig` loses them (they are feature engineering parameters, not model parameters)
- [x] Ridge: simplified to 2-step pipeline (`build_feature_pipeline → Ridge`); old dead helpers removed (`RollingFeatureExtractor`, `_ModelFeatureSelector`, `_LowVarianceFeatureDropper`, `StandardScaler`, etc.)
- [x] GRU training and sweep: manual `OperatingModeNormalizer` + `default_feature_cols()` replaced by `build_feature_pipeline`; `feature_cols` derived from `pipeline.named_steps["feature_engineer"].feature_cols_`
- [x] Dead code deleted: `rolling.py`, `SequenceNormalizer`, `default_feature_cols`

**Design decisions:**

`windows` and `lag_steps` live in `FeatureConfig`, not `ModelConfig`, because they determine what features the model sees — not how the model is parameterized. Changing them without retraining produces an incompatible model.

`SensorColumnSelector` keeps `engine_id` as a pass-through so `FeatureEngineer` can do per-engine groupby for rolling and lag without it appearing in the final model input.

The `_AutoSensorNormalizer` subclass infers sensor columns from training data at fit time rather than requiring the caller to pass an explicit list. This avoids having to compute the "kept sensors" list at pipeline construction time (before `SensorDropper` has run).

Existing GRU checkpoints trained before this refactor are incompatible: `feature_cols` in the old payload included op cols and all 21 sensors. Those artifacts will fail to load with a retrain error.

## Completed — Unified Feature-Engineering Sweep (2026-05-29)

A single sweep harness now evaluates Ridge and GRU on identical feature inputs
through the shared `build_feature_pipeline`, across all four C-MAPSS subsets.

- [x] Unified feature-engineering sweep CLI — one entrypoint routing both Ridge and GRU through the same preprocessing contract; results carry a `model` column so separate runs stack and compare
- [x] `raw_plus_lag` feature set added as the lag-family companion to `raw_plus_rolling_mean`
- [x] `lag` semantics corrected to a normalized lag-difference `(x[t] - x[t-N]) / rolling_mean(x, N)` (previously returned the raw historical value `x[t-N]`)
- [x] Default output path `results/feature_sweep_{model}_{subset}.csv`; sweeps run for all four subsets across both models
- [x] Cross-model analysis reports grounded solely in the sweep data and EDA, with methodology citations: `docs/feature_sweep_ridge_report.md`, `docs/archive/feature_sweep_gru_report.md` (archived 2026-06-01), `docs/feature_sweep_ridge_vs_gru.md`
- [x] Removed the superseded `baseline_feature_comparison` job/results and stale pre-refactor reports
- [x] Per-model best-config feature engineering wired into the subset configs via `features.ridge` / `features.gru` blocks (resolved by `FeatureConfig.for_model`); each train CLI loads its own block

**Headline findings (engine-level validation split):**

- The GRU roughly halves best validation RMSE versus Ridge on the same features (e.g. FD001 10.9 vs 20.7).
- The optimal rolling-mean window is opposite by model — Ridge short (2–4), GRU long (~15) — because Ridge is memoryless while the GRU already models the sequence.
- Lag-difference features are counterproductive for both: additive-neutral for Ridge, and they prevent the GRU from converging (early-stop at epoch 1–3).

**Design decisions:**

- Sweep dimensions (`feature_families`, `windows`, `lag_steps`) are CLI arguments, not a second config file. The subset configs still own dataset-specific parameters (`sensor_cols_to_drop`, `n_modes`, `max_rul`).
- The bespoke `top_corr` / `top_corr_rolling` feature sets from the old GRU sweep were dropped; sensor selection lives at EDA time via `sensor_cols_to_drop`.

## Completed — Multi-Dataset Training

EDA, per-subset configuration, and the cross-dataset feature-engineering sweep are
complete. The remaining step is production training and a final benchmark.

- [x] Train and persist baseline and GRU production artifacts on FD002, FD003, FD004 using `configs/subsets/`
- [x] Verify feature engineering handles 6 operating conditions (FD002/FD004) — confirmed by the feature sweep running cleanly on all four subsets
- [x] Cross-dataset benchmark table from persisted production models (validation-split feature comparison already in `docs/feature_sweep_*`)

## Completed — GRU Temporal-Context and Capacity Sweep (2026-05-31)

Two-stage sweep harness that disentangles how much temporal context the GRU needs
(sequence window size and rolling features) from how much model capacity it needs
(hidden size and learning rate). Short engines are now padded rather than skipped so
no engine is dropped from any window size.

- [x] Left-zero-pad short engines: `_build_windows` pads engines with fewer cycles than `window_size` (instead of skipping them) and records a `padded` flag and per-window `lengths`; `SequenceDataset`/`build_sequence_loader` yield 3-tuples; `GRURULRegressor.forward` accepts optional `lengths` and uses `pack_padded_sequence`; training, eval, and predict loops thread lengths through; `GRUPredictor` pads in `allow_partial` mode (strict mode unchanged); GRU sweep rows record `n_engines_total`, `n_engines_padded`, `n_engines_full`
- [x] SLURM retrain driver: `jobs/slurm/run_gru_selected_retrain.sh`

The post-run analysis report and any selected-config updates are deferred until the
cluster sweeps complete.

## Completed — Model Registry (2026-06-02)

Formal model versioning and promotion via MLflow's Model Registry on the local
SQLite store, replacing path-based resolution of timestamped artifact
directories. The MLflow artifact store is now the authoritative home for model
bytes.

- [x] `turbofan.registry` seam over MLflow's registry: `model_name`,
  `log_and_register`, `promote`, `resolve_uri`, `load`, `list_registered`, and
  `RegisteredModelInfo`
- [x] Ridge and GRU packaged as `mlflow.pyfunc` wrappers that reuse the existing
  inference compute (`ridge_engine_predictions`, `gru_final_window_predictions`);
  the wrappers return an `engine_id`/`cycle`/`prediction` frame so callers
  reconstruct the per-row prediction contract through the pyfunc boundary
- [x] Production training auto-registers a new version of
  `turbofan-<model_type>-<subset>` linked to its run; promotion is manual
- [x] `turbofan-promote` (alias repointing / rollback) and `turbofan-models`
  (listing with `@production`, `val_rmse`, run link) CLIs
- [x] `turbofan-predict` resolves `--model <name>` (`--alias`, default
  `production`) or an explicit `models:/<name>@<alias>` URI; the serving API
  resolves `models:/<name>@production` from `TURBOFAN_MODEL_NAME` / `--model`

**Retired (full retirement, by user decision):** `inference/manifest.py` and the
path-based `load_predictor` resolution; writing `model_manifest.json`; the
`artifacts/models/<ts>/` directory as the model home. Training run directories
are kept only for lightweight run records (metrics, config, prediction CSVs);
model bytes (`model.joblib`/`model.pt`) and manifests are no longer written
there. The Dockerfile/compose now mount the MLflow store and resolve by name
instead of mounting a run directory.

**Contract change:** the pyfunc boundary validates strictly, so the
`--allow-partial` per-row-skipping warnings are no longer surfaced; the flag is
accepted for CLI compatibility but does not change validation.

## Additional Models — LSTM done, others future

The original plan included Random Forest, XGBoost, LSTM, and Transformer models. Current priority is depth on the existing models before breadth. LSTM is now supported; the rest remain future work.

- [x] LSTM (2026-06-03) — added through a shared RNN architecture registry
  (`SEQUENCE_ARCHITECTURES` / `build_sequence_model` in
  `src/turbofan/models/sequence_models.py`) and a single `SequenceRULRegressor`
  that owns the encoder, regression head, and packed-sequence path; the
  recurrent layer (`nn.GRU` / `nn.LSTM`) is selected by `sequence.architecture`.
  GRU and LSTM share the training (`train_sequence_model`), inference
  (`sequence_final_window_predictions`), and registry pyfunc
  (`SequenceFinalWindowModel`) paths; LSTM registers under its own
  `turbofan-lstm-<subset>` name with an independent `@production` alias. New
  `turbofan-train-sequence` CLI reads the architecture from config;
  `turbofan-train-sequence-gru` is retained as a backward-compatible alias.
  Per-subset `configs/subsets/fd00{1-4}_lstm.yaml` configs and a `features.lstm`
  override block are committed. **Deferred:** training real LSTM artifacts and a
  cross-model GRU-vs-LSTM benchmark report — no LSTM has been trained yet, and
  the `features.lstm` blocks are seeded from the GRU-tuned values.
- [ ] Transformer-based sequence model
- [ ] Tree-based baselines (XGBoost, Random Forest) if they add value to the comparison

## Future — Advanced Feature Engineering

Current features: constant sensor removal, rolling statistics, lag features, operating-condition normalization — all config-driven via `feature_families`. The original plan listed several more exotic variants:

- [ ] Degradation slope estimation
- [ ] Trend indicators
- [ ] Frequency-domain features (FFT/spectral)
- [ ] Composite health indicators

These are lower priority — the current feature set already supports meaningful model comparison.

## Future — MLOps Infrastructure

Deliberately deferred until the modeling contract stabilizes:

- [x] Experiment tracking — MLflow with a local SQLite store (`mlflow.db`); Ridge
  and GRU runs log under the `turbofan-training` and `turbofan-sweeps`
  experiments. Replaced the GRU-only `results/training_log.jsonl` audit log.
- [x] CI/CD (GitHub Actions for lint, type-check, tests)
- [x] Structured logging — leveled stdlib `logging` to stderr across the four
  surviving entrypoints (`--log-level`, `LOG_LEVEL` env fallback); genuine
  results stay on stdout. The two production training CLIs capture a per-run
  `run.log` attached as an MLflow artifact under `logs/`.
- [x] Model registry / formal versioning beyond timestamp directories (see
  "Completed — Model Registry" below)
- [ ] Speed up the test suite (~3 min). The subprocess-based CLI/sweep tests
  cold-start a fresh interpreter and re-import torch + mlflow on every run, which
  dominates wall time; `import mlflow` and per-test SQLite setup added overhead in
  the tracking migration. Options: `pytest-xdist` (`-n auto`) since these tests
  parallelize well, mark the subprocess integration tests `slow` and run a fast
  in-process subset by default, or convert subprocess CLI tests to in-process
  `main()` calls.

## Design Decisions Worth Preserving

**Operating-mode normalization is config-driven, not auto-derived.** `n_modes` lives in `FeatureConfig` (default `1`). Setting it to `6` in config enables KMeans-based per-mode normalization for FD002/FD004. The original approach of auto-deriving from `fd_subset` via a lookup table was removed in favor of explicit config so the user can override the count when needed.

**GRU artifacts are self-contained after the normalization migration.** The checkpoint stores `normalizer_type: operating_mode` and a `normalizer_payload` dict; inference reconstructs the normalizer from it without any runtime config. Legacy checkpoints (flat `normalizer_means`/`normalizer_stds`) are rejected with a retrain error. After the model-registry step the checkpoint payload is carried inside the MLflow GRU pyfunc model rather than a run-dir `model.pt`.



**Sensor dropping is config-driven, not threshold-based.** `sensor_cols_to_drop` in `FeatureConfig` holds the EDA-derived explicit drop list. `SensorDropper` applies it without recomputation — no runtime statistics. A single threshold cannot distinguish pre-normalization low-variance sensors (e.g. `s_16`) from post-normalization zero-variance sensors (e.g. `s_1`, `s_5`, `s_18`, `s_19` on FD002), and recomputing at fit time undermines reproducibility.

**EDA notebooks are the source of truth for sensor drop decisions.** `configs/subsets/fd00{1-4}.yaml` codify those decisions. The notebooks explain the reasoning; the configs enforce it at training time. Changing which sensors to drop means updating both.

**Config composition via `_base_`.** Subset configs reference `default.yaml` via `_base_` and override only what differs. Nested dicts are deep-merged key-by-key; lists and scalars are replaced. This keeps a single source of truth for shared settings (model hyperparameters, paths, seeds) while allowing per-subset variation in `fd_subset`, `sensor_cols_to_drop`, and `n_modes`.

**Prediction semantics should be one per engine.** Ridge scores all valid input rows and returns the last-cycle prediction per engine. GRU returns one final-window prediction per eligible engine.

**Inference schema contract.** Public input is always canonical raw C-MAPSS format (engine_id, cycle, op_1–3, s_1–21). Model-specific preprocessing is owned by the artifact. No pre-normalized or pre-featurized input is accepted.

**Short engines are left-zero-padded and processed with `pack_padded_sequence`.** The GRU pipeline pads engines shorter than the configured `window_size` and runs them as packed sequences so the final hidden state reflects only real timesteps.

**Single-layer recurrent models by design.** Kept simple to avoid grid explosion. LSTM is now supported alongside GRU through the RNN architecture registry (`SEQUENCE_ARCHITECTURES` / `build_sequence_model`), selected by `sequence.architecture`. The registry contract is deliberately RNN-scoped — it will be widened only when a non-RNN architecture (the planned Transformer) lands with real requirements in hand. Multi-layer stacks and non-RNN architectures (TCN, Transformer) remain deferred.
