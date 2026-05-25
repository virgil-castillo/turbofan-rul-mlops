# Inference Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build local-first batch and API inference for Ridge and GRU turbofan RUL artifacts.

**Architecture:** Add `turbofan.inference` as the shared runtime layer, with schema validation, manifest loading, model-specific predictors, and a FastAPI app factory. Batch and API entry points call this shared package so model loading and prediction semantics are tested once.

**Tech Stack:** Python 3.12, pandas, numpy, joblib, torch, scikit-learn, FastAPI, pydantic, pytest, ruff, mypy.

---

### Task 1: Schema and Manifest Core

**Files:**
- Create: `src/turbofan/inference/__init__.py`
- Create: `src/turbofan/inference/schemas.py`
- Create: `src/turbofan/inference/manifest.py`
- Test: `tests/inference/test_schemas.py`
- Test: `tests/inference/test_manifest.py`

- [ ] **Step 1: Write failing schema tests**

Cover canonical required columns, empty input, non-positive identifiers, non-numeric or non-finite feature values, duplicate `(engine_id, cycle)`, extra-column dropping, sorting, and partial row skipping.

- [ ] **Step 2: Write failing manifest tests**

Cover valid Ridge/GRU manifests, relative path resolution, compatibility loading from `model.joblib` and `model.pt`, and clear errors for ambiguous or missing artifacts.

- [ ] **Step 3: Run failing tests**

Run: `pytest tests/inference/test_schemas.py tests/inference/test_manifest.py -q`
Expected: FAIL because `turbofan.inference` does not exist.

- [ ] **Step 4: Implement schema and manifest modules**

Use dataclasses for internal runtime contracts and pydantic/dataclass-friendly dictionaries for JSON output. Validate canonical raw records only, ignore extra columns after validation, and return warnings for skipped rows in partial mode.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/inference/test_schemas.py tests/inference/test_manifest.py -q`
Expected: PASS.

### Task 2: Predictor Runtime

**Files:**
- Create: `src/turbofan/inference/predictors.py`
- Test: `tests/inference/test_predictors.py`

- [ ] **Step 1: Write failing predictor tests**

Use synthetic joblib and torch artifacts. Cover Ridge one-row-per-input prediction, GRU one-final-window-per-engine prediction, short-engine strict failure, short-engine partial skip with warnings, non-negative clipping, and output metadata fields.

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/inference/test_predictors.py -q`
Expected: FAIL because predictors do not exist.

- [ ] **Step 3: Implement predictors**

Load Ridge pipelines from joblib. Load GRU checkpoint payloads, rebuild `GRURULRegressor`, reconstruct `SequenceNormalizer`, build final windows, run CPU inference, clip predictions, and return `PredictionResult`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/inference/test_predictors.py -q`
Expected: PASS.

### Task 3: Batch CLI and API

**Files:**
- Create: `scripts/predict.py`
- Create: `scripts/serve_api.py`
- Create: `src/turbofan/inference/service.py`
- Test: `tests/inference/test_service.py`
- Test: `tests/inference/test_predict_cli.py`

- [ ] **Step 1: Write failing CLI and API tests**

Cover CSV and JSON batch input, `--allow-partial`, invalid artifact failure, `/health`, valid `/predict`, 422 validation errors, and partial mode behavior.

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/inference/test_predict_cli.py tests/inference/test_service.py -q`
Expected: FAIL because scripts and service do not exist.

- [ ] **Step 3: Implement CLI and API**

The CLI loads an artifact, reads CSV or JSON records, writes predictions CSV and metadata JSON, and prints a compact summary. The service factory accepts an artifact path or predictor, exposes `/health` and `/predict`, maps validation errors to 422, and leaves unexpected failures as 500.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/inference/test_predict_cli.py tests/inference/test_service.py -q`
Expected: PASS.

### Task 4: Training Manifests, Config, Docker

**Files:**
- Modify: `scripts/train_baseline.py`
- Modify: `scripts/train_sequence_gru.py`
- Modify: `src/turbofan/config/schema.py`
- Modify: `configs/default.yaml`
- Modify: `pyproject.toml`
- Create: `Dockerfile`
- Test: `tests/models/test_train_baseline_cli.py`
- Test: `tests/models/test_train_sequence_gru_cli.py`

- [ ] **Step 1: Write failing manifest assertions**

Update training CLI tests to require `model_manifest.json` with `schema_version`, `model_type`, `artifact_id`, `prediction_scope`, and relative artifact paths.

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/models/test_train_baseline_cli.py tests/models/test_train_sequence_gru_cli.py -q`
Expected: FAIL because training scripts do not write manifests.

- [ ] **Step 3: Implement manifest writes and configuration**

Add a reusable manifest save helper, add optional inference config fields, include FastAPI/uvicorn dependencies, add serving script, and create a Dockerfile that runs uvicorn against `scripts/serve_api.py`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/models/test_train_baseline_cli.py tests/models/test_train_sequence_gru_cli.py -q`
Expected: PASS.

### Task 5: Final Verification

**Files:**
- All changed files above.

- [ ] **Step 1: Run inference tests**

Run: `pytest tests/inference tests/models/test_train_baseline_cli.py tests/models/test_train_sequence_gru_cli.py -q`
Expected: PASS.

- [ ] **Step 2: Run full quality gates**

Run: `ruff check src/ tests/ scripts/`
Expected: PASS.

Run: `mypy src/turbofan`
Expected: PASS.

Run: `pytest`
Expected: PASS.
