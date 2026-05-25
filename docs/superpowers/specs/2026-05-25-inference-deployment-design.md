# Inference & Deployment: Sub-project 6 Design Spec

**Date:** 2026-05-25
**Sub-project:** 6 of 6
**Depends on:** Sub-project 1 (Foundation), Sub-project 4 (Model Training),
Sub-project 5 (Sequence Models)
**Status:** Draft

---

## Goal

Build a production-shaped but local-first inference layer for turbofan RUL
prediction. The milestone supports both trained model families already present
in the repository:

- Ridge baseline artifacts from `scripts/train_baseline.py`
- GRU sequence artifacts from `scripts/train_sequence_gru.py`

The deliverable includes a shared Python inference core, a batch prediction
CLI, a FastAPI serving app, a Dockerfile for local container serving, and
synthetic tests for schema validation, model loading, prediction behavior, and
API endpoints.

The public input contract is raw canonical C-MAPSS runtime records:
`engine_id`, `cycle`, `op_1`, `op_2`, `op_3`, and `s_1` through `s_21`.
Callers do not provide `rul`, engineered features, or pre-normalized values.
Model-specific preprocessing remains owned by the saved artifact and inference
adapter.

## Architecture

Add a new `turbofan.inference` package with one model-agnostic runtime
interface and two model-specific adapters. Batch inference and API serving both
use this package instead of duplicating model logic.

```text
src/turbofan/inference/
|-- __init__.py
|-- schemas.py        # canonical input/output dataclasses and validation
|-- manifest.py       # model_manifest.json parsing and compatibility detection
|-- predictors.py     # Predictor protocol, RidgePredictor, GRUPredictor
`-- service.py        # FastAPI app factory
```

The shared predictor interface is intentionally small:

```python
class Predictor(Protocol):
    """Runtime prediction interface for fitted RUL models."""

    @property
    def metadata(self) -> ModelMetadata:
        """Return loaded model metadata."""

    def predict(
        self,
        records: pd.DataFrame,
        *,
        allow_partial: bool = False,
    ) -> PredictionResult:
        """Predict RUL values from canonical raw records."""
```

Ridge and GRU keep separate internals because they have different inference
semantics:

- Ridge predicts one RUL value per valid input row.
- GRU predicts one RUL value per eligible engine final window.

Both adapters return the same `PredictionResult` structure so the CLI and API
can stay model-agnostic.

## Artifact Contract

The preferred entry point is a manifest file named `model_manifest.json` in a
run directory. The manifest declares the model type, artifact paths, schema
version, prediction scope, and runtime metadata.

```json
{
  "schema_version": 1,
  "model_type": "ridge",
  "artifact_id": "baseline/20260525-120000",
  "prediction_scope": "row",
  "model_path": "model.joblib",
  "config_path": "config.json",
  "metrics_path": "metrics.json"
}
```

GRU manifests use `model_type="gru"` and `prediction_scope="final_window"`.
The GRU checkpoint remains a torch payload containing the state dict, sequence
config, feature column order, normalizer means, and normalizer standard
deviations.

Inference must also support compatibility loading for existing local run
directories that do not yet contain a manifest:

- A directory with `model.joblib` loads as a Ridge run.
- A directory with `model.pt` loads as a GRU run.
- If both or neither are present, loading fails with a clear `ValueError`.

Training scripts should be updated to write `model_manifest.json` for new
runs. Existing model binaries remain unchanged.

## Input Schema

The canonical runtime schema contains these required columns:

- `engine_id`
- `cycle`
- `op_1`, `op_2`, `op_3`
- `s_1` through `s_21`

Validation rules:

- Input must contain at least one record.
- Required columns must be present.
- Extra columns are ignored unless explicitly preserved in future versions.
- `engine_id` and `cycle` must be positive integers.
- Feature columns must be numeric and finite.
- Duplicate `(engine_id, cycle)` pairs are invalid.
- Rows are sorted by `engine_id` and `cycle` before prediction.

The batch CLI accepts CSV and JSON files containing canonical records. The API
accepts JSON records only. Raw NASA whitespace files are not the serving
contract for this milestone; callers can convert them through existing loaders
or a later utility if needed.

## Output Schema

Prediction rows use one operational contract for both model families:

- `engine_id`
- `cycle`
- `prediction`
- `model_type`
- `artifact_id`
- `prediction_scope`
- `predicted_at`

`prediction` is clipped to non-negative values, matching the training
evaluation behavior. It is not clipped to `max_rul`; values above the training
label cap remain visible as diagnostic output.

Response-level metadata contains:

- `model_type`
- `artifact_id`
- `prediction_scope`
- `input_rows`
- `prediction_rows`
- `warnings`

Batch mode writes predictions to CSV and metadata to JSON. API mode returns
both predictions and metadata in one JSON response.

## Validation Behavior

Strict mode is the default. In strict mode, any schema error, bad value,
duplicate key, empty input, or insufficient GRU history fails the whole request.

Partial mode is opt-in through `allow_partial=True`. In partial mode:

- Invalid rows are skipped when a row-level validation issue can be isolated.
- GRU engines shorter than `window_size` are skipped.
- Warnings describe each skipped row or engine.
- If no predictions remain after skipping invalid inputs, the request fails.

Partial mode does not silently impute missing feature values or coerce invalid
non-numeric strings. It only allows scoring of valid records that can be
unambiguously separated from invalid records.

## Ridge Inference Contract

`RidgePredictor` loads the fitted sklearn pipeline from `model.joblib`.

Prediction flow:

1. Validate canonical records.
2. Drop ignored extra columns.
3. Pass raw rows to the fitted sklearn pipeline.
4. Clip predictions to non-negative values.
5. Return one prediction row per valid input row.

The saved Ridge pipeline already contains feature engineering, identifier
dropping, normalization, and the fitted estimator. Inference does not rebuild
or re-fit any feature pipeline components.

## GRU Inference Contract

`GRUPredictor` loads `model.pt`, rebuilds `GRURULRegressor`, restores weights,
and reconstructs `SequenceNormalizer` from checkpoint statistics.

Prediction flow:

1. Validate canonical records.
2. Sort by `engine_id` and `cycle`.
3. Apply stored train-fitted sequence normalization.
4. Build one final fixed-length window per eligible engine.
5. Run the GRU in evaluation mode.
6. Clip predictions to non-negative values.
7. Return one prediction row per eligible engine at that engine's final cycle.

Engines shorter than the checkpoint `window_size` fail strict mode or are
skipped in partial mode. Padding and masking remain out of scope.

## Batch CLI

Add a batch scoring script:

```bash
python scripts/predict.py \
  --artifact artifacts/models/baseline/20260525-120000 \
  --input data/to_score.csv \
  --output artifacts/predictions/predictions.csv \
  --metadata-output artifacts/predictions/metadata.json
```

CLI responsibilities:

1. Load artifact from `model_manifest.json` or compatibility directory
   detection.
2. Read canonical records from CSV or JSON.
3. Run strict prediction by default.
4. Support `--allow-partial`.
5. Write prediction CSV.
6. Write metadata JSON.
7. Print artifact ID, model type, prediction count, and output paths.

The CLI does not train models, download data, or accept raw NASA whitespace
files directly.

## FastAPI Service

Add a FastAPI app factory under `turbofan.inference.service`.

Endpoints:

- `GET /health`
- `POST /predict`

`GET /health` returns service status and loaded model metadata.

`POST /predict` accepts:

```json
{
  "records": [
    {
      "engine_id": 1,
      "cycle": 30,
      "op_1": 0.0,
      "op_2": 0.0,
      "op_3": 100.0,
      "s_1": 518.67
    }
  ],
  "allow_partial": false
}
```

The example omits most sensors for readability only; real requests must include
all required `s_1` through `s_21` fields.

The app loads one model artifact at startup. The artifact path is supplied by
environment variable, for example `TURBOFAN_MODEL_ARTIFACT`, or through a small
serving entry point argument if the implementation chooses a CLI wrapper.

API error behavior:

- Validation errors return HTTP 422 with clear field-level details when
  possible.
- Missing or invalid artifacts fail application startup.
- Runtime prediction errors return HTTP 500 only for unexpected failures.

## Docker Scope

Add a Dockerfile for local container serving. The image installs the package,
exposes the FastAPI app through uvicorn, and expects the model artifact to be
mounted at runtime.

Example usage:

```bash
docker build -t turbofan-rul-api .
docker run --rm -p 8000:8000 \
  -e TURBOFAN_MODEL_ARTIFACT=/models/run \
  -v "$(pwd)/artifacts/models/baseline/20260525-120000:/models/run:ro" \
  turbofan-rul-api
```

The Dockerfile is not expected to contain a model artifact. Container registry
publishing, cloud deployment, Kubernetes manifests, and CI image publishing are
out of scope.

## Configuration

Add an optional inference config section to `configs/default.yaml`:

```yaml
inference:
  artifact_path: null
  host: 0.0.0.0
  port: 8000
  allow_partial: false
```

This config supports local serving defaults without changing training config.
The batch CLI can operate entirely from command-line paths and does not require
the config file.

## File Structure

```text
turbofan-rul-mlops/
|-- Dockerfile
|-- configs/
|   `-- default.yaml
|-- scripts/
|   |-- predict.py
|   |-- serve_api.py
|   |-- train_baseline.py
|   `-- train_sequence_gru.py
|-- src/turbofan/
|   |-- config/
|   |   `-- schema.py
|   `-- inference/
|       |-- __init__.py
|       |-- manifest.py
|       |-- predictors.py
|       |-- schemas.py
|       `-- service.py
`-- tests/
    |-- inference/
    |   |-- __init__.py
    |   |-- test_manifest.py
    |   |-- test_predictors.py
    |   |-- test_schemas.py
    |   `-- test_service.py
    `-- models/
        |-- test_train_baseline_cli.py
        `-- test_train_sequence_gru_cli.py
```

Training CLI tests should be updated to assert that new runs write
`model_manifest.json`.

## Testing Strategy

All tests use synthetic data and temporary directories. No NASA data download,
network access, or real model artifact is required.

**Schema tests**

- Required canonical columns are enforced.
- Extra columns are ignored.
- Empty input fails.
- Non-positive `engine_id` and `cycle` fail.
- Non-numeric or non-finite features fail.
- Duplicate `(engine_id, cycle)` pairs fail.
- Partial mode skips isolated invalid rows and records warnings.

**Manifest tests**

- Valid Ridge and GRU manifests load.
- Relative artifact paths resolve from the manifest directory.
- Existing `model.joblib` run directories load through compatibility detection.
- Existing `model.pt` run directories load through compatibility detection.
- Ambiguous or missing artifacts raise clear errors.

**Predictor tests**

- Ridge predictor returns one prediction per input row.
- GRU predictor returns one final-window prediction per eligible engine.
- GRU strict mode fails when an engine is shorter than `window_size`.
- GRU partial mode skips short engines with warnings.
- Predictions are clipped to non-negative values.
- Output rows include operational fields and stable metadata.

**Batch CLI tests**

- CSV input produces prediction CSV and metadata JSON.
- JSON input produces prediction CSV and metadata JSON.
- `--allow-partial` returns valid predictions plus warnings.
- Missing input or invalid artifact path exits non-zero with a useful message.

**API tests**

- `/health` returns loaded model metadata.
- `/predict` returns predictions for valid canonical records.
- `/predict` returns 422 for schema errors.
- `/predict` honors `allow_partial`.

**Docker verification**

- Build command is documented.
- Automated tests do not need to build Docker unless a lightweight smoke check
  is practical in the local environment.

## What This Sub-project Does NOT Include

- Cloud deployment, Kubernetes, Terraform, or container registry publishing
- MLflow, Weights & Biases, or a remote model registry
- Authentication, authorization, rate limiting, or multi-tenant serving
- Streaming inference
- Online feature stores
- Batch schedulers or orchestration tools
- Padding or masking for short GRU engine histories
- Serving multiple artifacts from one API process
- Automatic retraining or model promotion
- Real C-MAPSS benchmark claims in docs
- Accepting raw NASA whitespace files as the public prediction API contract

## Sub-project Sequence

| # | Sub-project | Depends on |
|---|-------------|------------|
| 1 | Foundation | -- |
| 2 | EDA | 1 |
| 3 | Feature Engineering | 1 |
| 4 | Model Training | 1, 3 |
| 5 | Sequence Models | 1, 3, 4 |
| 6 | **Inference & Deployment** <- you are here | 1, 4, 5 |
