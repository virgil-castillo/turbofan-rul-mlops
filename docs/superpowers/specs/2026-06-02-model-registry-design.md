# Model Registry — Design

**Date:** 2026-06-02
**Stage:** MLOps Infrastructure (step 4 of 4: model registry)
**Status:** Approved design, pending implementation plan

## Goal

Introduce formal model versioning and promotion via **MLflow's Model Registry**,
replacing the current path-based resolution of timestamped artifact directories.
Models are logged to MLflow at training time, registered as versioned entries per
`(model_type, subset)`, and promoted to production through an alias. Batch
prediction and the serving API resolve the production model **by name**, not by a
hard-coded path. MLflow's artifact store becomes the authoritative home for model
bytes.

This **revises the interim "disk is the source of truth" stance** from the
tracking step: a registry requires a single authoritative store, and keeping
model bytes in both the run dir and MLflow would create two sources of truth.
Models therefore move into MLflow; the timestamped run-dir/manifest scheme is
retired.

## Decisions (settled during brainstorming)

| Decision | Choice |
|---|---|
| Backend | MLflow Model Registry on the existing local SQLite store (`sqlite:///mlflow.db`) |
| Authoritative model store | MLflow artifact store; timestamped `artifacts/models/<ts>/` dirs retired |
| Logging flavor | Ridge → `mlflow.sklearn` (the pipeline is self-contained); GRU → custom `mlflow.pyfunc.PythonModel` wrapper (needs normalizer + windowing + rescaling) |
| Registration | **Auto-register a new version on every production training run** |
| Naming | One registered model per `(model_type, subset)`: `turbofan-{ridge,gru}-fd00{1-4}` |
| Promotion | Manual, via the `@production` **alias** (MLflow 3.x aliases, not legacy stages); immediate, **no approval gate** |
| Rollback | Promote an earlier version |
| Inference resolution | `predict`/`serve` load `models:/<name>@production` via `mlflow.pyfunc`; the path-based manifest layer is retired |
| Provenance | Native — each version links to the MLflow run (params/metrics/`val_loss` curve) that produced it |

## Architecture

### New module: `turbofan/registry.py`

A thin wrapper over MLflow's registry, mirroring the `tracking.py` seam:

- `model_name(model_type: str, subset: str) -> str` — canonical registered-model
  name, e.g. `turbofan-gru-fd001`.
- `log_and_register(model: Any, *, model_type: str, subset: str, ...) -> int` —
  log the model into the **active run** with the right flavor + signature, register
  it under `model_name(...)`, and return the new version number.
- `promote(name: str, version: int, alias: str = "production") -> None` — set the
  alias to a version.
- `resolve_uri(name: str, alias: str = "production") -> str` — `models:/<name>@<alias>`.
- `load(name: str, alias: str = "production") -> mlflow.pyfunc.PyFuncModel` — load
  the aliased model for inference.
- `list_registered() -> list[...]` — registered models, versions, current
  `@production` alias, key metric, and run link (for the listing CLI).

### Model packaging — reuse the inference logic, repackage the storage

The hard-won inference algorithms (feature engineering, windowing, prediction
scope, clipping) are **preserved and moved into MLflow model wrappers**; only the
packaging and resolution change.

- **Ridge** — `build_baseline_pipeline` is a self-contained sklearn `Pipeline`,
  logged via `mlflow.sklearn.log_model` with a signature. The engine-scope
  selection (last cycle per engine) and RUL clipping that live outside the
  pipeline are encapsulated in a thin pyfunc wrapper so the logged model honors
  the full `engine`-scope inference contract.
- **GRU** — a custom `mlflow.pyfunc.PythonModel` carrying the same payload as
  today's `model.pt` (`model_state_dict`, `feature_cols`, `normalizer_payload`,
  `sequence_config`, `max_rul`) and running the full `final_window`-scope path
  (normalize → window → forward → rescale → clip). It reuses the existing
  `inference/predictors.py` predict logic internally.

### Training integration

Inside the production MLflow run (created in the tracking step), after evaluation:
`registry.log_and_register(...)` logs + registers a **new version** named
`model_name(model_type, cfg.data.fd_subset)`; prediction/eval outputs (validation
and official-test predictions) are logged as MLflow artifacts. Promotion is **not**
automatic.

### Promotion / rollback CLI (no approval gate)

- `turbofan-promote <name> <version> --to production` → `registry.promote(...)`;
  repoints `@production` immediately. MLflow records the alias change.
- `turbofan-models` → lists registered models, versions, the `@production` alias,
  key metric (`val_rmse`), and run link.
- Rollback = `turbofan-promote <name> <older-version> --to production`.

### Inference resolution

- **`cli/predict.py`** — replace `--artifact <path>` with `--model <name>`
  (`--alias`, default `production`), or accept an explicit `models:/…` URI; load
  via `registry.load(...)`. The prediction/metadata CSV output contract is
  unchanged.
- **`inference/service.py` / `cli/serve_api.py`** — `create_app` resolves
  `models:/<name>@production` (via an env var, replacing `TURBOFAN_MODEL_ARTIFACT`,
  or an explicit arg) and loads the pyfunc model.

### Retired

- Writing `model_manifest.json`; `artifacts/models/<ts>/` as the model home;
  `inference/manifest.py`; the path-based `load_predictor` resolution. Model bytes
  and prediction artifacts now live in MLflow. The predictor predict logic is
  repackaged into the pyfunc wrappers (not rewritten).

### `run.log` — amendment to the logging step

Because run dirs are retired, the structured-logging step's `run.log` is logged as
an **MLflow artifact** on the run rather than written into the run dir. The
logging-step spec is amended accordingly.

## Testing

- **`registry.py`:** point MLflow at `tmp_path`; log + register a tiny synthetic
  model and assert a version is created; `promote` and assert
  `models:/<name>@production` resolves to it; roll back to an earlier version;
  `list_registered` shape.
- **Wrapper parity:** round-trip a tiny GRU and a tiny Ridge through
  `log_model → load → predict`, asserting predictions match the in-process
  predictor (guards the repackaging).
- **Training CLIs:** assert a registered version is created and linked to the run.
- **`predict`/`serve`:** resolve `models:/<name>@production` and predict; rewrite
  the path-based tests.
- All tests data-independent (synthetic fixtures only); no C-MAPSS download.

## Documentation

- Update README + `docs/roadmap.md`: models are logged and versioned in MLflow;
  promote with `turbofan-promote`; serving resolves `@production`. Mark step 4
  done.

## Out of scope (deferred)

- Approval / sign-off workflows (multi-party promotion gates).
- Additional aliases (`@staging`, `@champion`) — start with `@production` only.
- Remote / shared artifact store (S3, registry server) — store remains local.
- Automated, metric-threshold-driven promotion (CI-gated).

## Conventions honored

Google-style docstrings; full type annotations; `mypy --strict` clean; ruff
`E,F,W,I,UP,ANN` at line length 88; `--no-ff` merge; reports cite local data only.
