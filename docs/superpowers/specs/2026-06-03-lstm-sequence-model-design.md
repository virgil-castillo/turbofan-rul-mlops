# LSTM Sequence Model — Design

**Date:** 2026-06-03
**Status:** Approved for planning
**Scope:** One development branch

## Summary

Add LSTM as a second sequence RUL architecture alongside the existing GRU. The
LSTM layer itself is a small change (`nn.LSTM` instead of `nn.GRU`, plus handling
that LSTM returns a `(hidden, cell)` state tuple rather than just `hidden`). The
substantive work is generalizing the seams that are currently hardwired to GRU
by concrete type so the sequence stack admits a small **registry of RNN
architectures** without per-architecture fan-out. This is the change that
"unlocks" the sequence stack for the future Transformer without paying the
generalization cost twice.

The branch ships a real new model end-to-end: trainable via one generalized CLI,
versioned and promotable under its own MLflow registered-model name, and servable
through the existing inference and serving paths.

## Goals

- Introduce an LSTM sequence regressor that reuses the entire existing sequence
  pipeline: feature pipeline, operating-mode normalization, windowing,
  left/right-zero padding + `pack_padded_sequence`, training loop with early
  stopping, validation/official-test evaluation, and the final-window inference
  compute.
- Replace GRU-only concrete typing with an **RNN-scoped architecture registry**
  (name → builder) whose contract is defined by what GRU and LSTM genuinely
  share. The registry is deliberately **not** designed against the Transformer's
  needs; it will be widened when the Transformer lands with real requirements in
  hand.
- Keep **per-architecture** MLflow registered-model names
  (`turbofan-lstm-<subset>`, mirroring `turbofan-gru-<subset>`) so GRU and LSTM
  each hold their own `@production` alias and can serve simultaneously.
- Expose training through **one** generalized `turbofan-train-sequence` CLI whose
  architecture comes from config; keep `turbofan-train-sequence-gru` as a
  backward-compatible alias.
- Give LSTM its own optional `features.lstm` feature-engineering override block,
  seeded from the GRU-tuned values.
- Train, register, and benchmark LSTM production artifacts for the subsets GRU
  already covers.

## Non-Goals

- **Transformer** sequence model — deferred. The registry contract is scoped to
  RNNs and will be widened later, not designed speculatively now.
- **Tree baselines** (XGBoost / Random Forest) — different code path, separate
  future branch.
- **LSTM feature-engineering sweep + grounded report** — out of scope. We seed
  LSTM feature settings from GRU's tuned values; a dedicated LSTM sweep is a
  possible follow-up.
- No change to the Ridge path, the canonical raw C-MAPSS **inference input
  contract**, the **per-engine prediction semantics**, or the per-architecture
  registered-model naming convention.
- No move to a generic `turbofan-sequence-<subset>` registered name (explicitly
  rejected: it would force GRU and LSTM to compete for one `@production` slot).

## Background — Current GRU-Only Seams

The sequence stack works but is tied to GRU by concrete type in several places.
The design generalizes each:

| Seam | File | Current state |
| --- | --- | --- |
| Model class | `models/gru.py` | `GRURULRegressor` (`nn.GRU`); `forward` reads `hidden[-1]` |
| Training | `models/sequence_training.py` | `train_gru_model(model: GRURULRegressor, ...)`; `TrainingResult.model: GRURULRegressor` |
| Config | `config/schema.py` | `SequenceConfig.architecture: Literal["gru"]` |
| Train CLI | `cli/train_sequence_gru.py` | Asserts `architecture == "gru"`, else raises; constructs `GRURULRegressor` directly |
| Feature resolution | `config/schema.py` | `FeatureConfig.for_model(model: Literal["ridge", "gru"])`; `features.gru` block |
| Inference compute | `inference/predictors.py` | `gru_final_window_predictions`; `_MODEL_SCOPES` keyed `ridge`/`gru`; rebuilds `GRURULRegressor` |
| Inference schema | `inference/schemas.py` | `ModelType = Literal["ridge", "gru"]` |
| Registry | `registry.py` | `_log_gru_model`, `GRUFinalWindowModel` pyfunc, `model_type_from_name` accepts only `ridge`/`gru`, `_GRU_PIP_REQUIREMENTS` |

What is **already architecture-agnostic** and needs no change: `WindowedSequences`
and windowing (right-zero padding + `lengths`), `SequenceDataset` /
`build_sequence_loader` (yield 3-tuples), `OperatingModeNormalizer` and its
payload round-trip, the feature pipeline, the split, and the metrics.

## Architecture

### 1. Sequence model registry (PyTorch side)

Introduce an RNN-scoped registry mapping an architecture name to a builder that
returns an `nn.Module`. Two ways to realize this; the chosen contract is:

- A shared regressor module that owns the standard structure (RNN encoder →
  scalar `nn.Linear(hidden_size, 1)` head, `pack_padded_sequence` path,
  final-hidden-state extraction) and selects the recurrent layer (`nn.GRU` or
  `nn.LSTM`) by an `architecture` argument. The single behavioral difference —
  GRU's `forward` returns `hidden`, LSTM's returns `(hidden, cell)` — is
  normalized inside `forward` (take the hidden tensor, then `[-1]`).
- A name→builder registry (`"gru"`, `"lstm"`) so `model_type`/`architecture`
  strings resolve to a constructed module. The registry is the single seam new
  RNN architectures register through.

Constructor validation (positive sizes, `0 <= dropout < 1`, multi-layer dropout
gating) is shared, matching today's `GRURULRegressor` guards.

`GRURULRegressor` is preserved as a thin, backward-compatible shim (so existing
GRU checkpoints and imports keep working) **or** folded into the shared module
with `"gru"` as the registered name — the planning step picks whichever keeps the
existing GRU checkpoint state-dict keys loadable. Checkpoint compatibility for
already-registered GRU models is a hard requirement: their state-dict key names
must not change.

### 2. Training (`models/sequence_training.py`)

- Rename/generalize `train_gru_model` → `train_sequence_model`, typed against the
  shared module (or `nn.Module`) instead of `GRURULRegressor`.
- `TrainingResult.model` widens to the shared module type.
- The training loop, criterion (`MSELoss` on `targets / max_rul`), early stopping,
  history, and `predict_windows` are unchanged — they never depended on the GRU
  specifically.
- A `train_gru_model` alias may be retained if anything outside the CLI imports
  it; otherwise call sites are updated.

### 3. Config (`config/schema.py`)

- `SequenceConfig.architecture: Literal["gru", "lstm"]` (still defaults to
  `"gru"`).
- `FeatureConfig.for_model` accepts `Literal["ridge", "gru", "lstm"]`; add an
  optional `lstm: ModelFeatureConfig | None` block resolved exactly like `gru`.
- The shared `SequenceConfig` hyperparameters (`window_size`, `hidden_size`,
  `num_layers`, `dropout`, `learning_rate`, `batch_size`, `epochs`, `patience`,
  `device`) apply to both architectures unchanged — LSTM and GRU share the same
  hyperparameter surface.

### 4. Train CLI

- Generalize `cli/train_sequence_gru.py` into a `turbofan-train-sequence`
  entrypoint (new module, e.g. `cli/train_sequence.py`):
  - Remove the `architecture == "gru"` assertion.
  - Construct the model through the registry using `cfg.sequence.architecture`.
  - Resolve features with `cfg.features.for_model(cfg.sequence.architecture)`.
  - Register under `model_type=cfg.sequence.architecture`.
  - Tag `model_type` with the architecture; param-log `architecture`.
  - The checkpoint payload gains/sets `architecture` (it already serializes
    `sequence_config`, which carries it) so inference can rebuild the right layer.
- `pyproject.toml`: add `turbofan-train-sequence`; keep
  `turbofan-train-sequence-gru` pointing at the same `main` (alias) for backward
  compatibility.

### 5. Inference compute (`inference/predictors.py`)

- Generalize `gru_final_window_predictions` → a single
  `sequence_final_window_predictions(payload, frame)` that reads `architecture`
  from the payload's `sequence_config` and builds the correct module via the
  registry. The normalize → final-window → forward → rescale-by-`max_rul` → clip
  path is identical across RNNs.
- `_MODEL_SCOPES` gains `"lstm": "final_window"`.
- Keep a `gru_final_window_predictions` alias delegating to the generalized
  function if any external caller depends on the name.

### 6. Inference schema (`inference/schemas.py`)

- `ModelType = Literal["ridge", "gru", "lstm"]`. Both `gru` and `lstm` map to the
  `final_window` prediction scope; `ridge` stays `engine`.

### 7. MLflow registry (`registry.py`)

- One **shared sequence pyfunc wrapper** (rename `GRUFinalWindowModel` →
  `SequenceFinalWindowModel`, or keep `GRUFinalWindowModel` as an alias) that, in
  `predict`, calls the generalized `sequence_final_window_predictions`. Because
  the architecture is read from the payload, no per-architecture wrapper class is
  needed.
- `log_and_register` accepts `model_type in {"ridge", "gru", "lstm"}`; `gru` and
  `lstm` both route to the shared sequence logger (rename `_log_gru_model` →
  `_log_sequence_model`). Pinned pip requirements: the GRU set (torch-based) is
  reused for both RNNs (rename `_GRU_PIP_REQUIREMENTS` →
  `_SEQUENCE_PIP_REQUIREMENTS`).
- `model_type_from_name` accepts `gru` and `lstm`.
- `model_name("lstm", "FD001") == "turbofan-lstm-fd001"`. GRU and LSTM are
  independent registered models, each with its own version counter and
  `@production` alias.

### 8. Configs

- `default.yaml`: no architecture change (stays `gru`); optionally document the
  `lstm` feature block.
- Each `configs/subsets/fd00{1-4}.yaml`: add a `features.lstm` block seeded from
  that subset's existing `features.gru` values. (FD001 example: `feature_set:
  rolling_mean`, `windows: [20]`.) These are starting points; they may diverge
  after a future LSTM sweep.
- To train an LSTM for a subset, the operator sets `sequence.architecture: lstm`
  — either by editing the subset config, layering a small `architecture: lstm`
  override config on top via `_base_`, or (planning decision) adding a thin
  per-architecture config. The branch will ship a documented, reproducible way to
  select LSTM that does not require hand-editing the shared subset file at train
  time. **Open planning decision:** prefer a dedicated
  `configs/subsets/fd00{1-4}_lstm.yaml` (each `_base_`-referencing its GRU subset
  config and overriding only `sequence.architecture`) so GRU and LSTM training
  are both reproducible from version-controlled configs.

## Data Flow (unchanged shape, architecture-parameterized)

Training:

```
load_raw_train → add_rul_column → split_by_engine
  → build_feature_pipeline(features.for_model(architecture))
  → build_sliding_windows (per-engine, padded, lengths)
  → build_sequence_loader
  → registry.build(architecture, input_size, hidden_size, num_layers, dropout)
  → train_sequence_model (MSE on normalized target, early stop on val RMSE)
  → evaluate validation windows + optional official test
  → MLflow: log params/metrics/history, log_and_register(model_type=architecture)
```

Inference (batch CLI / serving API, unchanged entrypoints):

```
canonical raw records → validate_raw_records
  → registry.load(models:/turbofan-<arch>-<subset>@<alias>)
  → SequenceFinalWindowModel.predict
  → sequence_final_window_predictions (reads architecture from payload)
  → normalize → final windows → forward → rescale → clip
  → engine_id/cycle/prediction frame
```

## Error Handling

- Reuse existing checkpoint-payload validation in `inference/predictors.py`
  (`_positive_int`, `_string_sequence`, `_mapping`, etc.). Generalize the error
  messages from "GRU checkpoint field …" to "sequence checkpoint field …".
- Unknown architecture name (config or payload) → `ValueError` from the registry
  with the list of supported names. The train CLI surfaces this before training;
  inference surfaces it at load time.
- Existing GRU checkpoints remain loadable: the shared module must produce the
  same state-dict keys for `architecture == "gru"`. This is verified by a test
  loading a pre-change GRU payload (or an equivalently-keyed fixture).
- LSTM-specific: the `(hidden, cell)` tuple is unpacked inside `forward`; packed
  sequences behave identically. No new error surface beyond GRU.

## Testing

Mirror the existing GRU coverage; LSTM should not be a second-class citizen.

- **Model unit tests** (`models/`): LSTM constructor validation (positive sizes,
  dropout bounds, multi-layer dropout gating); `forward` output shape
  `(batch_size,)` for both packed (`lengths` given) and unpacked paths; the
  registry returns the correct layer type per name and raises on unknown names;
  GRU `forward` behavior unchanged.
- **Training** (`models/sequence_training` tests): `train_sequence_model` runs an
  LSTM for a couple of epochs on a tiny synthetic set, early-stops, and restores
  the best epoch; `predict_windows` rescales by `max_rul`.
- **Inference** (`inference/`): `sequence_final_window_predictions` round-trips an
  LSTM checkpoint payload to the engine/cycle/prediction frame; scope resolves to
  `final_window`; GRU payload still works through the generalized function.
- **Registry** (`registry` tests): `model_name("lstm", subset)`,
  `model_type_from_name("turbofan-lstm-fd001") == "lstm"`,
  `log_and_register(model_type="lstm")` registers a version; the shared pyfunc
  predicts; a registered LSTM resolves and predicts through `load_predictor`.
- **Config** (`config` tests): `for_model("lstm")` resolves the `features.lstm`
  block with GRU/shared fallback; `SequenceConfig` accepts `architecture: lstm`;
  `_base_` composition of an LSTM subset config (if adopted) overrides only
  `sequence.architecture`.
- **CLI**: `turbofan-train-sequence` trains and registers an LSTM in-process on a
  small fixture (consistent with the existing in-process CLI test approach);
  `turbofan-predict --model turbofan-lstm-<subset>` returns predictions.
- **Backward compatibility**: existing GRU checkpoint loads; the
  `turbofan-train-sequence-gru` alias still trains a GRU.

Tests follow TDD per the project contract and must satisfy `ruff`, `mypy
--strict`, and full-docstring style. New code keeps line length 88 and Google
docstrings.

## Production Artifacts & Benchmark

- Train and register LSTM production artifacts for the subsets GRU already
  covers, using the seeded `features.lstm` settings and the shared
  `SequenceConfig` hyperparameters.
- Extend the cross-model comparison so LSTM appears alongside Ridge and GRU on the
  engine-level validation split. Any report stays grounded solely in this repo's
  run data and EDA, per the project's grounded-reporting rule.

## Roadmap Impact

Closes the first item under "Future — Additional Models" (LSTM). Updates the
"Single-layer GRU by design … LSTM, TCN deferred" note to reflect that LSTM is
now supported through the architecture registry, and that the registry contract
is RNN-scoped pending the Transformer.

## Open Questions for Planning

1. **GRU module preservation vs. fold-in:** keep `GRURULRegressor` as a shim, or
   fold GRU into the shared module under the `"gru"` registry name? Decide by
   whichever guarantees existing GRU checkpoint state-dict keys load unchanged.
2. **LSTM config selection:** confirm the dedicated
   `configs/subsets/fd00{1-4}_lstm.yaml` approach (recommended) versus an inline
   `architecture` field flipped per run.
3. **Alias breadth:** confirm which legacy symbol names
   (`train_gru_model`, `gru_final_window_predictions`, `GRUFinalWindowModel`,
   `_GRU_PIP_REQUIREMENTS`) need retained aliases versus a clean rename with
   call-site updates, based on a grep for external references.
