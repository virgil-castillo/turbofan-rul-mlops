# Operating-Mode Normalization Design

## Context

The repository currently has two normalization paths:

- `turbofan.features.normalizer.OperationalNormalizer` is used by the
  sklearn baseline feature pipeline. It groups by exact
  `op_1`, `op_2`, and `op_3` tuples.
- `turbofan.sequences.normalize.SequenceNormalizer` is used by GRU
  training and inference. It applies one global z-score over all configured
  sequence features.

Both paths are train-fitted and avoid validation/test leakage, but neither is
the right long-term normalization contract for all C-MAPSS subsets. Exact
operating-setting tuples are too granular because the settings are
continuous-like. A local data check showed many exact tuples even for FD001,
which is documented as one operating condition. Global sequence normalization
is acceptable for FD001/FD003 but becomes a poor default for FD002/FD004,
where the same sensor can have different ranges under different operating
modes.

The implementation should add one shared operating-mode-aware normalization
contract and use it consistently from baseline training, GRU training, GRU
sweeps, and GRU inference.

## Goals

- Normalize sensor and sensor-derived features relative to learned operating
  mode buckets, not exact operating-setting tuples.
- Keep public inference input unchanged: raw canonical C-MAPSS records only.
- Fit all normalization state on training rows only.
- Replace old baseline and sequence normalizer paths with the shared
  preprocessing normalizer; do not keep public compatibility wrappers for
  stale abstractions.
- Make GRU artifacts self-contained by saving enough normalizer metadata to
  reconstruct preprocessing at inference time.
- Reject legacy GRU checkpoints that only contain flat `normalizer_means` and
  `normalizer_stds`; new models must be retrained with the operating-mode
  normalizer payload.

## Non-Goals

- Do not add MLflow, a model registry, or experiment tracking.
- Do not change the public prediction API schema.
- Do not add padding or masking for short GRU engines.
- Do not tune model hyperparameters as part of this change.
- Do not introduce a user-facing config option for mode count in v1; derive it
  from `fd_subset`.
- Do not preserve `turbofan.features.normalizer.OperationalNormalizer` as a
  wrapper or subclass.
- Do not preserve exact operating-setting tuple grouping.
- Do not keep `SequenceNormalizer` as a public or active fallback
  normalization abstraction.
- Do not support loading legacy flat-stat-only GRU checkpoints after this
  migration.

## Proposed API

Add a shared preprocessing package:

```text
src/turbofan/preprocessing/__init__.py
src/turbofan/preprocessing/normalization.py
```

Expose:

```python
CMAPSS_SUBSET_MODE_COUNTS = {
    "FD001": 1,
    "FD002": 6,
    "FD003": 1,
    "FD004": 6,
}

def mode_count_for_subset(fd_subset: str) -> int:
    """Return the EDA-confirmed mode count for a supported C-MAPSS subset."""
```

`CMAPSS_SUBSET_MODE_COUNTS` is fixed preprocessing policy for supported
C-MAPSS subsets. The counts come from the C-MAPSS subset definition and are
confirmed by EDA over the operating-setting columns. They are not dynamically
inferred during every training run. Training still fits mode centers and
normalization statistics from training rows only.

Add `OperatingModeNormalizer` in `normalization.py`. It should be
sklearn-compatible enough for pipeline use and also usable directly by the
sequence path.

Constructor:

```python
class OperatingModeNormalizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_cols: Sequence[str] | None = None,
        op_cols: Sequence[str] | None = None,
        n_modes: int = 1,
        std_floor: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        ...
```

Behavior:

- `feature_cols=None` means infer numeric feature columns during `fit` by
  excluding `engine_id`, `cycle`, and `rul`.
- `op_cols=None` defaults to `["op_1", "op_2", "op_3"]`.
- `sensor_feature_cols_` are fitted feature columns excluding operating
  setting columns.
- Operating setting columns included in `feature_cols` are normalized globally.
- Sensor and sensor-derived columns are normalized per operating mode.
- Metadata and target columns are copied through unchanged.

Persistence helpers:

```python
def to_payload(self) -> dict[str, object]:
    """Return a JSON/Torch-serializable normalizer payload."""

@classmethod
def from_payload(cls, payload: Mapping[str, object]) -> Self:
    """Reconstruct a fitted normalizer from a serialized payload."""
```

## Normalization Semantics

Mode assignment:

- If `n_modes == 1`, do not fit KMeans. Assign every row to mode `0`.
- If `n_modes > 1`, fit `sklearn.cluster.KMeans` on training
  `op_1`, `op_2`, and `op_3`.
- Use `random_state` from `cfg.data.random_seed`.
- Use `n_init=10` for stable behavior across environments.
- During `transform`, predict the mode for each row from its operating
  settings.

Statistics:

- Fit per-mode mean and standard deviation for sensor and sensor-derived
  columns.
- Fit global mean and standard deviation for all normalized feature columns.
- Use `ddof=0` to preserve the current population-standard-deviation
  convention.
- Replace missing, zero, or near-zero standard deviations with `1.0` when
  `abs(std) <= std_floor`.
- If a mode has missing stats for a transformed row, fall back to global stats.

Column policy:

- Preserve `engine_id`, `cycle`, and `rul` exactly.
- Preserve op columns if they are not in `feature_cols`.
- If op columns are in `feature_cols`, normalize them globally.
- Normalize only fitted feature columns that are present during `transform`.
- Raise `KeyError` when required `feature_cols` or `op_cols` are missing.
- Return a copy of the input DataFrame.

## Integration Design

### Baseline

Update `build_feature_pipeline` and `build_baseline_pipeline` to accept:

```python
n_modes: int = 1
random_state: int = 42
```

Import and use
`turbofan.preprocessing.normalization.OperatingModeNormalizer` directly. Do
not route baseline training through
`turbofan.features.normalizer.OperationalNormalizer`. Remove stale imports and
calls to `OperationalNormalizer`; the old baseline normalizer path is
replaced, not wrapped.

In `train_baseline.py`, derive:

```python
n_modes = mode_count_for_subset(cfg.data.fd_subset)
```

and pass `n_modes` and `random_state=cfg.data.random_seed` into
`OperatingModeNormalizer`.

The baseline estimator should still not receive operating setting columns as
model features. The op columns are used for mode assignment inside
normalization, then removed by the existing downstream feature selection and
identifier-dropping steps.

### GRU Training

In `train_sequence_gru.py`, replace the existing global sequence-normalizer
construction with:

```python
normalizer = OperatingModeNormalizer(
    feature_cols=feature_cols,
    n_modes=mode_count_for_subset(cfg.data.fd_subset),
    random_state=cfg.data.random_seed,
)
```

Use the same train/validation/test flow:

- `fit_transform(train_df)`
- `transform(val_df)`
- `transform(test_raw)`
- then build windows from normalized frames.

Persist the new payload:

```python
"normalizer_type": "operating_mode",
"normalizer_payload": normalizer.to_payload(),
```

Do not write `normalizer_means` and `normalizer_stds` as the normalization
contract for new checkpoints. Those fields describe the old flat-stat
checkpoint format and are not sufficient to reconstruct operating-mode-aware
preprocessing.

### GRU Sweeps

Update both GRU experiment entrypoints:

- `turbofan.experiments.sequence_gru_sweep`
- `turbofan.experiments.feature_gru_sweep`

Use `OperatingModeNormalizer` with subset-derived mode count and config random
seed. Feature sweeps that add rolling columns should pass the complete
`feature_cols` list after rolling feature construction.

Remove active source-code usage of `SequenceNormalizer` from baseline
training, GRU training, GRU sweeps, and GRU inference. Remove stale tests that
validate `SequenceNormalizer` behavior no active path uses.

### Inference

Update `turbofan.inference.predictors._normalizer_from_payload`:

- Require `normalizer_type == "operating_mode"` and `normalizer_payload`.
- Reconstruct with `OperatingModeNormalizer.from_payload`.
- If a checkpoint only has legacy flat stats such as `normalizer_means` and
  `normalizer_stds`, raise a clear `ValueError` instructing the user to
  retrain the model with an operating-mode normalizer payload.
- Do not reconstruct or expose `SequenceNormalizer` as the fallback.

`GRUPredictor.predict` should call `self._normalizer.transform(frame)` exactly
as it does today. No public inference schema changes are required.

## Serialization Payload

`OperatingModeNormalizer.to_payload()` should return only JSON/Torch-safe
objects:

```python
{
    "schema_version": 1,
    "normalizer_type": "operating_mode",
    "feature_cols": [...],
    "op_cols": ["op_1", "op_2", "op_3"],
    "sensor_feature_cols": [...],
    "n_modes": 1,
    "std_floor": 0.001,
    "random_state": 42,
    "mode_centers": None | [[...], ...],
    "global_means": {"s_1": 0.0, ...},
    "global_stds": {"s_1": 1.0, ...},
    "mode_means": {"0": {"s_1": 0.0, ...}},
    "mode_stds": {"0": {"s_1": 1.0, ...}},
}
```

For `n_modes == 1`, `mode_centers` should be `None`. For multi-mode payloads,
`from_payload` should reconstruct a fitted KMeans-compatible assignment state
from `mode_centers`. The implementation can either restore a minimal fitted
`KMeans` instance or compute nearest-center assignment directly during
`transform`; prefer direct nearest-center assignment because it is easier to
serialize and test.

## Error Handling

- `mode_count_for_subset` raises `ValueError` with a clear message for
  unsupported C-MAPSS subset names.
- Constructor raises `ValueError` if `n_modes <= 0` or `std_floor < 0`.
- `fit` raises `KeyError` for missing op columns or explicit feature columns.
- `fit` raises `ValueError` when `n_modes > number_of_training_rows`.
- `from_payload` raises `ValueError` for unsupported schema version, missing
  required keys, non-numeric stats, or inconsistent feature/stat keys.
- GRU checkpoint loading raises `ValueError` when `normalizer_type` is missing,
  is not `"operating_mode"`, or `normalizer_payload` is missing. Legacy
  flat-stat-only checkpoints must fail with a message that says to retrain the
  model.
- `transform` raises `RuntimeError` if called before `fit`.

## Test Plan

Add or update focused tests:

- `tests/preprocessing/test_normalization.py`
  - Supported C-MAPSS subsets resolve to positive integer mode counts.
  - Unsupported subset names raise a clear error.
  - FD001 and FD003 are treated as single-condition subsets.
  - FD002 and FD004 are treated as multi-condition subsets.
  - Single-mode normalization assigns all rows to mode `0` and does not fit
    KMeans or group by exact operating-setting tuples.
  - Multi-mode synthetic data fits learned operating-mode centers from
    training operating-setting rows and normalizes sensor features separately
    by learned mode.
  - Op columns included as features are normalized globally.
  - Metadata and `rul` are preserved.
  - Near-zero standard deviations are replaced according to `std_floor`.
  - `to_payload` and `from_payload` round-trip and produce identical
    transforms.

- `tests/features/test_normalizer.py`
  - Remove tests for `OperationalNormalizer`.
  - Remove or update imports that refer to
    `turbofan.features.normalizer.OperationalNormalizer`.

- `tests/sequences/test_normalize.py`
  - Remove stale `SequenceNormalizer` tests.
  - Add or keep only tests proving sequence-facing code can use
    `OperatingModeNormalizer` with explicit `feature_cols`.

- `tests/models/test_train_baseline_cli.py`
  - Assert baseline training constructs `OperatingModeNormalizer` directly.
  - Assert it receives the subset-derived mode count and config random seed.

- `tests/models/test_train_sequence_gru_cli.py`
  - Assert saved checkpoints contain `normalizer_type == "operating_mode"`
    and `normalizer_payload`.
  - Assert GRU training constructs `OperatingModeNormalizer` using
    subset-derived mode policy and config random seed.
  - Do not assert a literal FD002 value unless the test is explicitly named as
    a C-MAPSS policy test.

- GRU sweep tests, where present
  - Assert sweeps construct `OperatingModeNormalizer` using subset-derived mode
    policy and config random seed.

- `tests/inference/test_predictors.py`
  - New payload reconstructs an `OperatingModeNormalizer`.
  - Legacy flat-stat-only checkpoints are rejected with a clear retraining
    error.
  - Remove assertions that legacy payloads reconstruct `SequenceNormalizer`.

Run:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
ruff check src/ tests/
mypy src/turbofan
pytest
```

## Acceptance Criteria

- Single-condition C-MAPSS subsets use the single-mode normalization path.
- Multi-condition C-MAPSS subsets use EDA-confirmed subset mode counts and
  learn mode centers from training operating-setting rows.
- Baseline training, GRU training, GRU sweeps, and GRU inference all use
  `OperatingModeNormalizer`.
- GRU inference reproduces training preprocessing from the saved artifact.
- Public CLIs and inference request schemas remain unchanged.
- Legacy flat-stat-only GRU checkpoints are rejected with a clear retraining
  error instead of loading through `SequenceNormalizer`.
- Ruff, mypy strict, and pytest pass.

## Implementation Notes

- Keep changes tightly scoped. Do not refactor model training beyond replacing
  the normalizer construction and checkpoint payload.
- Prefer pandas operations that preserve column order and DataFrame indexes.
- Use Google-style docstrings and fully annotated public signatures.
- Avoid storing sklearn estimator objects inside Torch payloads. Store simple
  payload values instead.
