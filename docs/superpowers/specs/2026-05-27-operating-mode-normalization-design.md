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
- Preserve existing import surfaces where practical so current code and tests
  do not need broad rewrites.
- Make GRU artifacts self-contained by saving enough normalizer metadata to
  reconstruct preprocessing at inference time.
- Keep legacy GRU checkpoint loading working for artifacts that only contain
  `normalizer_means` and `normalizer_stds`.

## Non-Goals

- Do not add MLflow, a model registry, or experiment tracking.
- Do not change the public prediction API schema.
- Do not add padding or masking for short GRU engines.
- Do not tune model hyperparameters as part of this change.
- Do not introduce a user-facing config option for mode count in v1; derive it
  from `fd_subset`.

## Proposed API

Add a shared preprocessing package:

```text
src/turbofan/preprocessing/__init__.py
src/turbofan/preprocessing/normalization.py
```

Expose:

```python
FD_SUBSET_MODE_COUNTS = {
    "FD001": 1,
    "FD002": 6,
    "FD003": 1,
    "FD004": 6,
}

def mode_count_for_subset(fd_subset: str) -> int:
    """Return the operating-mode count for a C-MAPSS subset."""
```

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

Keep compatibility modules:

- `turbofan.features.normalizer.OperationalNormalizer` should become a thin
  subclass or wrapper around `OperatingModeNormalizer`.
- `turbofan.sequences.normalize.SequenceNormalizer` can remain as the global
  z-score normalizer for legacy tests and fallback checkpoint loading, but new
  GRU training should use `OperatingModeNormalizer`.

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
- Use `ddof=0` to match the existing sequence normalizer.
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

Pass these into `OperationalNormalizer`. In `train_baseline.py`, derive:

```python
n_modes = mode_count_for_subset(cfg.data.fd_subset)
```

and pass `random_state=cfg.data.random_seed`.

The baseline estimator should still not receive operating setting columns as
model features. The op columns are used for mode assignment inside
normalization, then removed by the existing downstream feature selection and
identifier-dropping steps.

### GRU Training

In `train_sequence_gru.py`, replace:

```python
normalizer = SequenceNormalizer(feature_cols=feature_cols)
```

with:

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

Persist both the new payload and legacy flat stats for one release:

```python
"normalizer_type": "operating_mode",
"normalizer_payload": normalizer.to_payload(),
"normalizer_means": normalizer.global_means_for(feature_cols),
"normalizer_stds": normalizer.global_stds_for(feature_cols),
```

The exact helper name for flat stats can differ, but the checkpoint must keep
`normalizer_means` and `normalizer_stds` until the inference fallback tests are
updated around the new payload.

### GRU Sweeps

Update both GRU experiment entrypoints:

- `turbofan.experiments.sequence_gru_sweep`
- `turbofan.experiments.feature_gru_sweep`

Use `OperatingModeNormalizer` with subset-derived mode count and config random
seed. Feature sweeps that add rolling columns should pass the complete
`feature_cols` list after rolling feature construction.

### Inference

Update `turbofan.inference.predictors._normalizer_from_payload`:

- If checkpoint has `normalizer_type == "operating_mode"` and
  `normalizer_payload`, reconstruct with `OperatingModeNormalizer.from_payload`.
- Otherwise keep the existing `SequenceNormalizer` fallback from
  `normalizer_means` and `normalizer_stds`.

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

- Constructor raises `ValueError` if `n_modes <= 0` or `std_floor < 0`.
- `fit` raises `KeyError` for missing op columns or explicit feature columns.
- `fit` raises `ValueError` when `n_modes > number_of_training_rows`.
- `from_payload` raises `ValueError` for unsupported schema version, missing
  required keys, non-numeric stats, or inconsistent feature/stat keys.
- `transform` raises `RuntimeError` if called before `fit`.

## Test Plan

Add or update focused tests:

- `tests/preprocessing/test_normalization.py`
  - `mode_count_for_subset` returns `1, 6, 1, 6`.
  - Single-mode normalization does not create exact tuple groups.
  - Multi-mode synthetic data normalizes sensors separately by learned mode.
  - Op columns included as features are globally normalized.
  - Metadata and `rul` are preserved.
  - Near-zero standard deviations are floored.
  - `to_payload` and `from_payload` round-trip and produce identical
    transforms.

- `tests/features/test_normalizer.py`
  - Existing `OperationalNormalizer` behavior still passes through the
    compatibility wrapper.
  - Update exact-tuple assumptions to mode-bucket assumptions where needed.

- `tests/sequences/test_normalize.py`
  - Keep existing tests for legacy `SequenceNormalizer`.
  - Add a sequence-facing test that the GRU path can use
    `OperatingModeNormalizer` with explicit `feature_cols`.

- `tests/models/test_train_baseline_cli.py`
  - Assert `build_baseline_pipeline` receives subset-derived `n_modes` and
    `random_state`.

- `tests/models/test_train_sequence_gru_cli.py`
  - Assert the saved checkpoint contains `normalizer_type` and
    `normalizer_payload`.
  - Assert `FD002` training uses six modes.

- `tests/inference/test_predictors.py`
  - New payload reconstructs an `OperatingModeNormalizer`.
  - Legacy payload without `normalizer_payload` still reconstructs the old
    `SequenceNormalizer` fallback.

Run:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
ruff check src/ tests/
mypy src/turbofan
pytest
```

## Acceptance Criteria

- Baseline and GRU training both use subset-derived operating-mode
  normalization.
- FD001 remains single-mode and does not group by exact op tuples.
- FD002/FD004 use six learned mode buckets.
- GRU inference reproduces training preprocessing from the saved artifact.
- Existing public CLIs and API request schemas remain unchanged.
- Legacy GRU artifacts with only flat normalizer stats still load.
- Ruff, mypy strict, and pytest pass.

## Implementation Notes

- Keep changes tightly scoped. Do not refactor model training beyond replacing
  the normalizer construction and checkpoint payload.
- Prefer pandas operations that preserve column order and DataFrame indexes.
- Use Google-style docstrings and fully annotated public signatures.
- Avoid storing sklearn estimator objects inside Torch payloads. Store simple
  payload values instead.
