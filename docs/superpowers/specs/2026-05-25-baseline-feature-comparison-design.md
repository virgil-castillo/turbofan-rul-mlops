# Baseline Feature Comparison Design Spec

**Date:** 2026-05-25
**Status:** Draft for review

---

## Goal

Add a repeatable experiment script for comparing tabular Ridge baselines across
feature families:

- raw sensor features only
- raw sensor features plus rolling sensor features
- rolling sensor features only

The script should evaluate multiple rolling windows on the same deterministic
engine-level train/validation split, rank results by validation RMSE, and
optionally write a CSV for later analysis. It should support parallel execution
for HPC use without changing the one-model artifact workflow in
`scripts/train_baseline.py`.

## Feature Semantics

Estimator inputs are sensor-derived only.

`engine_id` is retained only long enough to group rolling calculations by
engine. `op_1`, `op_2`, and `op_3` are retained only long enough for
operational-condition normalization. `cycle`, `engine_id`, and all `op_*`
columns are dropped before the final estimator for every feature family.

The feature families are:

- `raw`: normalized raw sensor columns.
- `raw_plus_rolling`: normalized raw sensor columns plus normalized rolling
  sensor statistics.
- `rolling`: normalized rolling sensor statistics only.

For the first version, each requested rolling window is evaluated separately.
For example, `--windows 5 10 20` creates separate `raw_plus_rolling_w5`,
`raw_plus_rolling_w10`, `raw_plus_rolling_w20`, `rolling_w5`, `rolling_w10`,
and `rolling_w20` experiment rows. Combined multi-window feature sets can be
added later behind the same script if single-window results justify it.

## Architecture

Add reusable feature-selection support to `src/turbofan/models/baseline.py`
instead of hard-coding experiment behavior inside the script. The baseline
pipeline remains an sklearn `Pipeline`:

1. sensor dropping
2. rolling feature extraction when requested
3. operational normalization
4. model feature selection
5. low-variance filtering
6. imputation
7. scaling
8. Ridge regression

The new model feature selector accepts a feature-set name and removes columns
that are not valid estimator inputs. It should use column naming conventions
instead of fixed sensor counts:

- raw sensor columns match `s_*` without rolling suffixes
- rolling sensor columns contain rolling statistic suffixes such as
  `_rmean_`, `_rstd_`, `_rmin_`, or `_rmax_`

This keeps the comparison script thin and makes the behavior testable without
spawning subprocesses for every case.

## CLI Design

Add `scripts/compare_baseline_features.py`:

```bash
python scripts/compare_baseline_features.py \
  --config configs/default.yaml \
  --feature-sets raw raw_plus_rolling rolling \
  --windows 5 10 20 \
  --n-jobs 4 \
  --output artifacts/baseline_feature_comparison.csv
```

Arguments:

- `--config`: project config path, defaulting to `configs/default.yaml`.
- `--feature-sets`: feature families to evaluate, defaulting to all three.
- `--windows`: rolling window sizes, defaulting to `5 10 20`.
- `--n-jobs`: number of parallel workers, defaulting to `1`.
- `--output`: optional CSV path for results.

The script loads the dataset once, computes capped RUL labels once, and creates
one deterministic engine-level train/validation split. It then constructs a
list of experiment specs and evaluates them independently. Results are sorted
by `rmse` ascending before printing and writing CSV.

Output columns:

- `feature_set`
- `windows`
- `alpha`
- `n_features`
- `raw_prediction_min`
- `raw_prediction_max`
- `rmse`
- `mae`
- `phm08_score`

## Parallel Execution

Use `joblib.Parallel` with `n_jobs=args.n_jobs` to evaluate independent
experiment specs concurrently. Each worker receives the same train/validation
DataFrames and builds its own unfitted estimator, so there is no shared mutable
model state.

The implementation should keep output deterministic by collecting worker
results and sorting the final DataFrame, not by relying on worker completion
order. The default remains serial execution to avoid surprising local resource
use. In HPC jobs, users can set `--n-jobs` to the CPU allocation supplied by
the scheduler.

The script should not write per-model artifacts by default. It only writes the
comparison CSV when `--output` is provided, avoiding concurrent artifact
directory creation and keeping this script focused on model selection.

## Error Handling

The script should fail fast when:

- any window size is non-positive
- an unknown feature set is requested
- `--n-jobs` is zero
- a requested feature family produces no estimator features

Training data loading errors should surface from the existing loader. Missing
official test files are irrelevant because this comparison uses validation
metrics only.

## Testing Strategy

Follow test-first implementation.

Unit tests in `tests/models/test_baseline.py`:

- `raw` exposes raw sensor columns to Ridge and drops `cycle`, `engine_id`, and
  `op_*`.
- `raw_plus_rolling` exposes both raw and rolling sensor columns.
- `rolling` exposes rolling sensor columns and drops raw sensor columns.
- unsupported feature-set names raise `ValueError`.

Script tests in `tests/models/test_compare_baseline_features_cli.py`:

- the comparison helper returns one `raw` row plus one row per rolling
  feature-set/window combination.
- results include the expected metric and metadata columns.
- non-positive windows and zero `n_jobs` raise `ValueError`.
- the CLI writes a CSV when `--output` is supplied.

Verification commands:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
pytest tests/models/test_baseline.py tests/models/test_compare_baseline_features_cli.py
ruff check src/ tests/ scripts/
mypy src/turbofan
```

## Non-goals

- Saving a fitted model for every comparison row.
- Running official C-MAPSS test evaluation for every comparison row.
- Full hyperparameter search over Ridge alpha.
- Combined multi-window feature sets in the first version.
- Changing sequence model training.
