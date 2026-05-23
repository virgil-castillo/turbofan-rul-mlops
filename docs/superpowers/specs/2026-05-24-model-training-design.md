# Model Training: Sub-project 4 Design Spec

**Date:** 2026-05-24
**Sub-project:** 4 of 6
**Depends on:** Sub-project 1 (Foundation), Sub-project 3 (Feature Engineering)
**Status:** Approved

---

## Goal

Build a tabular baseline RUL prediction training workflow using the existing
feature engineering pipeline. The milestone trains a deterministic sklearn
baseline, evaluates it with RUL-focused metrics, supports an engine-level
validation split from C-MAPSS training data, optionally evaluates against the
official C-MAPSS test RUL file, and writes local experiment artifacts.

The default baseline is `Ridge`, selected because it is fast, deterministic,
easy to inspect, and a useful yardstick for the feature pipeline. Neural and
sequence models are explicitly deferred to Sub-project 5.

## Architecture

The `models/` subpackage contains reusable training primitives split by
concern: engine-level splitting, metrics, baseline estimator construction,
evaluation helpers, and artifact persistence. `scripts/train_baseline.py` is a
thin orchestration entry point that loads config, calls these primitives, and
writes run outputs.

Training uses sklearn-compatible composition: the existing
`build_feature_pipeline()` is chained with an sklearn regressor so feature
fitting and model fitting happen only on training-engine rows. Validation uses
a split by `engine_id`, never a row-level split, to prevent the same engine's
trajectory from appearing in both training and validation data. Official test
evaluation is supported when `test_*.txt` and `RUL_*.txt` are present; it
predicts only the final available cycle for each test engine, matching the
C-MAPSS test label semantics.

Experiment tracking is local and lightweight for this milestone. Each run
writes a fitted `joblib` model, metrics JSON, resolved config JSON, and
optional prediction CSVs under a gitignored artifact directory. MLflow is
deferred until the baseline contract is stable.

## Tech Stack

Python 3.12, pandas, numpy, scikit-learn, joblib, pytest

---

## File Structure

```
turbofan-rul-mlops/
|-- configs/
|   `-- default.yaml                       # add model/artifact config
|-- scripts/
|   `-- train_baseline.py                  # CLI orchestration
|-- src/turbofan/
|   `-- models/
|       |-- __init__.py
|       |-- artifacts.py                   # local run artifact persistence
|       |-- baseline.py                    # baseline pipeline factory
|       |-- evaluate.py                    # validation/test evaluation helpers
|       |-- metrics.py                     # RMSE, MAE, PHM08 score
|       `-- split.py                       # engine-level train/validation split
|-- tests/
|   `-- models/
|       |-- __init__.py
|       |-- test_artifacts.py
|       |-- test_baseline.py
|       |-- test_evaluate.py
|       |-- test_metrics.py
|       `-- test_split.py
`-- .gitignore                             # ignore local model artifacts
```

---

## Component Designs

### 1. Config additions

Extend the Pydantic config schema with a `ModelConfig` nested under the
top-level project config:

```python
class ModelConfig(BaseModel):
    """Configuration for baseline model training.

    Args:
        name: Baseline model identifier.
        alpha: Ridge regularization strength.
        artifact_dir: Directory for local run artifacts.
    """

    name: Literal["ridge"] = "ridge"
    alpha: float = Field(default=1.0, gt=0.0)
    artifact_dir: Path = Path("artifacts/models")
```

`ProjectConfig` gains `model: ModelConfig = Field(default_factory=ModelConfig)`.
Keeping model configuration explicit allows `scripts/train_baseline.py` to be
driven by `configs/default.yaml` while maintaining backward-compatible
defaults for tests that construct `ProjectConfig` or `DataConfig` directly.

`configs/default.yaml` adds:

```yaml
model:
  name: ridge
  alpha: 1.0
  artifact_dir: artifacts/models
```

`pyproject.toml` adds `joblib` to main dependencies because the package imports
it directly when saving fitted estimators.

### 2. `models/split.py` - engine-level validation split

```python
def split_by_engine(
    df: pd.DataFrame,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training DataFrame into train/validation engine groups."""
```

- Discovers unique `engine_id` values.
- Shuffles engine IDs with `numpy.random.default_rng(random_seed)`.
- Assigns whole engines to train or validation sets.
- Guarantees no engine appears in both outputs.
- Raises `ValueError` when fewer than two engines are available or when the
  requested split would create an empty train or validation set.

This function operates on already-loaded training data and does not compute
labels. Label computation remains in `turbofan.data.labels`.

### 3. `models/metrics.py` - RUL metrics

```python
def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute root mean squared error."""

def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute mean absolute error."""

def phm08_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute the asymmetric PHM08 RUL score."""

def regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Compute all baseline regression metrics."""
```

`RMSE` is the primary model selection metric. `MAE` is included because it is
easy to interpret. The PHM08 score is included as a domain-specific secondary
metric using the standard asymmetric scoring form:

- Let `d = y_pred - y_true`.
- If `d < 0`, add `exp(-d / 13) - 1`.
- If `d >= 0`, add `exp(d / 10) - 1`.

This penalizes late predictions more sharply than early predictions. Metrics
validate equal lengths and reject NaN inputs with `ValueError`.

### 4. `models/baseline.py` - baseline training pipeline

```python
def build_baseline_pipeline(
    model_name: Literal["ridge"] = "ridge",
    alpha: float = 1.0,
    windows: list[int] | None = None,
    op_cols: list[str] | None = None,
) -> Pipeline:
    """Build an unfitted feature-plus-regressor sklearn Pipeline."""
```

The returned pipeline has two named steps:

- `"features"`: the existing `build_feature_pipeline(windows, op_cols)`
- `"model"`: `Ridge(alpha=alpha)`

The design deliberately keeps the first model simple. A stronger sklearn
ensemble can be added later behind the same factory signature, but this
milestone only requires `ridge`.

### 5. `models/evaluate.py` - validation and official test evaluation

```python
def add_rul_column(df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    """Return a copy of training data with a computed ``rul`` column."""

def split_features_target(
    df: pd.DataFrame,
    target_col: str = "rul",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from target labels."""

def evaluate_rows(
    estimator: RegressorMixin,
    df: pd.DataFrame,
    target_col: str = "rul",
) -> dict[str, float]:
    """Evaluate predictions for labeled rows."""

def select_last_cycle_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Select the final available row for each engine."""

def align_official_test_labels(
    last_cycle_df: pd.DataFrame,
    rul_labels: pd.Series,
) -> pd.Series:
    """Align official RUL labels to final-cycle test rows."""
```

Validation evaluation runs on all validation rows because the training file has
labels for every cycle. Official test evaluation first selects each test
engine's final available cycle, then aligns the one-row-per-engine predictions
to `RUL_*.txt`. This avoids comparing official test labels to intermediate
cycles, which would be semantically wrong.

Predictions are clipped to non-negative values before metrics are computed.
The baseline does not clip to `max_rul` by default; exceeding the cap is useful
diagnostic information for a regression baseline.

### 6. `models/artifacts.py` - local experiment artifacts

```python
def create_run_dir(
    artifact_dir: Path,
    run_name: str,
    timestamp: datetime | None = None,
) -> Path:
    """Create and return a timestamped local run directory."""

def save_model(estimator: BaseEstimator, path: Path) -> Path:
    """Persist a fitted sklearn estimator with joblib."""

def save_json(payload: Mapping[str, object], path: Path) -> Path:
    """Write a JSON artifact."""

def save_predictions(df: pd.DataFrame, path: Path) -> Path:
    """Write prediction rows to CSV."""
```

Run outputs are organized as:

```
artifacts/models/baseline/<YYYYMMDD-HHMMSS>/
|-- model.joblib
|-- metrics.json
|-- config.json
|-- validation_predictions.csv
`-- official_test_predictions.csv          # only when official test is run
```

The artifact directory is ignored by git. The repository tracks code and
documentation, not generated model binaries.

### 7. `scripts/train_baseline.py` - thin CLI

```bash
python scripts/train_baseline.py --config configs/default.yaml
```

The script:

1. Loads and validates config.
2. Loads raw training data.
3. Adds capped RUL labels.
4. Splits by engine into train and validation rows.
5. Builds and fits the baseline pipeline.
6. Evaluates validation metrics.
7. Attempts official test evaluation when test data and RUL labels exist.
8. Writes model, metrics, config, and predictions to a run directory.
9. Prints the run directory and primary metrics.

Missing official test files should not prevent validation training; the script
logs that official test evaluation was skipped. Missing training data remains
a hard `FileNotFoundError` from the existing loader.

---

## Testing Strategy

All tests use synthetic DataFrames and temporary directories. No real NASA data
or network access is required.

**`tests/models/test_split.py`**
- split preserves all input rows
- no `engine_id` appears in both train and validation
- fixed seed produces deterministic engine assignment
- invalid tiny datasets raise `ValueError`

**`tests/models/test_metrics.py`**
- RMSE and MAE match hand-computed examples
- PHM08 score matches hand-computed early and late prediction examples
- unequal lengths and NaNs raise `ValueError`
- `regression_metrics()` returns `rmse`, `mae`, and `phm08_score`

**`tests/models/test_baseline.py`**
- `build_baseline_pipeline()` returns named steps `features` and `model`
- default model is `Ridge`
- synthetic training data can fit and predict without NaNs
- unknown model names raise `ValueError`

**`tests/models/test_evaluate.py`**
- `add_rul_column()` delegates to capped RUL semantics
- `split_features_target()` removes only the target column
- `select_last_cycle_per_engine()` returns one row per engine
- official RUL labels align to sorted final-cycle test rows
- row evaluation clips negative predictions before metrics

**`tests/models/test_artifacts.py`**
- run directory is created under the configured artifact directory
- model round-trip with joblib preserves predictions
- JSON artifacts write valid JSON
- prediction CSV artifacts write expected columns

**CLI smoke test**
- runs `scripts/train_baseline.py` against temporary synthetic C-MAPSS-style
  files
- verifies that metrics and model artifacts are written

---

## What This Sub-project Does NOT Include

- Neural networks, LSTMs, transformers, or sequence windowing
- Hyperparameter search beyond a configured Ridge `alpha`
- MLflow, Weights & Biases, or remote experiment tracking
- Feature selection or model explainability reports
- Deployment, inference API, or batch scoring service
- Real-data benchmark claims committed to docs

---

## Sub-project Sequence

| # | Sub-project | Depends on |
|---|-------------|------------|
| 1 | Foundation | -- |
| 2 | EDA | 1 |
| 3 | Feature Engineering | 1 |
| 4 | **Model Training** <- you are here | 1, 3 |
| 5 | Sequence Models | 1, 3, 4 |
| 6 | Inference & Deployment | 1, 4 or 5 |
