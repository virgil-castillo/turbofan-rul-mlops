# Sequence Models: Sub-project 5 Design Spec

**Date:** 2026-05-24
**Sub-project:** 5 of 6
**Depends on:** Sub-project 1 (Foundation), Sub-project 3 (Feature
Engineering), Sub-project 4 (Model Training)
**Status:** Approved

---

## Goal

Build a focused PyTorch GRU baseline for C-MAPSS RUL prediction using
fixed-length sequence windows. The milestone introduces sequence windowing,
train-fitted normalization for raw sensor and operational inputs, a compact GRU
regressor, validation metrics aligned to final-engine test semantics, and local
artifacts for reproducible runs.

The default sequence model is a GRU because it is a real recurrent baseline
with less training and tuning surface than an LSTM, TCN, or transformer. This
sub-project intentionally does not add model serving, deployment contracts,
remote experiment tracking, hyperparameter search, or an architecture
comparison suite.

## Architecture

Sequence code is isolated from the existing tabular baseline. The current
`turbofan.models` utilities remain the source of truth for engine-level
splitting, RUL metrics, capped label creation, official-test final-cycle
semantics, and local artifact helpers. New sequence-specific preprocessing
lives in `turbofan.sequences`, while GRU model and training helpers live under
`turbofan.models`.

Training uses raw operational setting and sensor columns as timestep features:
`op_1`, `op_2`, `op_3`, and `s_1` through `s_21`. A train-fitted normalizer
learns column means and standard deviations from training-engine rows only,
then applies those statistics to validation and test rows. The existing
feature-engineering pipeline is not applied before the GRU by default; the GRU
is responsible for learning temporal patterns from the raw sequence.

Windowing uses fixed-length sliding windows per engine, sorted by cycle. Each
training sample predicts the RUL label at the final timestep of the window.
Validation reports two metric groups:

- `validation_final_window`: one final full window per validation engine. This
  is the primary validation metric because it matches official C-MAPSS test
  label semantics.
- `validation_windows`: all validation sliding windows. This is a diagnostic
  view of sequence-level fit across validation trajectories.

Official test evaluation is optional and follows the same final-window
semantics. If `test_*.txt` and `RUL_*.txt` are present, the script builds one
final full window per eligible test engine and compares those predictions to
the official RUL labels. Missing official test files skip only official test
evaluation; missing training data remains a hard loader error.

Experiment tracking remains local. Each run writes a PyTorch checkpoint,
metrics JSON, resolved config JSON, training history CSV, and prediction CSVs
under `artifacts/models/sequence_gru/<timestamp>/`.

## Tech Stack

Python 3.12, pandas, numpy, PyTorch, scikit-learn-compatible existing helpers,
pytest

---

## File Structure

```
turbofan-rul-mlops/
|-- configs/
|   `-- default.yaml                         # add sequence config
|-- scripts/
|   `-- train_sequence_gru.py                # CLI orchestration
|-- src/turbofan/
|   |-- config/
|   |   `-- schema.py                        # add SequenceConfig
|   |-- models/
|   |   |-- gru.py                           # GRU module
|   |   `-- sequence_training.py             # train/evaluate helpers
|   `-- sequences/
|       |-- __init__.py
|       |-- dataset.py                       # PyTorch Dataset helpers
|       |-- normalize.py                     # train-fitted raw column scaler
|       `-- windowing.py                     # sliding and final windows
|-- tests/
|   |-- config/
|   |   `-- test_schema.py                   # sequence config coverage
|   |-- models/
|   |   |-- test_gru.py
|   |   |-- test_sequence_training.py
|   |   `-- test_train_sequence_gru_cli.py
|   `-- sequences/
|       |-- __init__.py
|       |-- test_dataset.py
|       |-- test_normalize.py
|       `-- test_windowing.py
`-- pyproject.toml                          # add torch runtime dependency
```

---

## Component Designs

### 1. Config additions

Extend the Pydantic config schema with a `SequenceConfig` nested under the
top-level project config:

```python
class SequenceConfig(BaseModel):
    """Configuration for GRU sequence model training.

    Args:
        architecture: Sequence model architecture identifier.
        window_size: Number of cycles per sequence window.
        batch_size: Training batch size.
        hidden_size: GRU hidden state width.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability between GRU layers.
        learning_rate: Adam optimizer learning rate.
        epochs: Maximum training epochs.
        patience: Early-stopping patience in epochs.
        device: Requested torch device.
        artifact_dir: Directory for local sequence run artifacts.
    """

    architecture: Literal["gru"] = "gru"
    window_size: int = Field(default=30, gt=0)
    batch_size: int = Field(default=64, gt=0)
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=1, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    epochs: int = Field(default=50, gt=0)
    patience: int = Field(default=8, gt=0)
    device: Literal["cpu", "cuda"] = "cpu"
    artifact_dir: Path = Path("artifacts/models")
```

`ProjectConfig` gains
`sequence: SequenceConfig = Field(default_factory=SequenceConfig)`.
Keeping this separate from `ModelConfig` preserves the Ridge baseline defaults
and makes sequence-specific settings explicit.

`configs/default.yaml` adds:

```yaml
sequence:
  architecture: gru
  window_size: 30
  batch_size: 64
  hidden_size: 64
  num_layers: 1
  dropout: 0.0
  learning_rate: 0.001
  epochs: 50
  patience: 8
  device: cpu
  artifact_dir: artifacts/models
```

`pyproject.toml` adds `torch` to runtime dependencies because the package will
import it directly.

### 2. `sequences/normalize.py` - raw timestep normalization

```python
class SequenceNormalizer:
    """Normalize raw sequence feature columns with train-fitted statistics."""
```

Responsibilities:

- Use only configured feature columns, defaulting to `op_1`-`op_3` and
  `s_1`-`s_21`.
- Fit means and standard deviations on training-engine rows only.
- Replace zero or missing standard deviations with `1.0`.
- Transform validation and test rows with stored training statistics.
- Return a DataFrame that preserves `engine_id`, `cycle`, optional `rul`, and
  normalized feature columns.

The normalizer is deliberately simple and independent of sklearn. It avoids
leakage while keeping sequence preprocessing easy to test with pandas.

### 3. `sequences/windowing.py` - fixed-length windows

```python
@dataclass(frozen=True)
class WindowedSequences:
    """Container for sequence windows, labels, and row metadata."""

    X: npt.NDArray[np.float32]
    y: npt.NDArray[np.float32]
    metadata: pd.DataFrame


def build_sliding_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window_size: int,
    target_col: str = "rul",
) -> WindowedSequences:
    """Build fixed-length sliding windows per engine."""


def build_final_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window_size: int,
    target_col: str | None = "rul",
) -> WindowedSequences:
    """Build one final full window per eligible engine."""
```

Windowing rules:

- Sort rows by `engine_id` and `cycle` before building windows.
- Never allow a window to cross engine boundaries.
- Use each window's final timestep as the metadata row.
- For labeled data, use the final timestep's RUL as the label.
- Skip engines shorter than `window_size` by default.
- Raise `ValueError` when no eligible windows can be built.

Skipping short engines keeps the first sequence milestone simple. Padding and
masking are deferred until there is a concrete need for early-life predictions
from short histories.

### 4. `sequences/dataset.py` - PyTorch dataset helpers

```python
class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for RUL sequence windows."""
```

The dataset wraps `WindowedSequences` and returns:

- feature tensor shape `(window_size, n_features)`
- target tensor shape `()`

Helper functions build deterministic training and evaluation `DataLoader`
instances. Training loaders use `shuffle=True` by default; evaluation loaders
use `shuffle=False`.
Most preprocessing behavior remains in pandas/numpy modules so the PyTorch
surface stays narrow.

### 5. `models/gru.py` - GRU RUL regressor

```python
class GRURULRegressor(nn.Module):
    """GRU-based RUL regressor for fixed-length sensor windows."""
```

Architecture:

- `nn.GRU` with `batch_first=True`
- final hidden state from the last GRU layer
- linear regression head producing one scalar RUL prediction per window

The model returns shape `(batch_size,)`. A single GRU baseline is the only
architecture in this milestone; LSTM, TCN, and transformer variants are
deferred until this baseline and windowing contract are stable.

### 6. `models/sequence_training.py` - training and evaluation helpers

```python
@dataclass(frozen=True)
class TrainingResult:
    """Best GRU state and epoch-level training history."""

    model: GRURULRegressor
    history: pd.DataFrame
    best_epoch: int
    best_metric: float


def resolve_device(requested: Literal["cpu", "cuda"]) -> torch.device:
    """Resolve configured device to an available torch device."""


def train_gru_model(
    model: GRURULRegressor,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    validation_final_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    validation_windows_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    config: SequenceConfig,
    device: torch.device,
    random_seed: int,
) -> TrainingResult:
    """Train a GRU model with early stopping."""


def predict_windows(
    model: GRURULRegressor,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> npt.NDArray[np.float64]:
    """Predict RUL values for a sequence DataLoader."""
```

Training semantics:

- Default to CPU. Use CUDA only when `device="cuda"` is configured and CUDA is
  available; otherwise raise a clear `ValueError`.
- Set Python, numpy, and torch seeds from `cfg.data.random_seed`.
- Use `MSELoss` and Adam.
- Clip predictions to non-negative values only for metric calculation and
  prediction artifacts, matching the Ridge baseline.
- Evaluate each epoch on `validation_final_window`; use its RMSE for early
  stopping.
- Track `train_loss`, `validation_final_window_rmse`,
  `validation_final_window_mae`, `validation_final_window_phm08_score`, and
  diagnostic `validation_windows_*` metrics in history.
- Restore the best model state before returning.

Early stopping keeps the CLI practical on CPU and avoids adding
hyperparameter-search scope.

### 7. `scripts/train_sequence_gru.py` - thin CLI

```bash
python scripts/train_sequence_gru.py --config configs/default.yaml
```

The script:

1. Loads and validates config.
2. Loads raw training data.
3. Adds capped RUL labels.
4. Splits rows by engine into train and validation sets.
5. Fits `SequenceNormalizer` on training rows.
6. Transforms train, validation, and optional test rows.
7. Builds training sliding windows, validation final windows, and validation
   diagnostic sliding windows.
8. Trains the GRU with early stopping.
9. Evaluates primary and diagnostic validation metrics.
10. Attempts official test evaluation when test data and RUL labels exist.
11. Writes checkpoint, metrics, config, history, and prediction CSV artifacts.
12. Prints the run directory and primary validation metrics.

Run outputs:

```
artifacts/models/sequence_gru/<YYYYMMDD-HHMMSS>/
|-- model.pt
|-- metrics.json
|-- config.json
|-- training_history.csv
|-- validation_final_window_predictions.csv
|-- validation_window_predictions.csv
`-- official_test_predictions.csv          # only when official test is run
```

`model.pt` contains enough metadata for later inference work to understand the
checkpoint without creating a deployment API in this milestone:

- model state dict
- feature column order
- sequence config values needed to rebuild the GRU
- normalizer means and standard deviations
- training subset identifier and random seed

---

## Testing Strategy

All tests use synthetic DataFrames and temporary directories. No real NASA data
or network access is required.

**`tests/config/test_schema.py`**
- sequence config has stable defaults when omitted
- custom sequence config values load from YAML
- invalid architecture, window size, batch size, dropout, learning rate,
  epochs, patience, and device raise validation errors

**`tests/sequences/test_normalize.py`**
- normalizer fits means and standard deviations on training rows
- transform uses stored training statistics for validation/test rows
- zero-variance feature columns use standard deviation `1.0`
- identifier and target columns are preserved
- missing required feature columns raise `KeyError`

**`tests/sequences/test_windowing.py`**
- sliding windows are sorted by engine and cycle
- windows never cross engine boundaries
- each label equals the final timestep RUL
- metadata records the final timestep `engine_id` and `cycle`
- short engines are skipped
- no eligible windows raises `ValueError`
- final-window helper returns one sample per eligible engine

**`tests/sequences/test_dataset.py`**
- dataset length matches the number of windows
- items return expected tensor shapes and dtypes
- dataloader helpers preserve evaluation order and shuffle training data only
  when requested

**`tests/models/test_gru.py`**
- `GRURULRegressor` forward pass returns shape `(batch_size,)`
- model supports configurable hidden size, layer count, and dropout
- invalid constructor values raise `ValueError` where validation is not already
  handled by config

**`tests/models/test_sequence_training.py`**
- device resolution returns CPU by default and raises clearly when CUDA is
  requested but unavailable
- training loop reduces loss or overfits a tiny deterministic synthetic dataset
- early stopping preserves the best state
- prediction helper returns one value per input window
- metrics reuse `regression_metrics` and clip negative predictions before
  evaluation

**CLI smoke tests**
- `scripts/train_sequence_gru.py` trains on temporary C-MAPSS-style files
- expected artifacts are written under `sequence_gru`
- metrics JSON contains `validation_final_window` and `validation_windows`
- official test prediction writes one row per eligible test engine when files
  exist
- official test evaluation is skipped cleanly when test or RUL files are absent

---

## What This Sub-project Does NOT Include

- Deployment, inference API, or batch scoring service
- MLflow, Weights & Biases, model registry, or remote experiment tracking
- Optuna or any other hyperparameter search
- LSTM, TCN, transformer, or architecture comparison framework
- Padding and masking for engines shorter than the configured window size
- GPU-specific test requirements
- Real-data benchmark claims committed to docs
- Replacement of the existing Ridge baseline training script

---

## Sub-project Sequence

| # | Sub-project | Depends on |
|---|-------------|------------|
| 1 | Foundation | -- |
| 2 | EDA | 1 |
| 3 | Feature Engineering | 1 |
| 4 | Model Training | 1, 3 |
| 5 | **Sequence Models** <- you are here | 1, 3, 4 |
| 6 | Inference & Deployment | 1, 4 or 5 |
