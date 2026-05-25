# GRU Hyperparameter Sweep: Design Spec

**Date:** 2026-05-26
**Depends on:** Sub-project 5 (Sequence Models)
**Status:** Draft

---

## Goal

Sweep `window_size`, `hidden_size`, and `learning_rate` for the GRU sequence
model on the same engine-level validation split used by the tabular baseline.
Produce a ranked comparison table sorted by PHM08 score, matching the output
format established by `compare_baseline_features.py`.

Two deliverables: a Python sweep script and a SLURM shell script.

## Motivation

The GRU sequence model is implemented but has only been trained with default
hyperparameters. The tabular baseline went through a structured feature and
alpha sweep before being finalized. The GRU needs the same treatment to
establish whether it beats the baseline (`rolling_w10`, PHM08 = 159,969) and
which hyperparameter region works best.

## Sweep Grid

| Parameter | Values | Rationale |
|---|---|---|
| `window_size` | 15, 20, 30, 45 | Temporal context length. Analogous to rolling window in the baseline. Too small loses degradation signal; too large pads short engines. |
| `hidden_size` | 32, 64, 128 | GRU capacity. Underfitting vs overfitting on C-MAPSS FD001 (~100 training engines). |
| `learning_rate` | 1e-3, 5e-4, 1e-4 | Adam optimizer step size. Controls convergence speed and stability. |

**Held fixed from config:** `num_layers=1`, `dropout=0.0`, `batch_size=64`,
`epochs=50`, `patience=8`.

Total: 4 x 3 x 3 = **36 runs**, executed sequentially on a single device.

### Why these parameters and not others

- **`num_layers`**: Stacking GRU layers rarely helps on short sequences
  (~200 cycles). Keeping it at 1 avoids a 2x grid expansion for minimal
  expected gain.
- **`dropout`**: Only applied between GRU layers when `num_layers > 1`.
  With `num_layers=1`, the GRU constructor sets dropout to 0 regardless of
  the config value. Not meaningful to sweep.
- **`batch_size`**: Affects training speed and noise but is not a
  model-quality knob on this dataset size. 64 is standard.
- **`epochs` / `patience`**: Early stopping handles convergence. 50 epochs
  with patience 8 is generous enough for all learning rates in the grid.

## Architecture

### Data Flow

```
load_raw_train
  -> add_rul_column(max_rul=125)
  -> split_by_engine(test_size=0.2, seed=42)
  -> SequenceNormalizer.fit_transform(train_df)  [fit once, reuse]
  -> SequenceNormalizer.transform(val_df)

For each (window_size, hidden_size, learning_rate):
  -> build_sliding_windows(train_normalized, window_size)
  -> build_final_windows(val_normalized, window_size)
  -> build_sequence_loader(train_windows, batch_size=64, shuffle=True)
  -> build_sequence_loader(val_final_windows, batch_size=64, shuffle=False)
  -> build_sequence_loader(val_all_windows, batch_size=64, shuffle=False)
  -> GRURULRegressor(input_size, hidden_size, num_layers=1, dropout=0.0)
  -> train_gru_model(model, loaders, config_override, device, seed=42)
  -> evaluate final-window validation metrics
  -> append row to results
```

The normalizer is fit once because it learns column means and standard
deviations from training rows, which do not change across specs. Windows must
be rebuilt per spec because `window_size` varies.

### Script: `scripts/sweep_sequence_gru.py`

**CLI arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config` | Path | `configs/default.yaml` | Project config path |
| `--window-sizes` | int list | `15 20 30 45` | Window sizes to sweep |
| `--hidden-sizes` | int list | `32 64 128` | Hidden sizes to sweep |
| `--learning-rates` | float list | `1e-3 5e-4 1e-4` | Learning rates to sweep |
| `--device` | str choice | `cpu` | Torch device (`cpu` or `cuda`), validated by argparse choices |
| `--output` | Path | None | Optional CSV output path |

**Testable function:**

```python
def run_gru_sweep(
    config_path: Path,
    window_sizes: list[int],
    hidden_sizes: list[int],
    learning_rates: list[float],
    device: str,
) -> pd.DataFrame:
```

Returns a DataFrame sorted by `phm08_score` (ascending = best first).

**Output columns:** `window_size`, `hidden_size`, `learning_rate`,
`best_epoch`, `rmse`, `mae`, `phm08_score`.

**Config override approach:** For each spec, copy the loaded
`SequenceConfig` using `model_copy(update={...})`, overriding `window_size`,
`hidden_size`, and `learning_rate`. All other values (epochs, patience,
batch_size, etc.) come from the loaded config.

**Incremental CSV writes:** After each completed run, append the result row
to the output CSV (if `--output` is provided). If the sweep crashes mid-run,
completed results are preserved. The final sorted DataFrame is written at the
end, overwriting the incremental file.

**Console output:** Print each completed row as it finishes (run N/36,
spec values, PHM08 score) so the user can monitor progress. Print the full
sorted table at the end.

### Script: `scripts/run_gru_sweep_slurm.sh`

Mirrors `run_baseline_feature_comparison_slurm.sh`:

```bash
#SBATCH -J rul_gru_sweep
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/gru_sweep.%j.out
```

**Environment variable overrides:**

| Variable | Default | Description |
|---|---|---|
| `CONFIG` | `configs/default.yaml` | Config path |
| `WINDOW_SIZES` | `15 20 30 45` | Space-separated window sizes |
| `HIDDEN_SIZES` | `32 64 128` | Space-separated hidden sizes |
| `LEARNING_RATES` | `1e-3 5e-4 1e-4` | Space-separated learning rates |
| `DEVICE` | `cuda` | Torch device (defaults to `cuda` on SLURM, unlike `cpu` default in Python script) |
| `OUTPUT` | `artifacts/gru_sweep_${SLURM_JOB_ID:-local}.csv` | Results CSV path |

Same conda activation pattern as the existing SLURM script (supports both
Linux and Windows conda layouts).

### Tests: `tests/models/test_sweep_sequence_gru.py`

Following the pattern of `test_compare_baseline_features_cli.py`:

1. **`test_gru_sweep_returns_expected_rows`** — Calls `run_gru_sweep` with
   2 window sizes, 2 hidden sizes, 1 learning rate on tiny synthetic data
   (4 engines, 6 cycles, 2 epochs, patience 2). Asserts 4 result rows with
   correct columns and expected spec combinations.

2. **`test_gru_sweep_validates_inputs`** — Invalid window sizes (0 or
   negative), empty hidden sizes, non-positive learning rates raise
   `ValueError`.

3. **`test_sweep_sequence_gru_cli_writes_csv`** — Subprocess smoke test
   with tiny config: runs the CLI, checks CSV output exists and is sorted
   by PHM08 ascending.

Tests use tiny grids and 2 epochs to keep runtime under a few seconds. All
data is synthetic (no download required).

## File Map

| File | Action | Responsibility |
|---|---|---|
| `scripts/sweep_sequence_gru.py` | Create | Sweep CLI and `run_gru_sweep` function |
| `scripts/run_gru_sweep_slurm.sh` | Create | SLURM job script for HPC execution |
| `tests/models/test_sweep_sequence_gru.py` | Create | Unit and CLI smoke tests |

No modifications to existing source files. The sweep script imports existing
library code and composes it.

## Out of Scope

- Multi-GPU parallelism or SLURM array jobs
- Sweeping `num_layers`, `dropout`, `batch_size`, or `epochs`
- Official test evaluation within the sweep (validation-only for speed)
- Artifact persistence per run (metrics CSV is the only output)
- MLflow or any remote experiment tracker
