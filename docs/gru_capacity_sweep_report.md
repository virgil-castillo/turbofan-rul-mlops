# GRU Stage 2 Capacity Sweep Report

Source files:

- `results/baselines/stage2_capacity_sweep_fd001.csv`
- `results/baselines/stage2_capacity_sweep_fd002.csv`
- `results/baselines/stage2_capacity_sweep_fd003.csv`
- `results/baselines/stage2_capacity_sweep_fd004.csv`
- `results/baselines/training_log.jsonl` (training durations and best epoch; last 48 entries)

Ranking is by validation RMSE (lower is better). The PHM08 score is computed only
on the official test set and is not used to rank validation runs.

> **Data provenance.** The Stage 1/Stage 2 data was generated 2026-05-31, before
> commit `3e5afc8` (2026-06-05) added standard-scaling of the engineered pipeline
> output, and before the post-scaling
> [feature-family screen](feature_family_screen_report.md) re-selected the
> production sequence configurations (now `raw+rolling_slope` on FD001/FD002,
> `raw` on FD003, and `raw+rolling_mean` on FD004 — see the README results). The
> within-sweep comparisons below (hidden width, learning rate) describe the
> pre-scaling pipeline; the screen fixed `hidden_size=64` rather than re-testing
> it, so the width choice rests on this sweep's pre-scaling evidence. The
> "selected configuration per subset" table records what this sweep chose at the
> time and is superseded as a production recommendation.

## Sweep design

Stage 2 takes the temporal-context choices fixed by Stage 1 and varies only model
capacity. For each subset the driver (`jobs/slurm/run_gru_capacity_sweep_stage2.sh`,
`TOP_K=2`) promotes the **top 2** Stage 1 configurations (by validation RMSE) and
crosses each with:

- Hidden sizes: **32, 64, 128**
- Learning rates: **0.001, 0.0003**

That is 2 bases × 3 hidden sizes × 2 learning rates = **12 runs per subset, 48
runs total**. The feature set, rolling `windows`, and `sequence_window_size`
columns in each row are inherited from the promoted Stage 1 base and are *not*
swept in Stage 2.

### Stage 1 bases promoted into Stage 2

These are the two top-RMSE rows of each `results/baselines/stage1_temporal_sweep_<fd>.csv`.
Note the feature family already differs by subset (Stage 1's own grid was
subset-specific), so the bases are not directly comparable across subsets:

| subset | n_modes | base | feature_set | windows | seq window | n_features |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: |
| FD001 | 1 | A | rolling_mean | (10,) | 60 | 15 |
| FD001 | 1 | B | rolling_mean | (20,) | 60 | 15 |
| FD002 | 6 | A | rolling_mean | (10,) | 60 | 16 |
| FD002 | 6 | B | rolling_mean | (15,) | 60 | 16 |
| FD003 | 1 | A | raw | — | 60 | 12 |
| FD003 | 1 | B | rolling_mean | (20,) | 30 | 12 |
| FD004 | 6 | A | raw_plus_rolling_mean | (10,) | 45 | 28 |
| FD004 | 6 | B | raw_plus_rolling_mean | (15,) | 45 | 28 |

### Fixed training configuration

Everything outside the two swept axes (and the inherited Stage 1 base) is held at
the `configs/subsets/<fd>.yaml` → `default.yaml` values for every run:

| parameter | value | source |
| :--- | :--- | :--- |
| architecture | single-layer GRU (`num_layers=1`) | `sequence` |
| dropout | 0.0 | `sequence` |
| batch_size | 64 | `sequence` |
| max epochs | 50 | `sequence.epochs` |
| early-stop patience | 8 | `sequence.patience` |
| optimizer | Adam (lr swept above) | training loop |
| max_rul (label clip) | 125 | `data` |
| validation split | engine-level, `test_size=0.2`, `random_seed=42` | `data` |
| operating-mode norm. | `n_modes=1` (FD001/FD003), `n_modes=6` (FD002/FD004) | per-subset `features` |
| device | CUDA | SLURM driver |

`best_epoch` in the result tables is the epoch of lowest validation RMSE under
patience-8 early stopping (max 50 epochs); the reported RMSE/MAE are that epoch's
validation metrics. There is **no** learning-rate schedule, weight decay, or
gradient clipping in the current training loop.

## Headline finding

`hidden_size = 64` is the best capacity on **all four** subsets, by both average
and single-best validation RMSE. 32 units underfit and 128 units give no gain (and
are usually slightly worse), so 64 is the recommended width everywhere.

| subset | best RMSE (hid=64) | avg RMSE hid=32 | avg RMSE hid=64 | avg RMSE hid=128 |
| :--- | ---: | ---: | ---: | ---: |
| FD001 | 10.186 | 10.521 | **10.430** | 10.667 |
| FD002 | 12.782 | 13.297 | **12.973** | 13.277 |
| FD003 | 10.067 | 10.883 | **10.637** | 10.670 |
| FD004 | 14.730 | 15.498 | **14.890** | 15.307 |

Learning rate is the weaker axis. The single best run uses `0.001` on FD002/FD003/FD004
and `0.0003` on FD001; on the per-subset averages the two rates are within ~0.3 RMSE
of each other. There is no learning rate that is best everywhere.

## Selected configuration per subset

Top-ranked run for each subset (this sweep's selection; superseded as a
production recommendation — see the provenance note):

| subset | feature_set | windows | seq window | hidden | lr | best epoch | RMSE | MAE | train time |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FD001 | rolling_mean | (20,) | 60 | 64 | 0.0003 | 18 | 10.186 | 7.605 | 39.8s |
| FD002 | rolling_mean | (10,) | 60 | 64 | 0.001 | 6 | 12.782 | 9.370 | 58.8s |
| FD003 | raw | — | 60 | 64 | 0.001 | 11 | 10.067 | 7.040 | 38.0s |
| FD004 | raw_plus_rolling_mean | (10,) | 45 | 64 | 0.001 | 2 | 14.730 | 9.486 | 53.1s |

## Per-subset results

### FD001 (single operating condition)

| rank | feature_set | windows | hidden | lr | epoch | RMSE | MAE | train time |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | rolling_mean | (20,) | 64 | 0.0003 | 18 | 10.186 | 7.605 | 39.8s |
| 2 | rolling_mean | (20,) | 32 | 0.0003 | 18 | 10.285 | 7.812 | 41.9s |
| 3 | rolling_mean | (10,) | 64 | 0.0003 | 21 | 10.396 | 7.778 | 45.4s |
| 4 | rolling_mean | (20,) | 32 | 0.001 | 9 | 10.479 | 7.996 | 24.9s |

The whole top tier uses `lr=0.0003`. FD001 is the one subset where `0.0003` clearly
wins (top three runs) and where 128 units are the *worst* capacity on average
(10.667) — the simplest subset gets nothing from extra width. Spread across all 12
runs is narrow (10.186–10.864), so capacity matters little here once the window is
fixed. Rank 4 is a leaner ~25s alternative giving up ~0.29 RMSE if fast iteration
is wanted.

### FD002 (six operating conditions)

| rank | feature_set | windows | hidden | lr | epoch | RMSE | MAE | train time |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | rolling_mean | (10,) | 64 | 0.001 | 6 | 12.782 | 9.370 | 58.8s |
| 2 | rolling_mean | (15,) | 64 | 0.0003 | 22 | 12.984 | 9.523 | 128.7s |
| 3 | rolling_mean | (15,) | 64 | 0.001 | 6 | 13.039 | 9.690 | 65.3s |
| 4 | rolling_mean | (15,) | 128 | 0.0003 | 6 | 13.066 | 9.669 | 64.0s |

The three best runs are all 64-unit. The rank-1 run is both the best and one of the
faster runs (58.8s, converges by epoch 6). Rank 2 reaches a comparable score but
costs more than twice the time (128.7s, epoch 22), so it is not worth promoting over
rank 1.

### FD003 (single operating condition, two fault modes)

| rank | feature_set | windows | hidden | lr | epoch | RMSE | MAE | train time |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | raw | — | 64 | 0.001 | 11 | 10.067 | 7.040 | 38.0s |
| 2 | raw | — | 128 | 0.0003 | 23 | 10.242 | 7.421 | 72.3s |
| 3 | rolling_mean | (20,) @ seq30 | 64 | 0.001 | 13 | 10.471 | 7.425 | 41.6s |
| 4 | raw | — | 128 | 0.001 | 6 | 10.480 | 7.651 | 32.5s |

This is the only subset where a **raw** Stage 1 base outranks the rolling-mean base:
the top two `raw` runs (seq window 60) beat every `rolling_mean (20,) @ seq30` run.
FD003 also produces the best RMSE in the whole sweep (10.067). `lr=0.001` leads here.

### FD004 (six operating conditions, two fault modes)

| rank | feature_set | windows | hidden | lr | epoch | RMSE | MAE | train time |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | raw_plus_rolling_mean | (10,) | 64 | 0.001 | 2 | 14.730 | 9.486 | 53.1s |
| 2 | raw_plus_rolling_mean | (15,) | 64 | 0.001 | 2 | 14.731 | 9.283 | 55.0s |
| 3 | raw_plus_rolling_mean | (10,) | 64 | 0.0003 | 6 | 14.980 | 9.321 | 72.4s |
| 4 | raw_plus_rolling_mean | (15,) | 128 | 0.001 | 2 | 15.057 | 9.486 | 59.4s |

The hardest subset: highest RMSE and the 64-unit advantage is largest here (avg
14.890 vs 15.498 for 32 and 15.307 for 128). See the caveat below — the best runs
early-stop at epoch 1–2.

## Caveats and interpretation flags

- **FD004 early-stopping (flag).** The top FD004 runs reach their best validation
  epoch at epoch 2, and one rank-6 run at epoch 1. The model is converging almost
  immediately and then regressing; the recorded RMSE is essentially a near-initial
  checkpoint. This pattern carried over from the Stage 1 / feature-sweep observation
  that `raw_plus_rolling_mean` on FD004 stops very early. Treat the FD004 number as a
  weak-convergence result, not a well-trained optimum — worth a learning-rate /
  patience follow-up before trusting it in production.
- **Window/feature choices are not re-tested here.** Each subset only explores
  capacity on the *two* Stage 1 winners. If a Stage 1 config ranked third or lower
  would have responded differently to wider hidden sizes, this sweep cannot see it.
- **Cross-sweep comparisons.** RMSE values here are lower than the earlier unified
  feature-sweep GRU numbers (e.g. FD001 ~10.9 in `docs/archive/feature_sweep_gru_report.md`
  vs 10.186 here). The two sweeps differ in `sequence_window_size` and stopping, so
  the improvement is plausible but not a controlled comparison — do not read it as a
  pure capacity gain.
- All numbers are validation-split metrics. No official test-set / PHM08 ranking is
  implied.

## Follow-up status

The per-subset selected configurations above were retrained
(`jobs/slurm/run_gru_selected_retrain.sh`, 2026-06-01) and served as the
production GRU models until the post-scaling
[feature-family screen](feature_family_screen_report.md) re-selected the sequence
configurations (see the provenance note). The FD004 epoch-2 early-stop question
was overtaken by that re-selection rather than answered within this sweep.
