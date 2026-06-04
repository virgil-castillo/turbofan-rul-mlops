# Rolling-Feature-Family Sweep for GRU and LSTM

**Date:** 2026-06-04
**Status:** Design approved. Stage 1 committed; Stage 2 documented but deferred.

## The question

Does any engineered rolling-feature family — rolling **std**, **slope**, or
**delta** — help the GRU or the LSTM more than the incumbent rolling **mean**, on
this project's data? Every sequence-model feature config selected to date
(`rolling_mean` for the GRU) was chosen from a search space that contained no
rolling statistic other than the mean, so the project cannot currently answer this.

## Background (what we already know)

Two findings frame the expected answer and are kept distinct by source:

- **This project's own data (grounded).** The archived GRU feature sweep
  (`docs/archive/feature_sweep_gru_report.md`) established that `rolling_mean`
  beats `raw` by ~4–7% validation RMSE on every subset. So `raw < rolling_mean` is
  already settled here — the open question is whether the *new* families beat
  `rolling_mean`, which is therefore the baseline to beat (not `raw`). The two-stage
  temporal sweep (`results/stage1_temporal_sweep_*.csv`) tested sequence windows
  {30, 45, 60} and found longer better (FD001: 60 best, 30 worst); nothing below 30
  was tested.
- **External literature (not grounded in this repo; cited as outside knowledge).**
  A comparative C-MAPSS study ([arXiv:2604.27234](https://arxiv.org/html/2604.27234))
  ran its LSTM/CNN on raw sliding-window sequences and reported handcrafted
  statistical features were "unnecessary" for the sequence models while they carried
  the *classical* models. A feature-importance study
  ([PMC10439180](https://pmc.ncbi.nlm.nih.gov/articles/PMC10439180/)) lists the
  standard engineered set as mean, std, min, max, last value, delta (first−last),
  and linear slope, singling out the degradation-*rate* features (slope/delta) as
  RUL-relevant. The canonical sequence-window value is 30 cycles, used because it is
  the largest window fitting the shortest engine record (31 cycles) without padding
  ([RUL-datasets](https://krokotsch.eu/rul-datasets/api/rul_datasets/reader/cmapss/),
  [LSTM guide](https://medium.com/@mihaitimoficiuc/predicting-jet-engine-failures-with-nasas-c-mapss-dataset-and-lstm-a-practical-guide-to-85b9513ea9ed));
  it is a practical ceiling, not a sweep-proven optimum over shorter windows.

The sweep is therefore **confirmatory by design**: the most likely outcome is that
the new families tie `rolling_mean` for the sequence models, confirming the
literature claim on this project's data. The design gives the strongest candidates
(the rate features) their best shot — a short sequence window and short rolling
horizon — before concluding it.

## Stage 1 — individual families (committed scope)

### The factorial

Each "run" trains one model once on one combination and records validation RMSE/MAE.
The factors varied:

| Factor | Levels | Count |
|---|---|---|
| Feature family | `raw`, `rolling_mean`, `rolling_std`, `rolling_slope`, `rolling_delta` | 5 |
| Rolling-feature window | 5, 20 (cycles each rolling statistic spans; `raw` exempt) | 2 |
| Sequence window | 30, 60 (cycles per model input; 30 = literature floor, 60 = tuned long) | 2 |
| Architecture | GRU, LSTM (identical capacity) | 2 |
| Subset | FD001, FD002, FD003, FD004 | 4 |

**Held fixed** (do not interact with which feature *type* wins): `hidden_size = 64`
(unanimous across selected configs), `learning_rate = 1e-3` (the default and modal
selection), and `seed = 42` for the main sweep (the seed-noise study below varies
it). `max_rul = 125` and the engine-level validation split (`test_size = 0.2`,
`random_seed = 42`) are inherited unchanged from the subset configs.

**Run count = 144** (`raw` carries no rolling-window dimension, so it is not crossed
with the rolling-window factor).

**Why both windows are factors, not fixed.** Both are temporal hyperparameters that
change which feature wins (the archived sweep proved rolling-window direction
matters; the rate features should help most at short sequence windows, where the
model has the least context to derive rates itself). Fixing either at one value
would make the feature winners conditional on an arbitrary choice. `hidden_size` and
`learning_rate` are representational/optimization capacity and do not interact with
feature *type*, so they stay fixed. That is the stopping rule for what gets crossed.

### Schema + `FeatureEngineer` changes

Add three singleton families to the `FeatureSet` Literal
(`src/turbofan/features/engineering.py`) and `FeatureSetName`
(`src/turbofan/config/schema.py`), each with a transform branch and a
`_compute_output_cols` entry, following the existing per-engine, no-boundary-crossing
groupby pattern:

- `rolling_std` → columns `{s}_rstd_{w}` = per-engine `rolling(w, min_periods=1).std()`;
  first-cycle `NaN` filled with `0.0` (matches existing `_rolling_stats` std handling).
- `rolling_slope` → columns `{s}_rslope_{w}` = per-engine rolling least-squares slope
  of the sensor against an integer cycle index `0..w-1` within the window
  (closed form `Σ(t−t̄)(x−x̄) / Σ(t−t̄)²`; a length-1 window has zero denominator →
  slope `0.0`). `min_periods=1`.
- `rolling_delta` → columns `{s}_rdelta_{w}` = per-engine `x[t] − x[t−(w−1)]`; early
  cycles with insufficient history backfill to a difference of `0.0`.

`rolling_delta` is **distinct from the retired `lag` family** (documented in the
docstring): it is an *un-normalized* windowed difference keyed to the rolling-window
span `w−1`, whereas `lag` was a *rolling-mean-normalized* difference at an arbitrary
`lag_steps` offset. The lag family was proven harmful and is not revived.

### Sweep harness (rebuild, resumable)

The original feature-sweep harness was deleted (only
`jobs/slurm/run_gru_selected_retrain.sh` survives), so this is a rebuild. New module
under `src/turbofan/experiments/`:

- Given a subset config and the factor grid, for each cell: resolve the fixed
  capacity, build the pipeline via `build_feature_pipeline(feature_set=family,
  windows=[rolling_window], …)`, build loaders via `build_sequence_loader` with the
  cell's `sequence_window`, train via `train_sequence_model`, and record
  engine-level validation RMSE/MAE. Reuses existing training/inference code
  unchanged — no new model or training logic.
- **Resumable and checkpointed.** Each completed cell's row is appended to the
  output CSV immediately; on restart the harness skips cells already present. A
  cell's identity for skip purposes is the full tuple (`feature_set`,
  `rolling_window`, `sequence_window`, `seed`) within the job's (`architecture`,
  `subset`) CSV — so the seed-noise re-runs (same family/window, different seed) are
  distinct cells and are not skipped. This is the one requirement the HPC 4-hour
  wall-clock limit imposes (see "Compute"): a job that hits the limit is simply
  re-queued and continues.
- Writes `results/feature_family_sweep_stage1_{arch}_{subset}.csv`, one row per cell,
  columns at least: `architecture`, `subset`, `feature_set`, `rolling_window`,
  `sequence_window`, `hidden_size`, `learning_rate`, `seed`, `n_features`,
  `n_engines_total`, `n_engines_padded`, `best_epoch`, `val_rmse`, `val_mae`. The
  `architecture` column lets GRU and LSTM rows stack into one comparison.
- Optionally logs each run to the existing `turbofan-sweeps` MLflow experiment; the
  CSV is the authoritative artifact.

### Analysis (no formal statistical model — by design)

The analysis is **descriptive, plus a small noise study** — not an inferential
model. A formal ANOVA / regression with significance tests would require replicating
every cell across many seeds (144 × n) to estimate within-cell variance validly, and
would produce p-values this project has no audience for. The decision we need —
"which families clearly beat `rolling_mean`, to carry to Stage 2?" — is made by
inspection against a measured noise floor, not a hypothesis test.

1. **Descriptive factorial.** Ranked validation-RMSE tables and interaction plots
   (RMSE vs feature family), faceted by `sequence_window` and `rolling_window`,
   **separately per (architecture, subset)** — the subsets are different
   populations, not replicates, so they are never pooled. The key read: does
   `rolling_slope` / `rolling_delta` overtake `rolling_mean` specifically in the
   short-sequence / short-rolling facet? Rank flips across facets are the
   interaction, read directly.
2. **Seed-noise band.** Re-run two families — `rolling_mean` (the baseline) and
   `rolling_slope` (the strongest new candidate) — at the short-sequence /
   short-rolling cell (`sequence_window = 30`, `rolling_window = 5`) for two
   (architecture, subset) pairs spanning a fast and a slow subset, **(GRU, FD001)**
   and **(LSTM, FD004)**, across seeds **42, 43, 44, 45, 46**. That is 2 families × 2
   (arch, subset) × 5 seeds = **20 extra runs, not 144 × 5**. Training noise is
   approximately cell-independent, so the per-cell spread (standard deviation of
   `val_rmse` across seeds) taken as a single "real difference" band (±X RMSE) is a
   sufficient yardstick for the whole sweep.
3. **The band is the Stage-1 → Stage-2 winner gate.** A family advances to Stage 2
   only if it beats `rolling_mean` by more than the band. This is the principled,
   *derived* threshold that replaces the pre-registered margin we deliberately did
   not commit to without data.

### Report

`docs/feature_family_sweep_report.md`: ranks the families per (architecture, subset,
window facet) from the CSVs only; cites the two external papers explicitly as outside
knowledge, never as this project's measured fact; states whether the result confirms
or contradicts the "sequence models don't need engineered features" claim; and ends
with the documented Stage-1 → Stage-2 advancement decision and its rationale,
grounded in the Stage 1 CSV and the seed-noise band.

### Testing (TDD, per CLAUDE.md)

- `tests/features/test_engineering.py`: each new family produces the expected output
  column names and shape; `rolling_std` first-cycle `NaN`→`0.0`; `rolling_slope`
  length-1 window → `0.0` and a known-slope fixture returns that slope;
  `rolling_delta` early-cycle backfill → `0.0`; per-engine grouping never crosses an
  engine boundary.
- Schema tests: new `FeatureSetName` members validate; `for_model` resolves them.
- Harness test on synthetic data: produces one row per cell with the expected columns
  and correct `architecture` tag for both `gru` and `lstm`; resume logic skips cells
  already present in the CSV.
- `mypy --strict` and `ruff check` clean; Google-style docstrings with
  `Args:`/`Returns:`/`Raises:`.

## Stage 2 — combinations of winners (documented, deferred — NOT committed scope)

**Intent.** Once Stage 1 has run, sweep the multi-family *unions* (pairs, triples, …)
of the families that cleared the seed-noise gate, at the same factor grid, to test
whether individually-good features stack. Output
`results/feature_family_sweep_stage2_{arch}_{subset}.csv`, same schema, with
`feature_set` recording the union members.

**Deferred parameter (the single placeholder):**

```
STAGE2_ADVANCING_FAMILIES = [raw, rolling_mean]   # PLACEHOLDER / DUMMY
# Replaced with the actual Stage 1 winners (those beating rolling_mean by more
# than the seed-noise band), plus the documented rationale, once Stage 1 CSVs
# exist. Every other aspect of Stage 2 above is fully specified.
```

**Why deferred, not built now.** Stage 2 needs a mechanism the project does not yet
have: `feature_set` must accept a *set* of families and concatenate their column
outputs (a "composable" feature set), rather than today's single-value enum. Building
that now is speculative generality — the most likely Stage 1 outcome is that nothing
clears the gate, leaving the mechanism as dead code. Stage 1 is a complete, valuable
unit on its own. The advancement decision already requires inspecting Stage 1
results, so writing the short Stage 2 increment then costs little and is paid only if
needed. **When Stage 2 is taken up, the composable list-valued `feature_set` refactor
(concat, dedup, stable ordering, single-value back-compat, and its tests) comes into
scope with it.**

**Early-exit.** If no family beats `rolling_mean` by more than the seed-noise band,
`STAGE2_ADVANCING_FAMILIES` stays `rolling_mean`-only, Stage 2 is skipped, and the
confirmatory result — "no engineered rolling family improves the sequence models on
this data" — is the project's finding.

## Compute and HPC

Runs are small (single-layer, `hidden_size=64`, early-stopped) and independent, so
the HPC **4-hour wall-clock limit is an orchestration constraint, not a rigor one**.
Estimated unit cost is ~¾ min/run on the single-condition subsets (FD001/FD003, ~100
engines) and ~3 min/run on the multi-condition subsets (FD002/FD004, ~250 engines),
with LSTM ~30% heavier than GRU. The harness's resumable/checkpointed CSV (above)
makes any job re-queueable.

## Execution (SLURM)

The 144 main runs are decomposed into **2 SLURM jobs** — the three lighter-to-mid
subsets together, FD004 alone — each running both architectures:

| Job | Subsets | Models | runs | est. wall |
|----:|---------|--------|-----:|----------:|
| 1 | FD001 + FD002 + FD003 | GRU + LSTM | 108 | ~3.0 h |
| 2 | FD004 | GRU + LSTM | 36 | ~2.5 h |

Both fit under the 4-hour node limit on the current estimate. **Job 1 is the tight
one (~3.0 h):** FD002 is a *heavy* multi-condition subset (~250 engines, ~3 min/run)
like FD004 — not a light one like FD001/FD003 — so it alone contributes ~2 h of Job
1's ~3 h. If the per-run cost on FD002 runs materially above ~3 min, Job 1 can exceed
4 h; the resumable CSV (above) covers this — the job is re-queued and resumes by
skipping completed cells, at the cost of one extra queue cycle.

A single templated SLURM script under `jobs/slurm/` takes a subset list and runs both
architectures over it, submitted twice (Job 1 = `FD001 FD002 FD003`, Job 2 =
`FD004`). Each completed run appends to
`results/feature_family_sweep_stage1_{arch}_{subset}.csv` immediately, so a preempted
or walled job resumes by skipping done cells. The ~20 seed-noise runs (seeds 42–46)
are appended to Job 1 (the GRU/FD001 cell) and Job 2 (the LSTM/FD004 cell).

No pre-run timing is needed: submit both jobs, and react only if one comes back
incomplete. If Job 1 hits the wall before finishing, re-queue it — it resumes by
skipping completed cells — or split FD002 into its own job. The resumable CSV makes
an overrun a non-event, so the estimates above are guidance, not a gate.

## Non-goals (YAGNI)

- No Ridge pre-filter and no Ridge re-run — feature preferences do not transfer
  across model classes.
- No capacity re-tuning and no separate LSTM capacity sweep — `hidden_size` and
  `learning_rate` are held fixed; both architectures run at identical capacity.
- No `min` / `max` / `mean_std` / 4-stat-bundle families — the weakest candidates and
  the most redundant; their exclusion is part of keeping the matrix bounded.
- No FFT / entropy / autocorrelation families — weak fit for C-MAPSS's slow
  thermodynamic sensors.
- No formal statistical model (ANOVA / p-values) at this stage — see Analysis.
- No Stage 2 implementation and no composable `feature_set` refactor until Stage 1
  justifies it.

## Honest expectation

`raw < rolling_mean` is already established on this project's data. The new families
most likely tie `rolling_mean` in the long-sequence / long-rolling cells; their best
shot — and the genuinely open result — is the short-sequence / short-rolling cell,
where the model has the least temporal context to derive rates itself. The design
gives them that shot. A family clearly beating `rolling_mean` beyond the seed-noise
band is the surprise that makes Stage 2 worth its own spec.
