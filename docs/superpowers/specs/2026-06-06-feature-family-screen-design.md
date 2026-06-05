# Feature-Family Screen for GRU and LSTM (scaled redo)

**Date:** 2026-06-06
**Status:** Design finalized, ready to implement.
**Supersedes:** `2026-06-04-rolling-feature-family-sweep-design.md` (the two-stage
individual-then-combinations design). That design is retired for two reasons: (1) every
sweep predating commit `3e5afc8` ("scale engineered pipeline output") fed the models
engineered features on **mixed magnitudes** — raw-normalized channels (~[0, 1]) sitting
beside unscaled slope/delta/std — so all prior cross-family comparisons are confounded
and are discarded; (2) the question is collapsed from a power-set search to a single
greedy **"raw + one family"** screen, which is what the literature and standard practice
actually support for sequence models.

## The question

Now that the pipeline standard-scales its engineered output (`pipeline.py:165`), does
adding any **single** engineered family on top of `raw` improve a GRU or LSTM on this
project's data — and by more than seed noise? This is a one-step greedy / forward-
selection screen, not a subset enumeration.

Why this shape (settled in discussion, recorded here):
- **No power set.** 8 families ⇒ 255 subsets before crossing windows/arch/subset.
  Greedy add-one is O(n), answers the same keep/drop question, and matches what papers
  report (classical models get the engineered bundle; sequence models are reported on
  raw windows ± light denoising). The honest "which family matters" is
  **add-one-column ablation against held-out RMSE** — which is exactly this screen.

## Factor grid

Each run trains one model once on one cell and records engine-window validation
RMSE/MAE. `raw` is always present; each non-raw family is added singly.

| Factor | Levels | Count |
|---|---|---|
| Feature config | `[raw]`, `[raw,rolling_mean]`, `[raw,rolling_std]`, `[raw,rolling_min]`, `[raw,rolling_max]`, `[raw,rolling_slope]`, `[raw,rolling_delta]`, `[raw,lag]` | 8 |
| Rolling window | 5, 20 (applies to the 6 `rolling_*` configs only) | 2 |
| Lag step | 1, 5 (applies to `[raw,lag]` only) | 2 |
| Sequence (model) window | 30, 60 | 2 |
| Architecture | GRU, LSTM | 2 |
| Subset | FD001, FD002, FD003, FD004 | 4 |

**Window/lag apply per-config, so cells per (arch, subset, sequence-window):**
`[raw]` = 1; six `rolling_*` × 2 windows = 12; `[raw,lag]` × 2 lag steps = 2 ⇒ **15**.

**Main run count = 15 × 2 (seq) × 2 (arch) × 4 (subset) = 240.**

**Held fixed** (representational/optimization capacity — does not interact with which
feature *type* wins): `hidden_size=64`, `num_layers=1`, `dropout=0.0`,
`learning_rate=1e-3`, `batch_size=64`, `epochs=50`, `patience=8`, `seed=42`.
`max_rul=125`, `test_size=0.2`, split `random_seed=42`. These are deliberately decoupled
from the per-subset *tuned* values in `configs/subsets/*.yaml` (which vary `window_size`
and `lr`): comparability **across cells** matters here, not per-subset absolute best.

**Inherited per subset** (must not be flattened — correctness depends on it):
`sensor_cols_to_drop` and `n_modes` from each `configs/subsets/fd00X.yaml`. FD002/FD004
carry `n_modes=6`; per-mode normalization is mandatory there (EDA: raw correlations
vanish otherwise). The harness reads the subset config for these and overrides only the
swept factors.

`[raw,lag]` is included on purpose: the prior "lag is harmful / non-convergence
(early-stop epoch 1–3)" finding is the prime suspect for a **pure scaling artifact**
(tiny-magnitude unscaled differences wrecking input conditioning). Re-running it scaled
either revives lag or confirms its death cleanly.

## No code changes outside the harness

`FeatureEngineer.feature_families` and `build_feature_pipeline(feature_families=…)`
already accept a **list** and concatenate family outputs; the pipeline already appends
`StandardScaler`. So `[raw, rolling_slope]` works today — **no composable-feature-set
refactor, no new transform, no model/training change.** The only new code is the harness.

## Harness (the deliverable) — `src/turbofan/experiments/feature_family_screen.py`

A resumable runner that reuses the existing training stack and writes one CSV row per
cell. It reuses, unchanged: `load_raw_train` → `add_rul_column` → `split_by_engine` →
`build_feature_pipeline` (per-cell families/windows/lag_steps) → `build_sliding_windows`
→ `build_sequence_loader` → `build_sequence_model(arch, …)` → `train_sequence_model`,
then reads `result.best_metric` (val RMSE) and `result.best_epoch`.

**Deliberately NOT done per cell** (this is a screen, not production training): no
official-test evaluation, no `registry.log_and_register`, no checkpoint persistence. MLflow
logging to the `turbofan-sweeps` experiment is optional; **the CSV is authoritative.**

### Resume mechanism (the one thing explicitly requested)

- Output: `results/feature_family_screen_{arch}_{subset}.csv`, one row per cell, appended
  and flushed **immediately** on completion.
- **Cell identity** = the tuple `(feature_config, rolling_window, lag_step,
  sequence_window, seed)`. Inapplicable factors use a fixed sentinel (empty string) so the
  key is well-defined: `[raw]` → `rolling_window="" , lag_step=""`; `rolling_*` →
  `lag_step=""`; `[raw,lag]` → `rolling_window=""`.
- On start, the harness reads any existing CSV for that `(arch, subset)`, builds the set
  of completed keys, and **skips** cells already present. A partial/killed/walltime-
  preempted job is simply re-invoked and continues. Because `seed` is part of the key,
  adding seeds later re-runs only the new seed cells.
- Crash-safety: append-and-flush per row (not a single write at the end), so an
  interrupted run never loses completed cells and never leaves a half-written row that
  the skip-set would misread (write the full row atomically or guard parsing).

### CSV columns

`architecture, subset, feature_config, rolling_window, lag_step, sequence_window,
hidden_size, learning_rate, seed, n_features, n_train_windows, n_val_windows,
best_epoch, val_rmse, val_mae, training_duration_seconds`.

`feature_config` is a stable label, e.g. `raw`, `raw+rolling_slope`, `raw+lag`. The
`architecture` and `subset` columns let all CSVs stack into one comparison frame.

### CLI

`src/turbofan/cli/run_feature_screen.py` (argparse): `--subsets FD001 FD002 …`,
`--architectures gru lstm`, `--seeds 42`, plus the grid levels with the defaults above
(`--rolling-windows 5 20`, `--lag-steps 1 5`, `--sequence-windows 30 60`). Iterates cells,
honors the resume skip-set, logs progress. A templated `jobs/slurm/` script wraps it; the
resumable CSV makes any walltime overrun a non-event (re-queue, it skips done cells).

## Noise floor — committed scope, keyed by the same resume mechanism

The 240 runs are **uninterpretable without a noise band** (a "win" smaller than seed
spread is not real), so the band is part of this build — not a follow-up. Total
committed scope = **240 main + 16 band = 256 runs.** It is broken out separately only
because it reuses the same seed-keyed resume rather than expanding the grid:

- Re-run `[raw,rolling_mean]` (incumbent) and `[raw,rolling_delta]` (top rate candidate)
  at the most-favorable cell for engineered features — `rolling_window=5`,
  `sequence_window=30` — on a fast and a slow pair, **(GRU, FD001)** and **(LSTM, FD004)**,
  seeds **43, 44, 45, 46** (42 is already in the main grid).
- 2 configs × 2 pairs × 4 seeds = **16 extra runs**. The per-cell std of `val_rmse` across
  seeds is the single ±band; a family counts as beating `raw`/`rolling_mean` only beyond it.

Run by invoking the harness with `--seeds 43 44 45 46` restricted to those cells; the
skip-set leaves the main grid untouched.

## Decision rule and report

Per (architecture, subset, window facet), rank configs by validation RMSE against the
`[raw]` baseline and the `[raw,rolling_mean]` incumbent. A family "earns inclusion" only
if it beats the incumbent by more than the seed-noise band. Report
`docs/feature_family_screen_report.md`: ranks from the CSVs only; cites the external
literature (sequence models rarely need engineered features; rate features —
slope/delta — are the literature's RUL-relevant ones) explicitly as **outside
knowledge, never as measured fact**; states per subset/arch whether any family clears the
band; and explicitly reports the `[raw,lag]` scaled result as the resolution of the
prior (confounded) "lag is harmful" finding.

## Estimate

~¾ min/run on single-condition subsets (FD001/FD003), ~3 min/run on multi-condition
(FD002/FD004), LSTM ~30% heavier than GRU. 240 + 16 runs ≈ a few CPU-hours total,
decomposable into per-subset SLURM jobs under the 4-hour node limit; the resumable CSV
covers any overrun.

## Non-goals (unchanged intent)

No power set / multi-family unions (greedy add-one only; revisit unions only if ≥2
families clear the band). No FFT /
entropy / autocorrelation families. No capacity re-tuning (`hidden_size`/`lr` fixed). No
official-test eval or model registration inside the screen. No Ridge re-run (feature
preferences do not transfer across model classes).
