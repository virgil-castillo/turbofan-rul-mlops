# Spec — GRU Temporal-Context & Capacity Sweep

**Roadmap frame:** new near-term item to slot in **between** "Completed — Multi-Dataset Training" and "Future — Additional Models" in `docs/roadmap.md`. It pre-empts the LSTM/Transformer work the roadmap defers under *"Current priority is depth on the existing models before breadth."* It also closes a gap explicitly flagged in `docs/feature_sweep_ridge_vs_gru.md`: the rolling-window vs. sequence-window cross was never run.

## 1. Motivation

The unified feature-engineering sweep (PR #3, 2026-05-29) pinned each subset's best GRU feature config (`configs/subsets/*.yaml` → `features.gru`) **while holding `sequence.window_size=45` and `sequence.hidden_size=64` fixed** (`configs/default.yaml`). Three unresolved questions follow:

1. **Denoising hypothesis.** The cross-model report argues the GRU prefers longer rolling means (window ≈ 15) because they act as denoisers the Ridge doesn't get for free. If that is true, increasing the sequence window should reduce the marginal value of long rolling means; if not, the long window is doing something rolling means cannot.
2. **FD002 / FD004 generalization gap.** The multi-condition subsets show the largest validation→test degradation. Capacity (`hidden_size`) and temporal context (`window_size`) were never jointly tuned for them.
3. **Inherited production settings.** The current production GRU artifacts (per `results/training_log_*`) were trained with the *feature-sweep* defaults — not jointly optimized against the feature conclusions.

Resolving these is a prerequisite to claiming the GRU baseline is "stressed enough" to justify adding a second architecture (LSTM, then Transformer).

## 2. Scope

### In scope
- **Stage 0 prerequisite:** left-zero-padding infrastructure for short engines, replacing the current silent-skip behaviour with `pack_padded_sequence` so all engines participate in training, validation, and test evaluation regardless of `window_size`.
- Three-stage sweep (Stage 0 infrastructure → Stage 1 temporal-context cross → Stage 2 capacity sweep) over **sequence-level** parameters (`window_size`, `hidden_size`) crossed with the **narrowed** feature space identified by prior reports.
- All four subsets: FD001, FD002, FD003, FD004.
- Validation-only ranking (engine-level split). Official test evaluation is **only** run on the final selected configurations, and only to refresh the production benchmark table — not to rank sweep runs.
- Job scripts under `jobs/slurm/` that let the owner run Stage 1, Stage 2, and final selected-config retraining on the cluster.
- Post-run analysis/reporting once owner-produced CSVs and artifacts are available.

### Out of scope (explicitly deferred)
- SHAP / interpretability — deferred until after this sweep so it targets a stable model contract.
- LSTM, Transformer, tree-based models — roadmap "Future — Additional Models".
- Lag-family features — both prior reports show they are neutral-to-harmful; do not re-introduce.
- MLflow / registry / CI work — roadmap "Future — MLOps Infrastructure".
- Multi-layer GRU / dropout sweep — current GRU is single-layer by design (see roadmap "Design Decisions Worth Preserving"). Add only if Stage 2 indicates capacity saturation.

## 3. Experiment design

### Stage 0 — Short-engine padding infrastructure

**Problem.** `_build_windows()` in `src/turbofan/sequences/windowing.py` silently skips engines shorter than `window_size` (`if len(group) < window_size: continue`). When the sweep tests `window_size=60`, some FD002/FD004 engines vanish from training *and* evaluation, making scores across window sizes incomparable. The inference predictor (`src/turbofan/inference/predictors.py`) already identifies short engines but either errors or skips them.

**Solution.** Left-zero-pad short engines at windowing time and use `torch.nn.utils.rnn.pack_padded_sequence` so the GRU processes only the real timesteps.

#### 3.0.1 WindowedSequences changes

Add a `lengths` field to `WindowedSequences`:

```python
@dataclass(frozen=True)
class WindowedSequences:
    X: npt.NDArray[np.float32]       # (n_windows, window_size, n_features)
    y: npt.NDArray[np.float32]       # (n_windows,)
    metadata: pd.DataFrame           # gains a "padded" bool column
    lengths: npt.NDArray[np.int64]   # (n_windows,) actual timesteps per window
```

In `_build_windows()`, when `len(group) < window_size`:
- Build one left-zero-padded window: `np.zeros((window_size, n_features))` with real data right-aligned.
- Record `actual_length = len(group)` in `lengths`. Full-length windows get `length = window_size`.
- Set `metadata["padded"] = True` for that window.
- For sliding mode, short engines produce exactly one padded window (can't slide). For `final_only` mode, same — one padded window.

**Why pad at windowing time, not at DataLoader collation time.** The existing pipeline expects uniform `(n_windows, window_size, n_features)` tensors in `WindowedSequences.X`. Padding at windowing time preserves this contract. Collation-time padding would require variable-length storage and a custom `collate_fn` — a larger refactor for negligible memory savings, since only a handful of engines per subset are short.

#### 3.0.2 GRU model changes

`GRURULRegressor.forward()` gains an optional `lengths` parameter:

```python
def forward(self, X: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
```

When `lengths` is provided:
1. Call `pack_padded_sequence(X, lengths.cpu(), batch_first=True, enforce_sorted=False)`.
2. Pass the packed sequence to `self.gru`.
3. The GRU's returned `hidden` state already reflects only the real timesteps — `hidden[-1]` is the hidden state after the last *real* timestep, not the last *padded* position.

When `lengths` is `None`: unchanged behaviour, backward compatible.

**Why `enforce_sorted=False`.** Sorting batches by length would require a custom `collate_fn` and undo-sort after the GRU. `enforce_sorted=False` handles unsorted batches with a small performance cost that is negligible given batch sizes of 64.

#### 3.0.3 Dataset and DataLoader changes

`SequenceDataset` stores an optional `_lengths` tensor. When present, `__getitem__` returns a 3-tuple `(features, target, length)`.

`build_sequence_loader` detects whether `WindowedSequences.lengths` exists (it always will after Stage 0, but backward compat is preserved for any code that builds `WindowedSequences` without it).

The batch type becomes:
```python
type SequenceBatch = tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Training loop functions (`_train_one_epoch`, `_evaluate_loader`, `_predict_windows_and_targets`) unpack the optional third element and pass it through to `model.forward()`.

#### 3.0.4 Sweep CSV reporting columns

Every sweep CSV row gains three columns:
- `n_engines_total` — total unique engines in the validation split.
- `n_engines_padded` — engines shorter than `window_size` that were zero-padded.
- `n_engines_full` — engines that produced full-length windows.

Derived from `WindowedSequences.metadata["padded"]`. The training log `extra` block also records these for the training split.

**Why include padded engines in all metrics.** `pack_padded_sequence` makes padded engines legitimate GRU inputs — the hidden state evolves only over real timesteps. Excluding them from scoring would re-introduce the population-thinning problem. Including them and *reporting the counts* lets the reader judge comparability. If padding hurts predictions, the score reflects it naturally.

#### 3.0.5 Inference predictor alignment

`GRUPredictor._filter_short_engines()`:
- `allow_partial=True`: pad short engines instead of skipping. Still emit a per-engine warning.
- `allow_partial=False`: unchanged (error on short engines — strict mode means the caller expects all engines to be long enough).

#### 3.0.6 Roadmap update

Update `docs/roadmap.md` "Short engines are skipped, not padded" to reflect the new behaviour: "Short engines are left-zero-padded and processed with `pack_padded_sequence`."

### Stage 1 — Temporal-context cross

Per-subset grid. Fixed: `hidden_size=64`, `learning_rate=0.001`, `batch_size=64`, `epochs/patience` as in `configs/default.yaml`.

| Dimension          | Values per subset                                                                                 |
|--------------------|---------------------------------------------------------------------------------------------------|
| `feature_set`      | subset's current best (per `configs/subsets/<x>.yaml`) **and** `raw` as a control                  |
| rolling `window`   | subset's current best ±1 neighbour each side (e.g. FD001 → `{10, 15, 20}`; FD004 → `{5, 10, 15}`)  |
| `sequence.window_size` | `{30, 45, 60}`                                                                                |

`raw` rows have no rolling window dimension, so the grid per subset is roughly `1 × 3 + 3 × 3 = 12` runs × 4 subsets = **~48 GRU trainings**.

Purpose: isolate whether sequence context shifts the rolling-window preference. Read-outs:
- If best rolling window shrinks (or `raw` closes the gap) as `window_size` grows → denoising hypothesis confirmed.
- If best rolling window is stable across `window_size` → rolling mean is doing something the sequence window doesn't replicate.

### Stage 2 — Capacity sweep

Inputs: the **top 2 configurations per subset** from Stage 1 (by validation PHM08, then RMSE as tiebreak).

| Dimension          | Values                                                  |
|--------------------|---------------------------------------------------------|
| `hidden_size`      | `{32, 64, 128}`                                         |
| `learning_rate`    | current value (`0.001`) and one lower (`0.0003`)        |
| `batch_size`       | unchanged unless Stage 1 shows training instability      |

Grid per subset: `2 × 3 × 2 = 12` runs × 4 subsets = **~48 GRU trainings**.

Purpose: check whether FD002 / FD004 specifically benefit from more capacity once temporal context is right-sized.

### Total budget

≈ 96 GRU trainings. With current default of 50 epochs and early stopping at patience 8, expect each run to terminate well before the cap. Ridge is not included — Ridge's best feature config is already pinned; re-tuning Ridge here would conflate axes.

## 4. Ranking & selection

- **Primary metric:** engine-level validation PHM08 score, lower is better.
- **Secondary metric:** validation RMSE, used as tiebreak and reported alongside.
- **Selection per subset:** one production GRU config that wins on the primary metric in either stage, with a documented rationale if a higher-capacity Stage 2 winner is preferred over a simpler Stage 1 winner.
- **Anti-leakage discipline (preserve):** sweep CSVs do **not** include any official test-set columns. Test-set evaluation runs only for the four selected production artifacts, mirroring the discipline established in the roadmap's "Sequence Modeling" / "Unified Feature-Engineering Sweep" entries.

## 5. Required code surface

This is "what must exist," not "how to build it" — implementation choices are deferred to the plan.

### Stage 0 code surface
- `WindowedSequences` gains a `lengths: npt.NDArray[np.int64]` field and `metadata` gains a `padded` boolean column.
- `_build_windows()` pads short engines instead of skipping them; full-length windows get `length = window_size`.
- `GRURULRegressor.forward()` accepts optional `lengths` and uses `pack_padded_sequence` when provided.
- `SequenceDataset` and `build_sequence_loader` propagate `lengths` through the 3-tuple batch path.
- `_train_one_epoch`, `_evaluate_loader`, and `_predict_windows_and_targets` unpack and forward `lengths`.
- `GRUPredictor._filter_short_engines()` pads instead of skipping in `allow_partial=True` mode.
- All existing tests updated; new tests for padded windowing, packed GRU forward pass, and round-trip through the loader.

### Sweep code surface
- Sweep harness must accept **sequence-side dimensions** (`window_size`, `hidden_size`, optionally `learning_rate`) in addition to the existing `feature_set` / `windows` / `lag_steps` axes. Today `src/turbofan/experiments/feature_sweep.py` reads those from `cfg.sequence` only.
- Result rows must include the swept sequence/capacity parameters **and** the Stage 0 engine-count columns (`n_engines_total`, `n_engines_padded`, `n_engines_full`) so Stage 1 and Stage 2 CSVs can be analysed together. Current row schema (`_evaluate_gru_spec`) only carries `feature_set`, `windows`, `lag_steps`, `best_epoch`, `n_features`, and metrics.
- Output paths should distinguish stages and subsets, e.g. `results/gru_temporal_sweep_stage1_<subset>.csv` and `..._stage2_<subset>.csv`.
- SLURM scripts under `jobs/slurm/` should follow the existing conventions in `run_feature_sweep_gru_all_datasets.sh`: configurable `PROJECT_DIR`, `CONDA_HOME`, `CONDA_ENV`, `DEVICE`, stage-specific grid environment variables, `results/` and `outputs/logs/` creation, and explicit echoing of resolved parameters before execution.
- Provide a multi-dataset Stage 1 runner and a Stage 2 runner. The Stage 2 runner may either consume Stage 1 CSV paths/top-k selectors directly or accept explicit top-config arguments, but it must not require hand-editing Python code between subsets.
- Training-log append (`append_training_log`) must continue to capture the swept hyperparameters in the `extra` block so the persistent log remains the source of truth.

The plan will decide whether to extend the existing `turbofan-sweep-features` CLI or add a sibling `turbofan-sweep-gru-temporal` CLI.

## 6. Deliverables

0. **Stage 0:** Padding infrastructure landed and tested — all existing tests pass, new tests cover padded windowing, packed GRU forward, and the loader round-trip. No sweep runs required; this is pure infrastructure.
1. Sweep harness support for the Stage 1 and Stage 2 grids. The harness writes CSVs under `results/` when run, and each row includes `n_engines_total`, `n_engines_padded`, `n_engines_full`.
2. Runnable SLURM job scripts under `jobs/slurm/`, following the existing script style:
   - Stage 1 all-subset temporal-context sweep job.
   - Stage 2 capacity sweep job that consumes Stage 1 results or explicit top configs.
   - Optional selected-config retrain/benchmark-refresh job for after results are reviewed.
   The scripts are the handoff artifact; the owner will run training.
3. A short post-run report `docs/gru_temporal_capacity_sweep.md`, produced after owner-run CSVs are available, that:
   - states which of the three Stage-1 hypotheses (denoising / context-substitution / capacity) the data supports per subset;
   - ranks the final selected config per subset with the validation metric and the runner-up;
   - explicitly notes whether the FD002 / FD004 validation→test gap narrowed.
   Must follow the existing "grounded reports" rule: cite only this repo's sweep CSVs, EDA notebooks, and prior reports; no external knowledge.
4. After owner-run results are reviewed, update `configs/subsets/<x>.yaml gru` blocks for selected configs and refresh the official-test benchmark table in `README.md` / `docs/feature_sweep_*` for retrained production GRUs only.
5. New roadmap entry under a "Completed — GRU Temporal-Capacity Sweep" section, mirroring the structure of the existing "Completed — Unified Feature-Engineering Sweep" block, after the sweep has actually been run and analyzed.

## 7. Success criteria

- **Stage 0:** All existing tests pass after padding refactor. New tests verify: (a) short engines produce one left-zero-padded window with correct `lengths` value, (b) `GRURULRegressor.forward(X, lengths)` produces identical output to `forward(X)` when all lengths equal `window_size`, (c) the loader round-trips 3-tuple batches correctly.
- Stage 1 and Stage 2 job scripts can be invoked without code edits and are parameterized enough for local dry runs or cluster submission.
- Owner-run sweep CSVs contain valid metric rows (`best_epoch >= 1`, finite PHM08) for the completed jobs.
- At least one of the three Stage-1 hypotheses is answered for each subset, with the evidence cited.
- For each subset, the **selected** GRU configuration's validation PHM08 is ≤ the current production GRU's validation PHM08 (no regressions). If a config is selected that is *worse* on validation but materially simpler, the report must justify it.
- The production benchmark refresh either improves FD002 / FD004 test RMSE or, if it doesn't, the report explicitly states that the temporal/capacity axes did not close that gap — feeding the next roadmap decision (LSTM vs. interpretability).

## 8. Risks & non-goals

- **Risk:** Stage 1's `raw` arm may be the strongest on some subset, which would imply re-litigating the feature-sweep conclusion. Mitigation: report it honestly; do not silently drop the `raw` arm. The point of including it is exactly to test the denoising story.
- **Risk (resolved by Stage 0):** Sequence window 60 may exceed the shortest test engines on FD002/FD004. Previously these engines were silently skipped. **Mitigation:** Stage 0 replaces skipping with left-zero-padding and `pack_padded_sequence`, so all engines participate in every window-size configuration. Sweep CSVs report `n_engines_padded` per row so the reader can assess whether padded engines disproportionately affect the score.
- **Non-goal:** This sweep is not an MLOps-infrastructure deliverable. The existing CSV + `training_log` pattern is sufficient. MLflow / registry remain deferred per the roadmap.

## 9. Proposed roadmap entry (verbatim, to slot in before "Future — Additional Models")

> ## Future — GRU Temporal-Context & Capacity Sweep
>
> Cross sequence `window_size`, `hidden_size`, and the narrowed rolling-feature
> candidates to validate the denoising hypothesis from
> `docs/feature_sweep_ridge_vs_gru.md` and to address the FD002 / FD004
> validation→test generalization gap, before adding a second architecture (LSTM).
> Three stages: short-engine padding infrastructure (Stage 0), temporal-context
> cross at fixed capacity (Stage 1), then capacity sweep on the top configs per
> subset (Stage 2). Ranking is validation-only; official test evaluation is
> reserved for the selected production retrains.
