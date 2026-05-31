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
- Two-stage joint sweep over **sequence-level** parameters (`window_size`, `hidden_size`) crossed with the **narrowed** feature space identified by prior reports.
- All four subsets: FD001, FD002, FD003, FD004.
- Validation-only ranking (engine-level split). Official test evaluation is **only** run on the final selected configurations, and only to refresh the production benchmark table — not to rank sweep runs.
- One follow-on retrain of production artifacts from the selected configs and a refresh of the README/benchmark table.

### Out of scope (explicitly deferred)
- SHAP / interpretability — deferred until after this sweep so it targets a stable model contract.
- LSTM, Transformer, tree-based models — roadmap "Future — Additional Models".
- Lag-family features — both prior reports show they are neutral-to-harmful; do not re-introduce.
- MLflow / registry / CI work — roadmap "Future — MLOps Infrastructure".
- Multi-layer GRU / dropout sweep — current GRU is single-layer by design (see roadmap "Design Decisions Worth Preserving"). Add only if Stage 2 indicates capacity saturation.

## 3. Experiment design

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

- Sweep harness must accept **sequence-side dimensions** (`window_size`, `hidden_size`, optionally `learning_rate`) in addition to the existing `feature_set` / `windows` / `lag_steps` axes. Today `src/turbofan/experiments/feature_sweep.py` reads those from `cfg.sequence` only.
- Result rows must include the swept sequence/capacity parameters so Stage 1 and Stage 2 CSVs can be analysed together. Current row schema (`_evaluate_gru_spec`) only carries `feature_set`, `windows`, `lag_steps`, `best_epoch`, `n_features`, and metrics.
- Output paths should distinguish stages and subsets, e.g. `results/gru_temporal_sweep_stage1_<subset>.csv` and `..._stage2_<subset>.csv`.
- A multi-dataset runner equivalent to `jobs/slurm/run_feature_sweep_gru_all_datasets.sh` for each stage.
- Training-log append (`append_training_log`) must continue to capture the swept hyperparameters in the `extra` block so the persistent log remains the source of truth.

The plan will decide whether to extend the existing `turbofan-sweep-features` CLI or add a sibling `turbofan-sweep-gru-temporal` CLI.

## 6. Deliverables

1. Sweep CSVs for both stages, all four subsets, under `results/`.
2. A short report `docs/gru_temporal_capacity_sweep.md` that:
   - states which of the three Stage-1 hypotheses (denoising / context-substitution / capacity) the data supports per subset;
   - ranks the final selected config per subset with the validation metric and the runner-up;
   - explicitly notes whether the FD002 / FD004 validation→test gap narrowed.
   Must follow the existing "grounded reports" rule: cite only this repo's sweep CSVs, EDA notebooks, and prior reports; no external knowledge.
3. Re-trained production GRU artifacts for any subset whose selected config differs from the current `configs/subsets/<x>.yaml gru` block; update those blocks in-place.
4. Refresh of the official-test benchmark table in `README.md` / `docs/feature_sweep_*` for the new production GRUs only.
5. New roadmap entry under a "Completed — GRU Temporal-Capacity Sweep" section, mirroring the structure of the existing "Completed — Unified Feature-Engineering Sweep" block.

## 7. Success criteria

- All ~96 sweep runs complete and produce valid metric rows (`best_epoch >= 1`, finite PHM08).
- At least one of the three Stage-1 hypotheses is answered for each subset, with the evidence cited.
- For each subset, the **selected** GRU configuration's validation PHM08 is ≤ the current production GRU's validation PHM08 (no regressions). If a config is selected that is *worse* on validation but materially simpler, the report must justify it.
- The production benchmark refresh either improves FD002 / FD004 test RMSE or, if it doesn't, the report explicitly states that the temporal/capacity axes did not close that gap — feeding the next roadmap decision (LSTM vs. interpretability).

## 8. Risks & non-goals

- **Risk:** Stage 1's `raw` arm may be the strongest on some subset, which would imply re-litigating the feature-sweep conclusion. Mitigation: report it honestly; do not silently drop the `raw` arm. The point of including it is exactly to test the denoising story.
- **Risk:** Sequence window 60 may exceed the shortest test engines on FD002/FD004 (engines shorter than the window are currently skipped — see roadmap "Short engines are skipped, not padded"). Mitigation: report skipped-engine counts in the sweep CSV alongside metrics so a high score on a thinned set isn't mistaken for a real win.
- **Non-goal:** This sweep is not an MLOps-infrastructure deliverable. The existing CSV + `training_log` pattern is sufficient. MLflow / registry remain deferred per the roadmap.

## 9. Proposed roadmap entry (verbatim, to slot in before "Future — Additional Models")

> ## Future — GRU Temporal-Context & Capacity Sweep
>
> Cross sequence `window_size`, `hidden_size`, and the narrowed rolling-feature
> candidates to validate the denoising hypothesis from
> `docs/feature_sweep_ridge_vs_gru.md` and to address the FD002 / FD004
> validation→test generalization gap, before adding a second architecture (LSTM).
> Two stages: temporal-context cross at fixed capacity, then capacity sweep on
> the top configs per subset. Ranking is validation-only; official test
> evaluation is reserved for the selected production retrains.
