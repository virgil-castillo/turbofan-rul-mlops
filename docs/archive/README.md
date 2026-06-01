# Archived analysis artifacts

Reports and result CSVs kept for provenance after being superseded. Nothing here
reflects the current best models or recommendations — use the live reports in
[`../`](../) instead. Files were moved (not regenerated) on **2026-06-01**; git
history preserves their original `docs/` and `results/` locations.

| Archived file | What it was | Superseded by | Why |
| :--- | :--- | :--- | :--- |
| `gru_sweep_report.md` | Original FD001-only GRU architecture sweep (window × hidden × lr) | [`../gru_capacity_sweep_report.md`](../gru_capacity_sweep_report.md) + Stage 1 temporal sweep | Pre-unified-pipeline, single dataset, old CSV schema (no `feature_set`/`fd_subset`/padding). The two-stage sweep covers all four subsets with the short-engine padding fix. |
| `../../results/archive/gru_sweep.csv` | Source data for the above | `results/stage1_temporal_sweep_*.csv`, `results/stage2_capacity_sweep_*.csv` | Same lineage. |
| `feature_sweep_gru_report.md` | GRU feature-engineering sweep at fixed `hidden_size=64`, `window_size=45` | [`../gru_capacity_sweep_report.md`](../gru_capacity_sweep_report.md) (headline RMSE + window/capacity selection) | Stage 1 re-explored the sequence window and Stage 2 the capacity per subset. The **feature-mechanism findings** (rolling-window direction, lag harm, role of `raw`) remain valid and are still cited by [`../feature_sweep_ridge_vs_gru.md`](../feature_sweep_ridge_vs_gru.md). |
| `../../results/archive/feature_sweep_gru_fd00{1-4}.csv` | Source data for the GRU feature sweep | `results/stage1_temporal_sweep_*.csv` (window/feature selection) | Same lineage; mechanism tables in the report still derive from these. |

The corresponding data CSVs live under [`../../results/archive/`](../../results/archive/).

Live (non-archived) sweep analysis:
[Ridge feature sweep](../feature_sweep_ridge_report.md) ·
[Ridge vs GRU](../feature_sweep_ridge_vs_gru.md) ·
[GRU two-stage capacity sweep](../gru_capacity_sweep_report.md)
