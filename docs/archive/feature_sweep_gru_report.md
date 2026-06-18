> **Archived / superseded (2026-06-01).** This sweep fixed the GRU at
> `hidden_size=64`, `window_size=45` and varied only feature engineering. Its
> headline best-RMSE figures and window/capacity selection are superseded by the
> two-stage temporal + capacity sweep
> ([`../gru_capacity_sweep_report.md`](../gru_capacity_sweep_report.md)), which
> re-explored the sequence window and model capacity per subset with the
> short-engine padding fix. The **feature-family mechanism findings** here (rolling
> window direction, lag-family harm, the role of `raw`) are still cited by
> [`../feature_sweep_ridge_vs_gru.md`](../feature_sweep_ridge_vs_gru.md) and remain
> valid at the fixed capacity tested — but do not use this file's RMSE values as
> the current best GRU numbers.

# GRU Feature-Engineering Sweep

Analysis of `results/baselines/archive/feature_sweep_gru_fd00{1-4}.csv`.

## Method

Each sweep trained one GRU (single layer, `hidden_size=64`, sequence
`window_size=45`, `max_rul=125`, early stopping with `patience=8`) per feature
configuration, evaluated on validation windows from an engine-level split. Feature
families and grids match the Ridge sweep: `raw`, `rolling_mean`,
`raw_plus_rolling_mean`, `lag`, `raw_plus_lag`; windows `2,4,6,8,10,15,20`; lag
steps `2,4,8`.

Two window concepts coexist:

- the sequence window (45 cycles, fixed) — how many consecutive cycles the GRU sees;
- the rolling-feature window (the `windows` column) — per-channel smoothing applied
  before sequencing.

The "window" findings below concern the rolling-feature window. The `best_epoch`
column is diagnostic: working feature sets converge over ≈5–17 epochs; broken sets
early-stop at epoch 1–3 (no validation improvement). Ranking uses validation RMSE
(MAE reported alongside); the PHM08 score is reserved for official-test evaluation.

## Best configuration per dataset

| Dataset | Best feature set | Window | n_features | best_epoch | RMSE | MAE |
|---|---|---:|---:|---:|---:|---:|
| FD001 | `rolling_mean` | 20 | 15 | 12 | 10.82 | 7.92 |
| FD002 | `rolling_mean` | 20 | 16 | 3 | 13.06 | 9.77 |
| FD003 | `rolling_mean` | 15 | 12 | 9 | 10.57 | 7.66 |
| FD004 | `raw_plus_rolling_mean` | 10 | 28 | 2 | 14.73 | 9.49 |

Plain `rolling_mean` wins on the three cleaner subsets; the multi-mode FD004
prefers `raw_plus_rolling_mean`. The optimal window is large (10–20).

## Finding 1 — Feature-family ranking on every dataset

Sorting by RMSE gives a consistent tiering:

1. `rolling_mean` ≈ `raw_plus_rolling_mean` (top, interleaved)
2. `raw` (close behind)
3. `raw_plus_lag` (a large drop)
4. `lag` alone (catastrophic)

FD001 spread, showing how close the top tier and `raw` are and how far the lag
families fall:

| Feature set (FD001, best window/step) | best_epoch | RMSE | vs. best |
|---|---:|---:|---:|
| `rolling_mean` (w20) | 12 | 10.82 | — |
| `raw_plus_rolling_mean` (w15) | 5 | 11.00 | +1.6% |
| `raw` | 6 | 11.36 | +5.0% |
| `raw_plus_lag` (step 4) | 22 | 21.88 | +102% |
| `lag` (step 8) | 1 | 28.93 | +167% |

## Finding 2 — Large rolling windows are preferred (opposite of Ridge)

For plain `rolling_mean`, RMSE improves monotonically as the window grows, reaching
its minimum at 20 (FD001):

| Window | 2 | 4 | 6 | 8 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RMSE | 11.47 | 11.32 | 11.28 | 11.25 | 11.10 | 10.91 | 10.82 |

The pattern repeats across subsets (best plain-rolling windows: 20, 20, 15, and 10
for FD004). A GRU already models temporal dependencies across its 45-cycle input
[1, 2], so the rolling mean's role is denoising rather than trend extraction.
A wide moving-average window suppresses high-frequency sensor noise [3] while the
sequence still supplies the dynamics, so wider windows help — the reverse of the
short-window responsiveness a memoryless linear model needs.

## Finding 3 — `raw` alone is already competitive

`raw` lands within ~5% of the best on the cleaner subsets and only widens its gap
on FD004:

| Dataset | `raw` RMSE | best RMSE | gap |
|---|---:|---:|---:|
| FD001 | 11.36 | 10.82 | +5.0% |
| FD002 | 14.04 | 13.06 | +7.6% |
| FD003 | 10.97 | 10.57 | +3.8% |
| FD004 | 15.50 | 14.73 | +5.2% |

A gated recurrent network learns temporal features internally [1, 2], so it does
not depend on engineered rolling features the way a per-row linear model does;
rolling mainly contributes a denoising bonus.

## Finding 4 — `lag` features actively harm training, even with `raw` present

This is sharper than for Ridge, where `raw_plus_lag` ≈ `raw`. For the GRU,
`raw_plus_lag` is far worse than `raw`:

| Dataset | `raw` RMSE | `raw_plus_lag` best RMSE | `lag` best RMSE |
|---|---:|---:|---:|
| FD001 | 11.36 | 21.88 | 28.93 |
| FD002 | 14.04 | 21.13 | 28.70 |
| FD003 | 10.97 | 16.91 | 19.76 |
| FD004 | 15.50 | 20.28 | 25.65 |

The `best_epoch` column identifies the mechanism: lag and `raw_plus_lag` runs
early-stop at epoch 1–3 (e.g. FD002 lag/`raw_plus_lag` runs at best_epoch 1–2),
meaning validation loss never improved past the first epoch [5]. The lag-difference
channels are on a different, near-zero-centered scale than the normalized levels;
poorly conditioned input scaling slows and destabilizes gradient-based training
[4], so the network fails to converge. The lag family is not merely redundant for
the GRU — it is damaging.

## Finding 5 — Retaining `raw` helps only under multiple operating modes

Whether `raw_plus_rolling_mean` beats plain `rolling_mean` splits by
operating-mode count:

| Dataset | modes | `rolling_mean` best RMSE | `raw_plus_rolling_mean` best RMSE | winner |
|---|---:|---:|---:|---|
| FD001 | 1 | 10.82 | 11.00 | rolling_mean |
| FD002 | 6 | 13.06 | 13.72 | rolling_mean |
| FD003 | 1 | 10.57 | 10.57 | rolling_mean |
| FD004 | 6 | 15.10 | 14.73 | raw_plus_rolling_mean |

On the single-condition subsets the raw channels are redundant and marginally hurt.
On FD004 — the hardest multi-mode subset — retaining the normalized raw value
beside the smoothed value supplies mode-residual detail a rolling mean washes out,
improving RMSE ~2.4%. FD002 (also 6 modes) does not show this, so the effect is
real but not guaranteed by mode count alone.

## Per-dataset notes (EDA context and sweep behavior)

### FD001 — 100 engines, 1 condition (EDA: 20,631 rows; lifetimes 128/199/362)
`n_modes=1`, 15 sensors retained. Cleanest result: best RMSE 10.82 at
`rolling_mean` w20; `raw` only 5% behind; lag families catastrophic.

### FD002 — 260 engines, 6 conditions (EDA: 53,759 rows; lifetimes 128/199/378)
EDA: raw correlations vanish until per-mode normalization (`n_modes=6`). Best RMSE
13.06 at `rolling_mean` w20. Top runs early-stop at epoch 3 — multi-mode data is
harder to fit and the GRU converges quickly then plateaus.

### FD003 — 100 engines, 1 condition (EDA: 24,720 rows; lifetimes 145/220/525)
Only 12 sensors retained (most aggressive pruning). Best RMSE 10.57 — the best of
all four subsets — at `rolling_mean` w15, despite longer engines than FD001.

### FD004 — 249 engines, 6 conditions (EDA: 61,249 rows; lifetimes 128/234/543)
Largest, longest-lived, hardest. The only subset where `raw_plus_rolling_mean`
wins, at a smaller window (10) and best_epoch 2. Highest RMSE (14.73).

## Cross-dataset note

By best RMSE, the GRU orders subsets FD003 (10.6) ≈ FD001 (10.8) < FD002 (13.1) <
FD004 (14.7), cleanly tracking operating-mode count (1 → 6) and overall complexity.
Unlike Ridge, the GRU's RMSE ordering matches the intuitive difficulty ordering,
indicating it handles the capped-RUL plateau and varying lifetimes more uniformly
than the linear model.

## Recommendations

- `rolling_mean` is recommended for single-condition subsets;
  `raw_plus_rolling_mean` for FD004.
- A large rolling window is preferred (~15–20; ~10 for FD004): the sequence supplies
  dynamics, so the rolling mean should denoise.
- The `lag` family should be removed from the GRU search space: it prevents
  convergence (best_epoch 1–3) and roughly doubles RMSE.
- `raw` is an acceptable minimal fallback (~5% off best on clean subsets), with
  rolling-mean denoising as the low-cost upgrade.

## Official test-set results

Production GRU models, evaluated on the C-MAPSS official test set. These models
were trained with the configs selected by the *prior* PHM08-based ranking (FD001
and FD002 at window 15); RMSE re-ranking now favors window 20 on those two subsets,
so a production refresh at the new configs is pending and the table reflects the
deployed models as-is:

| Subset | Feature config (deployed) | Val RMSE | Test RMSE | Test MAE | Test PHM08 |
|--------|---------------|---:|---:|---:|---:|
| FD001 | rolling_mean, w=15 | 10.91 | 15.81 | 11.53 | 314 |
| FD002 | rolling_mean, w=15 | 13.17 | 24.63 | 15.56 | 4,945 |
| FD003 | rolling_mean, w=15 | 10.57 | 14.76 | 10.11 | 512 |
| FD004 | raw_plus_rolling_mean, w=10 | 14.73 | 23.67 | 16.81 | 3,144 |

The PHM08 score here is the canonical one-prediction-per-engine final-test metric.
The same pattern holds: single-condition subsets (FD001, FD003) generalize well
(~5 point val→test gap), while multi-condition subsets (FD002, FD004) show a larger
gap (~9–11 points).

## References

Methodology references support the mechanism explanations only; all empirical
results are from the sweep data above.

1. Cho, K., van Merriënboer, B., Gulcehre, C., et al. (2014). *Learning Phrase
   Representations using RNN Encoder–Decoder for Statistical Machine Translation.*
   Proc. EMNLP 2014. arXiv:1406.1078. (Introduces the GRU.)
2. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). *Empirical Evaluation of
   Gated Recurrent Neural Networks on Sequence Modeling.* arXiv:1412.3555.
3. Smith, S. W. (1997). *The Scientist and Engineer's Guide to Digital Signal
   Processing* (Ch. 15, Moving Average Filters). California Technical Publishing.
   ISBN 0-9660176-3-3.
4. LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). *Efficient BackProp.*
   In *Neural Networks: Tricks of the Trade*, Springer LNCS 1524, 9–50.
   (Input scaling/normalization conditions gradient-based training.)
5. Prechelt, L. (1998). *Early Stopping — But When?* In *Neural Networks: Tricks of
   the Trade*, Springer LNCS 1524, 55–69.
