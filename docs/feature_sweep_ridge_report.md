# Ridge Feature-Engineering Sweep

Analysis of `results/feature_sweep_ridge_fd00{1-4}.csv`.

## Method

Each sweep evaluated one Ridge model (`alpha=100`, `max_rul=125`) per feature
configuration on an engine-level validation split. Feature families:

- `raw` — normalized sensors only (1 configuration)
- `rolling_mean` — per-engine rolling mean (one configuration per window)
- `raw_plus_rolling_mean` — raw + rolling mean concatenated (one per window)
- `lag` — normalized lag-difference `(x[t]-x[t-N]) / rollmean(x,N)` (one per step)
- `raw_plus_lag` — raw + lag-difference (one per step)

Windows: `2, 4, 6, 8, 10, 15, 20`. Lag steps: `2, 4, 8`. Recorded metrics: RMSE,
MAE, and the PHM08 score (the sort key; lower is better).

The PHM08 column is an unnormalized sum over validation samples (values span
~3.7e4 to ~5.6e6), so its magnitude scales with the number of validation rows.
FD002 (260 engines) and FD004 (249 engines) have ~2.5× the engines of FD001/FD003
(100 each), which inflates their PHM08 independently of difficulty. PHM08 is
therefore used for ranking within a dataset; RMSE/MAE are used for comparison
across datasets.

## Best configuration per dataset

| Dataset | Best feature set | Window | n_features | RMSE | MAE | PHM08 |
|---|---|---:|---:|---:|---:|---:|
| FD001 | `raw_plus_rolling_mean` | 2 | 30 | 20.72 | 16.08 | 36,679 |
| FD002 | `raw_plus_rolling_mean` | 4 | 32 | 19.35 | 14.74 | 93,063 |
| FD003 | `raw_plus_rolling_mean` | 4 | 24 | 17.07 | 12.66 | 36,932 |
| FD004 | `raw_plus_rolling_mean` | 8 | 28 | 18.47 | 13.27 | 272,864 |

`raw_plus_rolling_mean` is optimal on all four datasets, at a small window (2–8).

## Finding 1 — The feature-family ranking is identical on every dataset

Sorting each subset by PHM08 produces the same tiering:

1. `raw_plus_rolling_mean` ≈ `rolling_mean` (interleaved at the top)
2. `raw` ≈ `raw_plus_lag` (a clear step worse)
3. `lag` alone (catastrophic)

The tier-1-to-tier-2 gap is modest; the gap to tier 3 is enormous. FD001 shows the
full spread:

| Feature set (FD001, best window/step) | RMSE | PHM08 | vs. best |
|---|---:|---:|---:|
| `raw_plus_rolling_mean` (w2) | 20.72 | 36,679 | — |
| `rolling_mean` (w2) | 20.73 | 36,720 | +0.1% |
| `raw_plus_lag` (step 8) | 21.01 | 38,345 | +4.5% |
| `raw` | 21.02 | 38,474 | +4.9% |
| `lag` (step 2) | 41.77 | 1,270,933 | +3,365% |

## Finding 2 — `lag` alone is unusable

The `lag` family reaches RMSE ≈ 40–42 on every subset (vs ~17–21 for the working
families) and PHM08 in the millions:

| Dataset | `lag` best RMSE | `lag` best PHM08 |
|---|---:|---:|
| FD001 | 41.77 | 1,270,933 |
| FD002 | 41.62 | 3,356,558 |
| FD003 | 41.02 | 2,469,492 |
| FD004 | 40.22 | 5,585,818 |

A lag-difference is a discrete first-difference operator, which acts as a high-pass
filter: it discards the absolute sensor level and retains only short-term change
[2]. Ridge is a memoryless linear estimator that scores each cycle
independently [1], so with the level removed there is nothing to anchor the
prediction to, and performance collapses to near-constant. RMSE being flat across
lag steps 2/4/8 (all ≈ 41) confirms no lag horizon recovers usable signal.

## Finding 3 — Adding `lag` to `raw` is neutral, not additive

`raw_plus_lag` essentially reproduces `raw`: FD001 `raw` = 38,474 vs `raw_plus_lag`
= 38,345–38,936; FD002 `raw` = 110,250 vs `raw_plus_lag` = 110,142–125,172. Once
the absolute level is present, the high-pass lag channels carry no marginal
information for the linear model. The lag family's value to Ridge is effectively
zero.

## Finding 4 — Small rolling windows are preferred

For plain `rolling_mean`, PHM08 degrades monotonically as the window grows
(FD001):

| Window | 2 | 4 | 6 | 8 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PHM08 | 36,720 | 37,056 | 38,204 | 39,455 | 40,831 | 44,769 | 49,621 |

The pattern holds across subsets (best plain-rolling windows: FD001→2, FD002→4,
FD003→4, FD004→8). A rolling mean is a moving-average low-pass filter; widening it
attenuates more high-frequency content but also introduces group delay, so the
feature increasingly lags the true state [2]. Because Ridge sees only the
current cycle's row [1], a lagged, over-smoothed feature erases the fast
end-of-life movement the linear fit depends on. A short window keeps the feature
tracking the current condition.

## Finding 5 — Retaining `raw` makes the window choice robust

`raw_plus_rolling_mean` is far less window-sensitive than plain `rolling_mean`
(FD001, PHM08 range across windows 2–20):

| Family | min PHM08 | max PHM08 | spread |
|---|---:|---:|---:|
| `rolling_mean` | 36,720 (w2) | 49,621 (w20) | +35% |
| `raw_plus_rolling_mean` | 36,679 (w2) | 37,968 (w15) | +3.5% |

The retained raw channels preserve the un-delayed current value, so over-smoothing
from a wide window no longer dominates. `raw_plus_rolling_mean` is consequently the
robust default: near-best at the optimal window and forgiving when the window is
mis-set.

## Per-dataset notes (EDA context and sweep behavior)

### FD001 — 100 engines, 1 operating condition (EDA: 20,631 rows; lifetimes 128/199/362)
Single condition, no per-mode normalization (`n_modes=1`); 15 informative sensors
retained after dropping `s_1, s_5, s_10, s_16, s_18, s_19`. The best Ridge RMSE
here (20.72) is the highest of all subsets despite being the simplest dataset — see
the cross-dataset note. Best configuration: `raw_plus_rolling_mean`, window 2.

### FD002 — 260 engines, 6 operating conditions (EDA: 53,759 rows; lifetimes 128/199/378)
EDA shows all raw sensor–RUL correlations fall below threshold until per-mode
normalization is applied (`n_modes=6`): operating-mode variance masks the
degradation signal entirely. With normalization in the pipeline, Ridge reaches RMSE
19.35. 16 sensors retained. Best window 4.

### FD003 — 100 engines, 1 operating condition (EDA: 24,720 rows; lifetimes 145/220/525)
Single condition with longer-lived engines than FD001. Most aggressive pruning
(only 12 sensors retained; `s_5, s_6, s_15, s_16, s_20, s_21` dropped). Best Ridge
RMSE of all subsets (17.07) — see the plateau note. Best configuration:
`raw_plus_rolling_mean`, window 4.

### FD004 — 249 engines, 6 operating conditions (EDA: 61,249 rows; lifetimes 128/234/543)
Largest and longest-lived subset; 6 conditions (`n_modes=6`), 14 sensors retained.
Highest PHM08 (272,864 at best), partly from engine count and partly real
difficulty. The optimal window is the largest of the four (8), consistent with
needing slightly more smoothing under multi-mode noise.

## Cross-dataset note: RMSE and PHM08 disagree on difficulty

Best-achievable RMSE orders the subsets FD003 (17.1) < FD004 (18.5) < FD002 (19.4)
< FD001 (20.7) — i.e. the single-condition FD001 has the worst Ridge RMSE, the
opposite of what operating-condition count would predict. This aligns with the EDA
lifetime statistics combined with the `max_rul=125` cap: longer-lived subsets
(FD003 median 220, FD004 median 234) spend many cycles in the flat capped-RUL=125
plateau, which are trivial rows that pull RMSE down, whereas FD001 (median 199) has
proportionally fewer plateau rows. This is an interpretation consistent with the
lifetime/cap numbers, not a directly measured quantity. PHM08, a size-scaled
asymmetric sum, instead tracks engine count (FD002/FD004 highest).

## Recommendations

- `raw_plus_rolling_mean` is the recommended feature set: best or tied-best on all
  four subsets and robust to window choice.
- A small window is preferred — 2–4 for single-condition subsets, ~8 for FD004.
- The `lag` family should be removed from the Ridge search space: unusable alone,
  and additive-neutral when combined with `raw`.
- Per-mode normalization is mandatory for FD002/FD004 — EDA shows raw correlations
  vanish without it, and the configs already encode `n_modes=6`.

## References

Methodology references support the mechanism explanations only; all empirical
results are from the sweep data above.

1. Hoerl, A. E., & Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for
   Nonorthogonal Problems.* Technometrics, 12(1), 55–67.
   doi:10.1080/00401706.1970.10488634
2. Smith, S. W. (1997). *The Scientist and Engineer's Guide to Digital Signal
   Processing* (Ch. 15, Moving Average Filters; differencing as a high-pass
   operation). California Technical Publishing. ISBN 0-9660176-3-3.
