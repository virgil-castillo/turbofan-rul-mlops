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

Windows: `2, 4, 6, 8, 10, 15, 20`. Lag steps: `2, 4, 8`. Ranking uses validation
RMSE (the sort key; lower is better), with MAE reported alongside. The PHM08 score
is reserved for official-test evaluation and is not used to rank validation runs.

## Best configuration per dataset

| Dataset | Best feature set | Window | n_features | RMSE | MAE |
|---|---|---:|---:|---:|---:|
| FD001 | `raw_plus_rolling_mean` | 20 | 30 | 20.28 | 15.70 |
| FD002 | `raw_plus_rolling_mean` | 4 | 32 | 19.35 | 14.74 |
| FD003 | `raw_plus_rolling_mean` | 6 | 24 | 17.05 | 12.58 |
| FD004 | `raw_plus_rolling_mean` | 6 | 28 | 18.44 | 13.24 |

`raw_plus_rolling_mean` is optimal on all four datasets. The window is small (4–6)
on three subsets; FD001 is the exception, improving all the way to window 20
(see Finding 5).

## Finding 1 — The feature-family ranking is identical on every dataset

Sorting each subset by RMSE produces the same tiering:

1. `raw_plus_rolling_mean` ≈ `rolling_mean` (interleaved at the top)
2. `raw` ≈ `raw_plus_lag` (a clear step worse)
3. `lag` alone (catastrophic)

The tier-1-to-tier-2 gap is modest; the gap to tier 3 is enormous. FD001 shows the
full spread:

| Feature set (FD001, best window/step) | RMSE | vs. best |
|---|---:|---:|
| `raw_plus_rolling_mean` (w20) | 20.28 | — |
| `rolling_mean` (w4) | 20.67 | +1.9% |
| `raw_plus_lag` (step 4) | 21.01 | +3.5% |
| `raw` | 21.02 | +3.6% |
| `lag` (step 2) | 41.77 | +106% |

## Finding 2 — `lag` alone is unusable

The `lag` family reaches RMSE ≈ 40–42 on every subset, versus ~17–21 for the
working families:

| Dataset | `lag` best RMSE |
|---|---:|
| FD001 | 41.77 |
| FD002 | 41.62 |
| FD003 | 41.02 |
| FD004 | 40.22 |

A lag-difference is a discrete first-difference operator, which acts as a high-pass
filter: it discards the absolute sensor level and retains only short-term change
[2]. Ridge is a memoryless linear estimator that scores each cycle
independently [1], so with the level removed there is nothing to anchor the
prediction to, and performance collapses to near-constant. RMSE being flat across
lag steps 2/4/8 (all ≈ 41) confirms no lag horizon recovers usable signal.

## Finding 3 — Adding `lag` to `raw` is neutral, not additive

`raw_plus_lag` essentially reproduces `raw`: their best-RMSE values are within 0.02
on every subset (FD001 21.02 vs 21.01; FD002 20.39 vs 20.40; FD003 18.06 vs 18.05;
FD004 20.01 vs 20.00). Once the absolute level is present, the high-pass lag
channels carry no marginal information for the linear model. The lag family's value
to Ridge is effectively zero.

## Finding 4 — Small rolling windows are preferred (plain rolling_mean)

For plain `rolling_mean`, RMSE degrades monotonically as the window grows past its
small optimum (FD001):

| Window | 2 | 4 | 6 | 8 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RMSE | 20.73 | 20.67 | 20.75 | 20.88 | 21.03 | 21.48 | 21.98 |

The pattern holds across subsets (best plain-rolling windows: FD001→4, FD002→4,
FD003→4, FD004→6). A rolling mean is a moving-average low-pass filter; widening it
attenuates more high-frequency content but also introduces group delay, so the
feature increasingly lags the true state [2]. Because Ridge sees only the
current cycle's row [1], a lagged, over-smoothed feature erases the fast
end-of-life movement the linear fit depends on. A short window keeps the feature
tracking the current condition.

## Finding 5 — Retaining `raw` makes the window choice robust

`raw_plus_rolling_mean` is far less window-sensitive than plain `rolling_mean`
(FD001, RMSE range across windows 2–20):

| Family | min RMSE | max RMSE | spread |
|---|---:|---:|---:|
| `rolling_mean` | 20.67 (w4) | 21.98 (w20) | +6.3% |
| `raw_plus_rolling_mean` | 20.28 (w20) | 20.72 (w2) | +2.2% |

The retained raw channels preserve the un-delayed current value, so over-smoothing
from a wide window no longer dominates — on FD001 the combined family even keeps
improving out to window 20. `raw_plus_rolling_mean` is consequently the robust
default: near-best across the window range and forgiving when the window is mis-set.

## Per-dataset notes (EDA context and sweep behavior)

### FD001 — 100 engines, 1 operating condition (EDA: 20,631 rows; lifetimes 128/199/362)
Single condition, no per-mode normalization (`n_modes=1`); 15 informative sensors
retained after dropping `s_1, s_5, s_10, s_16, s_18, s_19`. The best Ridge RMSE
here (20.28) is the highest of all subsets despite being the simplest dataset — see
the cross-dataset note. Best configuration: `raw_plus_rolling_mean`, window 20.

### FD002 — 260 engines, 6 operating conditions (EDA: 53,759 rows; lifetimes 128/199/378)
EDA shows all raw sensor–RUL correlations fall below threshold until per-mode
normalization is applied (`n_modes=6`): operating-mode variance masks the
degradation signal entirely. With normalization in the pipeline, Ridge reaches RMSE
19.35. 16 sensors retained. Best window 4.

### FD003 — 100 engines, 1 operating condition (EDA: 24,720 rows; lifetimes 145/220/525)
Single condition with longer-lived engines than FD001. Most aggressive pruning
(only 12 sensors retained; `s_5, s_6, s_15, s_16, s_20, s_21` dropped). Best Ridge
RMSE of all subsets (17.05) — see the plateau note. Best configuration:
`raw_plus_rolling_mean`, window 6.

### FD004 — 249 engines, 6 operating conditions (EDA: 61,249 rows; lifetimes 128/234/543)
Largest and longest-lived subset; 6 conditions (`n_modes=6`), 14 sensors retained.
Best RMSE 18.44 at window 6, modestly larger than the single-condition subsets,
consistent with needing slightly more smoothing under multi-mode noise.

## Cross-dataset note: RMSE disagrees with operating-mode count on difficulty

Best-achievable RMSE orders the subsets FD003 (17.1) < FD004 (18.4) < FD002 (19.4)
< FD001 (20.3) — i.e. the single-condition FD001 has the worst Ridge RMSE, the
opposite of what operating-condition count would predict. This aligns with the EDA
lifetime statistics combined with the `max_rul=125` cap: longer-lived subsets
(FD003 median 220, FD004 median 234) spend many cycles in the flat capped-RUL=125
plateau, which are trivial rows that pull RMSE down, whereas FD001 (median 199) has
proportionally fewer plateau rows. This is an interpretation consistent with the
lifetime/cap numbers, not a directly measured quantity.

## Recommendations

- `raw_plus_rolling_mean` is the recommended feature set: best on all four subsets
  and robust to window choice.
- A small window is a safe default (~4–6); `raw_plus_rolling_mean` tolerates larger
  windows (FD001 optimal at 20), so the window is not a sensitive knob for it.
- The `lag` family should be removed from the Ridge search space: unusable alone,
  and additive-neutral when combined with `raw`.
- Per-mode normalization is mandatory for FD002/FD004 — EDA shows raw correlations
  vanish without it, and the configs already encode `n_modes=6`.

## Official test-set results

Production Ridge models, evaluated on the C-MAPSS official test set. These models
were trained with the configs selected by the *prior* PHM08-based ranking; RMSE
re-ranking shifts the best window on three subsets (FD001 2→20, FD003 4→6, FD004
8→6; FD002 unchanged at 4), so a production refresh at the new configs is pending
and the table reflects the deployed models as-is:

| Subset | Feature config (deployed) | Val RMSE | Test RMSE | Test MAE | Test PHM08 |
|--------|---------------|---:|---:|---:|---:|
| FD001 | raw_plus_rolling_mean, w=2 | 20.72 | 21.58 | 17.44 | 1,315 |
| FD002 | raw_plus_rolling_mean, w=4 | 19.35 | 31.31 | 22.85 | 17,733 |
| FD003 | raw_plus_rolling_mean, w=4 | 17.07 | 23.01 | 18.20 | 2,492 |
| FD004 | raw_plus_rolling_mean, w=8 | 18.47 | 32.88 | 26.20 | 9,646 |

The PHM08 score here is the canonical one-prediction-per-engine final-test metric.
Multi-condition subsets (FD002, FD004) show the largest val→test RMSE gap (~12–14
points), consistent with a harder distribution shift on those subsets.

## References

Methodology references support the mechanism explanations only; all empirical
results are from the sweep data above.

1. Hoerl, A. E., & Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for
   Nonorthogonal Problems.* Technometrics, 12(1), 55–67.
   doi:10.1080/00401706.1970.10488634
2. Smith, S. W. (1997). *The Scientist and Engineer's Guide to Digital Signal
   Processing* (Ch. 15, Moving Average Filters; differencing as a high-pass
   operation). California Technical Publishing. ISBN 0-9660176-3-3.
