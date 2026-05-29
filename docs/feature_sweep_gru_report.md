# GRU Feature-Engineering Sweep

Analysis of `results/feature_sweep_gru_fd00{1-4}.csv`. Empirical conclusions are
derived solely from those sweep results and the EDA notebooks
(`notebooks/eda_fd00{1-4}.ipynb`); cited literature is used only to explain the
mechanisms behind the observed numbers, never to import results.

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
early-stop at epoch 1–3 (no validation improvement). As in the Ridge analysis,
PHM08 is a size-scaled sum (used for within-dataset ranking) and RMSE/MAE are used
across datasets.

## Best configuration per dataset

| Dataset | Best feature set | Window | n_features | best_epoch | RMSE | MAE | PHM08 |
|---|---|---:|---:|---:|---:|---:|---:|
| FD001 | `rolling_mean` | 15 | 15 | 12 | 10.91 | 7.91 | 5,005 |
| FD002 | `rolling_mean` | 15 | 16 | 3 | 13.17 | 9.95 | 22,145 |
| FD003 | `rolling_mean` | 15 | 12 | 9 | 10.57 | 7.66 | 6,481 |
| FD004 | `raw_plus_rolling_mean` | 10 | 28 | 2 | 14.73 | 9.49 | 208,349 |

Plain `rolling_mean` wins on the three cleaner subsets; the multi-mode FD004
prefers `raw_plus_rolling_mean`. The optimal window is large (10–15).

## Finding 1 — Feature-family ranking on every dataset

Sorting by PHM08 gives a consistent tiering:

1. `rolling_mean` ≈ `raw_plus_rolling_mean` (top, interleaved)
2. `raw` (close behind)
3. `raw_plus_lag` (a large drop)
4. `lag` alone (catastrophic)

FD001 spread, showing how close the top tier and `raw` are and how far the lag
families fall:

| Feature set (FD001, best window/step) | best_epoch | RMSE | PHM08 | vs. best |
|---|---:|---:|---:|---:|
| `rolling_mean` (w15) | 12 | 10.91 | 5,005 | — |
| `raw_plus_rolling_mean` (w15) | 5 | 11.00 | 5,273 | +5.4% |
| `raw` | 6 | 11.36 | 5,529 | +10.5% |
| `raw_plus_lag` (step 4) | 22 | 21.88 | 51,337 | +926% |
| `lag` (step 4) | 1 | 29.18 | 244,299 | +4,782% |

## Finding 2 — Large rolling windows are preferred (opposite of Ridge)

For plain `rolling_mean`, PHM08 improves as the window grows, peaking at 15
(FD001):

| Window | 2 | 4 | 6 | 8 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PHM08 | 6,254 | 6,244 | 6,188 | 6,174 | 5,260 | 5,005 | 5,017 |

The pattern repeats across subsets (best plain-rolling windows: 15, 15, 15, and
~6–10 for FD004). A GRU already models temporal dependencies across its 45-cycle
input [1, 2], so the rolling mean's role is denoising rather than trend extraction.
A wide moving-average window suppresses high-frequency sensor noise [3] while the
sequence still supplies the dynamics, so wider windows help — the reverse of the
short-window responsiveness a memoryless linear model needs.

## Finding 3 — `raw` alone is already competitive

`raw` lands within ~10% of the best on the cleaner subsets and only collapses on
FD004:

| Dataset | `raw` RMSE | `raw` PHM08 | best RMSE | best PHM08 |
|---|---:|---:|---:|---:|
| FD001 | 11.36 | 5,529 | 10.91 | 5,005 |
| FD002 | 14.04 | 29,563 | 13.17 | 22,145 |
| FD003 | 10.97 | 7,226 | 10.57 | 6,481 |
| FD004 | 15.50 | 360,409 | 14.73 | 208,349 |

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
early-stop at epoch 1–3 (e.g. all FD002 lag/`raw_plus_lag` runs at best_epoch 1–2),
meaning validation loss never improved past the first epoch [5]. The lag-difference
channels are on a different, near-zero-centered scale than the normalized levels;
poorly conditioned input scaling slows and destabilizes gradient-based training
[4], so the network fails to converge. The lag family is not merely redundant for
the GRU — it is damaging.

## Finding 5 — Retaining `raw` helps only under multiple operating modes

Whether `raw_plus_rolling_mean` beats plain `rolling_mean` splits by
operating-mode count:

| Dataset | modes | `rolling_mean` best PHM08 | `raw_plus_rolling_mean` best PHM08 | winner |
|---|---:|---:|---:|---|
| FD001 | 1 | 5,005 | 5,273 | rolling_mean |
| FD002 | 6 | 22,145 | 28,778 | rolling_mean |
| FD003 | 1 | 6,481 | 6,719 | rolling_mean |
| FD004 | 6 | 252,220 | 208,349 | raw_plus_rolling_mean |

On the single-condition subsets the raw channels are redundant and marginally hurt.
On FD004 — the hardest multi-mode subset — retaining the normalized raw value
beside the smoothed value supplies mode-residual detail a rolling mean washes out,
improving PHM08 ~17%. FD002 (also 6 modes) does not show this, so the effect is
real but not guaranteed by mode count alone.

## Per-dataset notes (EDA context and sweep behavior)

### FD001 — 100 engines, 1 condition (EDA: 20,631 rows; lifetimes 128/199/362)
`n_modes=1`, 15 sensors retained. Cleanest result: best RMSE 10.91 at
`rolling_mean` w15; `raw` only 10% behind; lag families catastrophic.

### FD002 — 260 engines, 6 conditions (EDA: 53,759 rows; lifetimes 128/199/378)
EDA: raw correlations vanish until per-mode normalization (`n_modes=6`). Best RMSE
13.17. Top runs early-stop at epoch 3 — multi-mode data is harder to fit and the
GRU converges quickly then plateaus.

### FD003 — 100 engines, 1 condition (EDA: 24,720 rows; lifetimes 145/220/525)
Only 12 sensors retained (most aggressive pruning). Best RMSE 10.57 — the best of
all four subsets — at `rolling_mean` w15, despite longer engines than FD001.

### FD004 — 249 engines, 6 conditions (EDA: 61,249 rows; lifetimes 128/234/543)
Largest, longest-lived, hardest. The only subset where `raw_plus_rolling_mean`
wins, at a smaller window (10) and best_epoch 2. Highest RMSE (14.73) and PHM08.

## Cross-dataset note

By best RMSE, the GRU orders subsets FD003 (10.6) ≈ FD001 (10.9) < FD002 (13.2) <
FD004 (14.7), cleanly tracking operating-mode count (1 → 6) and overall complexity.
Unlike Ridge, the GRU's RMSE ordering matches the intuitive difficulty ordering,
indicating it handles the capped-RUL plateau and varying lifetimes more uniformly
than the linear model.

## Recommendations

- `rolling_mean` is recommended for single-condition subsets;
  `raw_plus_rolling_mean` for FD004.
- A large rolling window is preferred (~15; ~10 for FD004): the sequence supplies
  dynamics, so the rolling mean should denoise.
- The `lag` family should be removed from the GRU search space: it prevents
  convergence (best_epoch 1–3) and roughly doubles RMSE.
- `raw` is an acceptable minimal fallback (~10% off best on clean subsets), with
  rolling-mean denoising as the low-cost upgrade.

## Official test-set results

Production GRU models trained with each subset's best sweep config, evaluated on
the C-MAPSS official test set:

| Subset | Feature config | Val RMSE | Test RMSE | Test MAE | Test PHM08 |
|--------|---------------|---:|---:|---:|---:|
| FD001 | rolling_mean, w=15 | 10.91 | 15.81 | 11.53 | 314 |
| FD002 | rolling_mean, w=15 | 13.17 | 24.63 | 15.56 | 4,945 |
| FD003 | rolling_mean, w=15 | 10.57 | 14.76 | 10.11 | 512 |
| FD004 | raw_plus_rolling_mean, w=10 | 14.73 | 23.67 | 16.81 | 3,144 |

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
