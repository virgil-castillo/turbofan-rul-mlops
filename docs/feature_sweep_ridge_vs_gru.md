# Feature Engineering for Ridge vs. GRU — Comparison

A comparison of what the two models require from feature engineering. Empirical
conclusions are derived solely from `results/feature_sweep_{ridge,gru}_fd00{1-4}.csv`
and the EDA notebooks; cited literature explains the mechanisms only. Per-model
detail is in [feature_sweep_ridge_report.md](feature_sweep_ridge_report.md) and
[feature_sweep_gru_report.md](feature_sweep_gru_report.md).

## 1. Overall performance: the GRU dominates identical feature engineering

Best configuration per subset, both models (RMSE / PHM08):

| Dataset | Ridge RMSE | GRU RMSE | RMSE Δ | Ridge PHM08 | GRU PHM08 | PHM08 ratio |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 20.72 | 10.91 | −47% | 36,679 | 5,005 | 7.3× |
| FD002 | 19.35 | 13.17 | −32% | 93,063 | 22,145 | 4.2× |
| FD003 | 17.07 | 10.57 | −38% | 36,932 | 6,481 | 5.7× |
| FD004 | 18.47 | 14.73 | −20% | 272,864 | 208,349 | 1.3× |

The GRU roughly halves RMSE on the clean subsets. Its advantage is largest on
single-condition data (FD001/FD003) and smallest on FD004 (6 modes, longest
lifetimes), where the two models converge.

## 2. Primary divergence: the rolling-window direction is opposite

For plain `rolling_mean`, PHM08 vs window on FD001:

| Window | 2 | 4 | 6 | 8 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 36,720 | 37,056 | 38,204 | 39,455 | 40,831 | 44,769 | 49,621 |
| GRU | 6,254 | 6,244 | 6,188 | 6,174 | 5,260 | 5,005 | 5,017 |

- Ridge is optimal at the smallest window (2–4). As a memoryless linear estimator
  [1], its only temporal context is the rolling feature itself; a wider
  moving-average window introduces group delay [4], lagging the current state and
  erasing end-of-life movement — hence monotonic degradation.
- The GRU is optimal at a large window (~15). It already models 45 consecutive
  cycles [2, 3], so the rolling mean serves to denoise [4]; wider windows clean the
  input without losing dynamics.

The same transformation has opposite optimal settings because the two models
consume time differently.

## 3. The lag family: useless for Ridge, harmful for the GRU

| | Ridge | GRU |
|---|---|---|
| `lag` alone | Catastrophic (RMSE ~40–42, PHM08 in millions) | Catastrophic (RMSE ~28–31) |
| `raw_plus_lag` vs `raw` | Neutral (≈ equal to `raw`) | Harmful (RMSE ~2× `raw`) |
| Mechanism | First-difference removes the absolute level [4]; a memoryless linear fit [1] has nothing to anchor to | Off-scale lag channels degrade input conditioning [5]; training fails to improve and early-stops at epoch 1–3 [6] |

For Ridge the lag-difference carries no marginal information once `raw` is present.
For the GRU it is worse than useless: it destabilizes training, visible in the
`best_epoch` column (lag runs early-stop at epoch 1–3 vs 5–17 for working sets).
Either way, the lag family should be dropped from the search space.

## 4. The role of the raw channels differs

| | Ridge | GRU |
|---|---|---|
| `raw` alone vs best | ~5% worse (FD001) | ~10% worse (FD001) |
| Adding raw to rolling | Marginal gain, but stabilizes window choice (spread 35% → 3.5%) | Slightly hurts on single-condition; helps on FD004 (−17% PHM08) |

For Ridge, `raw_plus_rolling_mean` is the robust default: it retains the un-delayed
current value so a mis-set window cannot over-smooth. For the GRU, plain
`rolling_mean` is best on clean data; raw channels earn their place only under 6
operating modes (FD004), where the normalized raw value carries mode-residual
detail a rolling mean removes.

## 5. What "good feature engineering" means for each model

Ridge — engineer the temporal context, because the model has none [1]:
- a rolling mean is essential and must be short (responsive);
- retaining `raw` (`raw_plus_rolling_mean`) adds window robustness;
- lag adds nothing;
- per-mode normalization is mandatory for FD002/FD004 (EDA: raw correlations vanish
  otherwise).

GRU — engineer to denoise, because the model already has temporal context [2, 3]:
- a rolling mean helps mainly as a wide denoiser [4] on top of the 45-cycle
  sequence;
- `raw` alone is already competitive;
- lag is harmful (non-convergence [5, 6]), not just redundant;
- raw is added to rolling only under multiple operating modes (FD004);
- per-mode normalization is equally mandatory for FD002/FD004.

Common to both: rolling-mean denoising is the most reliable transform;
operating-mode normalization is a hard requirement on multi-condition subsets; the
lag family should be abandoned.

## 6. Notes on comparing the numbers

- PHM08 is an unnormalized sum, so its magnitude scales with validation size
  (FD002/FD004 have ~2.5× the engines of FD001/FD003). It ranks configurations
  within a dataset; RMSE/MAE compare across datasets.
- RMSE and PHM08 disagree on Ridge difficulty: by RMSE the single-condition FD001
  is Ridge's hardest subset, plausibly because shorter engine lives (EDA median
  199) give fewer cycles in the flat `max_rul=125` plateau than the longer-lived
  FD003/FD004 — an interpretation consistent with the lifetime and cap numbers, not
  a directly measured effect. The GRU's RMSE ordering instead tracks operating-mode
  count cleanly (FD003 ≈ FD001 < FD002 < FD004).

## 7. Limitations and open questions

These sweeps fixed the GRU at `hidden_size=64`, `window_size=45`, and Ridge at
`alpha=100`. The window-direction and lag findings concern feature choices at those
fixed model settings and are not guaranteed to hold if the sequence window or
regularization strength changes substantially. A sweep crossing the rolling-feature
window against the sequence window would test the denoising interpretation directly.

## References

Methodology references support the mechanism explanations only; all empirical
results are from the sweep data above.

1. Hoerl, A. E., & Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for
   Nonorthogonal Problems.* Technometrics, 12(1), 55–67.
   doi:10.1080/00401706.1970.10488634
2. Cho, K., van Merriënboer, B., Gulcehre, C., et al. (2014). *Learning Phrase
   Representations using RNN Encoder–Decoder for Statistical Machine Translation.*
   Proc. EMNLP 2014. arXiv:1406.1078. (Introduces the GRU.)
3. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). *Empirical Evaluation of
   Gated Recurrent Neural Networks on Sequence Modeling.* arXiv:1412.3555.
4. Smith, S. W. (1997). *The Scientist and Engineer's Guide to Digital Signal
   Processing* (Ch. 15, Moving Average Filters; differencing as a high-pass
   operation). California Technical Publishing. ISBN 0-9660176-3-3.
5. LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). *Efficient BackProp.*
   In *Neural Networks: Tricks of the Trade*, Springer LNCS 1524, 9–50.
6. Prechelt, L. (1998). *Early Stopping — But When?* In *Neural Networks: Tricks of
   the Trade*, Springer LNCS 1524, 55–69.
