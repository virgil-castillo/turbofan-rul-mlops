# Feature Engineering for Ridge vs. GRU — Comparison

A comparison of what the two models require from feature engineering. Ridge detail
is in [feature_sweep_ridge_report.md](feature_sweep_ridge_report.md); the current
GRU reference is the two-stage
[gru_capacity_sweep_report.md](gru_capacity_sweep_report.md). The GRU
feature-engineering mechanism figures used in §2–§5 come from the now-archived
[GRU feature sweep](archive/feature_sweep_gru_report.md) (fixed `hidden_size=64`,
`window_size=45`). Ranking is by validation RMSE; the PHM08 score appears only for
official-test results (§7).

## 1. Overall performance: the GRU roughly halves Ridge's error

Best validation RMSE per subset. Ridge is its best feature-sweep config
(`alpha=100`); the GRU is the best run from the two-stage temporal + capacity sweep
([gru_capacity_sweep_report.md](gru_capacity_sweep_report.md)). Both use the same
engine-level split and `max_rul=125`:

| Dataset | Ridge RMSE | GRU RMSE | RMSE Δ |
|---|---:|---:|---:|
| FD001 | 20.28 | 10.19 | −50% |
| FD002 | 19.35 | 12.78 | −34% |
| FD003 | 17.05 | 10.07 | −41% |
| FD004 | 18.44 | 14.73 | −20% |

The GRU roughly halves RMSE on the clean subsets. Its advantage is largest on
single-condition data (FD001/FD003) and smallest on FD004 (6 modes, longest
lifetimes), where the two models converge. The GRU figures improved slightly versus
the archived fixed-capacity sweep (e.g. FD001 10.82 → 10.19) once the sequence
window and hidden size were tuned per subset.

## 2. Primary divergence: the rolling-window direction is opposite

For plain `rolling_mean`, RMSE vs window on FD001:

| Window | 2 | 4 | 6 | 8 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 20.73 | 20.67 | 20.75 | 20.88 | 21.03 | 21.48 | 21.98 |
| GRU | 11.47 | 11.32 | 11.28 | 11.25 | 11.10 | 10.91 | 10.82 |

- Ridge is optimal at the smallest window (2–4). As a memoryless linear estimator
  [1], its only temporal context is the rolling feature itself; a wider
  moving-average window introduces group delay [4], lagging the current state and
  erasing end-of-life movement — hence monotonic degradation.
- The GRU is optimal at a large window (~20). It already models 45 consecutive
  cycles [2, 3], so the rolling mean serves to denoise [4]; wider windows clean the
  input without losing dynamics.

The same transformation has opposite optimal settings because the two models
consume time differently.

## 3. The lag family: useless for Ridge, harmful for the GRU

| | Ridge | GRU |
|---|---|---|
| `lag` alone | Catastrophic (RMSE ~40–42) | Catastrophic (RMSE ~26–31) |
| `raw_plus_lag` vs `raw` | Neutral (≈ equal to `raw`) | Harmful (RMSE ~2× `raw`) |
| Mechanism | First-difference removes the absolute level [4]; a memoryless linear fit [1] has nothing to anchor to | Off-scale lag channels degrade input conditioning [5]; training fails to improve and early-stops at epoch 1–3 [6] |

For Ridge the lag-difference carries no marginal information once `raw` is present.
For the GRU it is worse than useless: it destabilizes training, visible in the
`best_epoch` column (lag runs early-stop at epoch 1–3 vs 5–17 for working sets).
Either way, the lag family should be dropped from the search space.

## 4. The role of the raw channels differs

| | Ridge | GRU |
|---|---|---|
| `raw` alone vs best | ~3.6% worse (FD001) | ~5% worse (FD001) |
| Adding raw to rolling | Marginal gain, but stabilizes window choice (RMSE spread 6.3% → 2.2%) | Slightly hurts on single-condition; helps on FD004 (−2.4% RMSE) |

For Ridge, `raw_plus_rolling_mean` is the robust default: it retains the un-delayed
current value so a mis-set window cannot over-smooth. For the GRU, plain
`rolling_mean` is best on clean data; raw channels earn their place only under 6
operating modes (FD004), where the normalized raw value carries mode-residual
detail a rolling mean removes.

## 5. What "good feature engineering" means for each model

Ridge — engineer the temporal context, because the model has none [1]:
- a rolling mean is essential and is best kept short (responsive) when used alone;
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

- Validation runs are ranked by RMSE (with MAE alongside). The PHM08 score is the
  PHM08 Data Challenge final-test metric — a sum over one prediction per engine — so
  it is computed only on the official test set (§7), not over sliding-window
  validation predictions.
- By RMSE, the single-condition FD001 is Ridge's hardest subset, plausibly because
  shorter engine lives (EDA median 199) give fewer cycles in the flat `max_rul=125`
  plateau than the longer-lived FD003/FD004 — an interpretation consistent with the
  lifetime and cap numbers, not a directly measured effect. The GRU's RMSE ordering
  instead tracks operating-mode count cleanly (FD003 ≈ FD001 < FD002 < FD004).

## 7. Official test-set results

Production models, evaluated on the C-MAPSS official test set (unseen engines,
single prediction per engine at the last cycle). The PHM08 score is the canonical
final-test metric. The **GRU rows are the retrained Stage 2 capacity-sweep selected
configs** (CPU, 2026-06-01; see [gru_capacity_sweep_report.md](gru_capacity_sweep_report.md));
the **Ridge rows still reflect the prior deployed models**, whose production refresh
is pending:

| Subset | Ridge test RMSE | Ridge test MAE | GRU test RMSE | GRU test MAE | GRU RMSE Δ vs Ridge |
|--------|---:|---:|---:|---:|---:|
| FD001 | 21.58 | 17.44 | 15.40 | 11.36 | −29% |
| FD002 | 31.31 | 22.85 | 25.08 | 16.44 | −20% |
| FD003 | 23.01 | 18.20 | 14.16 | 10.08 | −38% |
| FD004 | 32.88 | 26.20 | 25.58 | 18.22 | −22% |

The validation-set advantage carries to official test: the GRU leads by 20–38% on
RMSE. Single-condition subsets (FD001, FD003) keep a ~4–5 point val→test gap, while
multi-condition subsets (FD002, FD004) widen to ~11–12 points — the official test
distribution likely differs more from training on the multi-mode data.

## 8. Limitations and open questions

The §2–§5 feature-engineering findings come from the archived
[GRU feature sweep](archive/feature_sweep_gru_report.md), which fixed the GRU at
`hidden_size=64`, `window_size=45` (Ridge at `alpha=100`). They concern feature
choices at those fixed settings. The follow-up two-stage sweep has since varied the
sequence window (Stage 1) and the capacity (Stage 2): it confirmed `hidden_size=64`
as the best width on all four subsets and produced the refined per-subset best RMSE
used in §1 (see [gru_capacity_sweep_report.md](gru_capacity_sweep_report.md)). The
lag family was excluded from that sweep on the strength of the §3 finding, so the
window-direction and lag conclusions are consistent with — but not re-measured by —
the newer sweep.

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
