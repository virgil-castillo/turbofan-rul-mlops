# Feature Engineering for Ridge vs. GRU — Comparison

A comparison of what the two models require from feature engineering. Ridge detail
is in [feature_sweep_ridge_report.md](feature_sweep_ridge_report.md); the current
sequence-model reference is the
[feature-family screen](feature_family_screen_report.md), whose winning
configurations are the production models behind the committed snapshot
(`results/baselines/latest_official_eval_summary.csv`) used in §1 and §7. The GRU
feature-engineering mechanism figures used in §2–§5 come from the now-archived
[GRU feature sweep](archive/feature_sweep_gru_report.md) (fixed `hidden_size=64`,
`window_size=45`); that sweep predates the pipeline scaling change — see §8.
Ranking is by validation RMSE; the PHM08 score appears only for
official-test results (§7).

## 1. Overall performance: the GRU cuts Ridge's validation error by 19–50%

Validation RMSE per subset for the production configurations, from the committed
snapshot (`results/baselines/latest_official_eval_summary.csv`): Ridge uses the
deployed feature-sweep config (`alpha=100`), the GRU the winning cell from the
[feature-family screen](feature_family_screen_report.md). Both use the same
engine-level split and `max_rul=125`. GRU figures are mean ± sd over five model
seeds (42–46); Ridge is deterministic at the production seed:

| Dataset | Ridge RMSE | GRU RMSE | RMSE Δ |
|---|---:|---:|---:|
| FD001 | 20.72 | 10.39 ± 0.33 | −50% |
| FD002 | 19.35 | 13.08 ± 0.15 | −32% |
| FD003 | 17.07 | 10.46 ± 0.22 | −39% |
| FD004 | 18.48 | 14.96 ± 0.10 | −19% |

The GRU halves RMSE on FD001, cuts it by roughly a third on FD002/FD003, and by
19% on FD004 (6 modes, longest lifetimes), where the two models come closest.

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

## 3. The lag family: no value for either model

From the archived pre-scaling sweep:

| | Ridge | GRU |
|---|---|---|
| `lag` alone | Catastrophic (RMSE ~40–42) | Catastrophic (RMSE ~26–31) |
| `raw_plus_lag` vs `raw` | Neutral (≈ equal to `raw`) | Harmful (RMSE ~2× `raw`) |
| Mechanism | First-difference removes the absolute level [4]; a memoryless linear fit [1] has nothing to anchor to | Off-scale lag channels degrade input conditioning [5]; training fails to improve and early-stops at epoch 1–3 [6] |

For Ridge the lag-difference carries no marginal information once `raw` is present.
The GRU column above is a pre-scaling measurement: the unscaled lag channels were
off-scale relative to the normalized raw inputs, and training failed to improve
(early-stop at epoch 1–3 vs 5–17 for working sets). The post-scaling
[feature-family screen](feature_family_screen_report.md) re-tested lag on scaled
inputs and the catastrophic non-convergence does not reproduce: lag is neutral to
mildly harmful there (all Δ vs raw between −0.51 and +1.46 across both
architectures and all subsets). The selection conclusion is the same in both
eras — lag earns no place in either model's feature set.

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
- lag adds nothing — pre-scaling it destabilized training outright [5, 6], and
  after scaling it remains neutral to mildly harmful (§3);
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
final-test metric. Values are the committed snapshot
(`results/baselines/latest_official_eval_summary.csv`), written by
`turbofan-regenerate-baselines`, which retrains each selected configuration and
evaluates it on the official test set: GRU as mean ± sd over five model seeds
(42–46), Ridge at the single production seed:

| Subset | Ridge test RMSE | Ridge test PHM08 | GRU test RMSE | GRU test PHM08 | GRU RMSE Δ vs Ridge |
|--------|---:|---:|---:|---:|---:|
| FD001 | 21.58 | 1,316 | 14.29 ± 0.11 | 330 ± 13 | −34% |
| FD002 | 31.30 | 17,700 | 24.34 ± 0.40 | 4,992 ± 501 | −22% |
| FD003 | 23.01 | 2,491 | 14.22 ± 0.17 | 414 ± 22 | −38% |
| FD004 | 32.88 | 9,643 | 26.00 ± 0.41 | 4,603 ± 507 | −21% |

The validation-set advantage carries to official test: the GRU leads by 21–38% on
RMSE. The GRU's val→test RMSE gap is ~4 points on the single-condition subsets
(FD001, FD003) and ~11 points on the multi-condition subsets (FD002, FD004) —
consistent with a larger train/test distribution shift on the multi-mode data (an
interpretation; the measured quantity is the gap).

## 8. Limitations and open questions

The §2–§5 feature-engineering findings come from the archived
[GRU feature sweep](archive/feature_sweep_gru_report.md), which fixed the GRU at
`hidden_size=64`, `window_size=45` (Ridge at `alpha=100`). Two boundaries apply.

First, they concern feature choices at those fixed settings; the follow-up
[two-stage sweep](gru_capacity_sweep_report.md) varied the sequence window and
capacity and found `hidden_size=64` the best width on all four subsets.

Second, the archived sweep, the Ridge sweep, and the two-stage sweep all ran
before commit `3e5afc8` (2026-06-05) added standard-scaling of the engineered
pipeline output, so their absolute numbers describe the pre-scaling pipeline. The
post-scaling [feature-family screen](feature_family_screen_report.md) re-measured
the sequence side on scaled inputs and now defines the production GRU/LSTM
configurations; §1 and §7 report those models from the committed snapshot. Where
the screen re-tested a §2–§5 mechanism, its verdict supersedes: the lag family's
"catastrophic for the GRU" result softened to neutral-to-mildly-harmful (§3),
while the wide-rolling-window direction for sequence models was confirmed (the
screen's window-20 families are the ones that help). The Ridge sweep has not been
re-run since the scaling change, so the Ridge mechanisms in §2–§5 stand as
pre-scaling observations (the re-run is tracked in the
[roadmap](roadmap.md)); the Ridge production configs and official numbers are
current as committed.

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
