> **Archived / superseded (2026-06-01).** This is the original single-dataset
> (FD001) GRU architecture sweep, run before the unified feature pipeline. It is
> superseded by the two-stage GRU sweep (sequence window in Stage 1, hidden size /
> learning rate in Stage 2 — see
> [`../gru_capacity_sweep_report.md`](../gru_capacity_sweep_report.md)), which
> covers all four subsets with the short-engine padding fix. Kept for provenance
> only; do not cite for current results.

# GRU Sweep Report

Source files:

- `results/baselines/archive/gru_sweep.csv`
- `results/baselines/training_log.jsonl` for training duration

Ranking is by validation RMSE (lower is better); the PHM08 score is not used to
rank validation runs.

## Recommendation

Use the top-ranked configuration:

| window_size | hidden_size | learning_rate | best_epoch | RMSE | MAE | training time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 45 | 128 | 0.001 | 5 | 11.664 | 8.730 | 62.2s |

By RMSE this is the best configuration in the sweep, and it is also fast: it
reaches its best validation epoch at epoch 5 and trains in ~62s. The leaner
alternative is the fourth-ranked run:

| rank | window_size | hidden_size | learning_rate | best_epoch | RMSE | MAE | training time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45 | 128 | 0.001 | 5 | 11.664 | 8.730 | 62.2s |
| 4 | 45 | 64 | 0.001 | 3 | 11.924 | 9.167 | 52.7s |

The fourth-ranked run is ~15% faster and gives up only 0.26 RMSE points (~2.2%),
converging at epoch 3. It is a reasonable lean default for fast iteration, but the
128-unit model is the better production candidate on quality at modest extra cost.

## Sweep Summary

The sweep evaluated 36 GRU configurations across:

- Window sizes: 15, 20, 30, 45
- Hidden sizes: 32, 64, 128
- Learning rates: 0.001, 0.0005, 0.0001

Validation RMSE is the primary selection metric, where lower is better.

## Top Configurations

| rank | window_size | hidden_size | learning_rate | best_epoch | RMSE | MAE | training time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45 | 128 | 0.001 | 5 | 11.664 | 8.730 | 62.2s |
| 2 | 45 | 128 | 0.0005 | 9 | 11.698 | 8.359 | 78.0s |
| 3 | 45 | 128 | 0.0001 | 20 | 11.739 | 8.680 | 131.9s |
| 4 | 45 | 64 | 0.001 | 3 | 11.924 | 9.167 | 52.7s |
| 5 | 45 | 32 | 0.0001 | 46 | 12.032 | 8.944 | 148.8s |

The whole top tier uses a 45-cycle window. Hidden size and learning rate matter,
but window size dominates this sweep.

## Parameter Effects

### Window size

| window_size | runs | average RMSE | best RMSE | worst RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 15 | 9 | 16.498 | 16.341 | 16.692 |
| 20 | 9 | 15.294 | 15.142 | 15.462 |
| 30 | 9 | 13.316 | 13.098 | 13.619 |
| 45 | 9 | 11.952 | 11.664 | 12.196 |

Longer context consistently improves validation performance. The 45-cycle
window is the clear winner.

### Hidden size

| hidden_size | runs | average RMSE | best RMSE | worst RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 12 | 14.275 | 12.032 | 16.545 |
| 64 | 12 | 14.366 | 11.924 | 16.609 |
| 128 | 12 | 14.154 | 11.664 | 16.692 |

Hidden size has a weaker effect than window size. The 128-unit model has both the
best average and the single best score, so it is the preferred capacity once the
window is fixed at 45.

### Learning rate

| learning_rate | runs | average RMSE | best RMSE | worst RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 0.0001 | 12 | 14.230 | 11.739 | 16.462 |
| 0.0005 | 12 | 14.285 | 11.698 | 16.692 |
| 0.001 | 12 | 14.280 | 11.664 | 16.577 |

Learning rate has almost no effect on the average. The differences are dominated by
window size; `0.001` produces the single best run and converges fastest, so it is a
good default.

## Next Step

Train the production candidate with:

```powershell
turbofan-train-sequence
```

using these config values:

```yaml
sequence:
  window_size: 45
  hidden_size: 128
  learning_rate: 0.001
```

Keep the `hidden_size=64`, `learning_rate=0.001` configuration as the lean,
faster-iterating fallback (~2.2% higher RMSE, ~15% less training time).
