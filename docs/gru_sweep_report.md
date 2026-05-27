# GRU Sweep Report

Source files:

- `results/gru_sweep.csv`
- `results/training_log.jsonl` for training duration

## Recommendation

Use the second-ranked configuration:

| window_size | hidden_size | learning_rate | best_epoch | RMSE | MAE | PHM08 score | training time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 45 | 64 | 0.001 | 3 | 11.924 | 9.167 | 6253.4 | 52.7s |

This is the best speed-quality tradeoff in the sweep. The top-ranked
configuration has a slightly better PHM08 score, but takes much longer:

| rank | window_size | hidden_size | learning_rate | best_epoch | RMSE | MAE | PHM08 score | training time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45 | 128 | 0.0001 | 20 | 11.739 | 8.680 | 6161.1 | 131.9s |
| 2 | 45 | 64 | 0.001 | 3 | 11.924 | 9.167 | 6253.4 | 52.7s |

The second-ranked run is about 2.5x faster while giving up only 92.3 PHM08
points, or about 1.5% relative to the best score. It also reaches its best
validation epoch at epoch 3, which makes it a better default for iteration.

## Sweep Summary

The sweep evaluated 36 GRU configurations across:

- Window sizes: 15, 20, 30, 45
- Hidden sizes: 32, 64, 128
- Learning rates: 0.001, 0.0005, 0.0001

PHM08 score is the primary selection metric, where lower is better.

## Top Configurations

| rank | window_size | hidden_size | learning_rate | best_epoch | RMSE | MAE | PHM08 score | training time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45 | 128 | 0.0001 | 20 | 11.739 | 8.680 | 6161.1 | 131.9s |
| 2 | 45 | 64 | 0.001 | 3 | 11.924 | 9.167 | 6253.4 | 52.7s |
| 3 | 45 | 64 | 0.0005 | 6 | 12.138 | 9.217 | 6391.4 | 53.6s |
| 4 | 45 | 32 | 0.0001 | 46 | 12.032 | 8.944 | 6562.3 | 148.8s |
| 5 | 45 | 128 | 0.001 | 5 | 11.664 | 8.730 | 6565.4 | 62.2s |

The whole top tier uses a 45-cycle window. Hidden size and learning rate matter,
but window size dominates this sweep.

## Parameter Effects

### Window size

| window_size | runs | average PHM08 | best PHM08 | worst PHM08 |
| ---: | ---: | ---: | ---: | ---: |
| 15 | 9 | 26939.9 | 21256.6 | 30999.7 |
| 20 | 9 | 17591.1 | 15926.0 | 20137.0 |
| 30 | 9 | 10657.6 | 9499.0 | 11767.3 |
| 45 | 9 | 6669.1 | 6161.1 | 7507.4 |

Longer context consistently improves validation performance. The 45-cycle
window is the clear winner.

### Hidden size

| hidden_size | runs | average PHM08 | best PHM08 | worst PHM08 |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 12 | 15716.6 | 6562.3 | 28167.6 |
| 64 | 12 | 15475.0 | 6253.4 | 29085.0 |
| 128 | 12 | 15201.7 | 6161.1 | 30999.7 |

Hidden size has a weaker effect than window size. The 128-unit model produces
the single best score, but 64 units is nearly as strong and trains faster in the
recommended configuration.

### Learning rate

| learning_rate | runs | average PHM08 | best PHM08 | worst PHM08 |
| ---: | ---: | ---: | ---: | ---: |
| 0.0001 | 12 | 14488.2 | 6161.1 | 27146.8 |
| 0.0005 | 12 | 15627.3 | 6391.4 | 28626.1 |
| 0.001 | 12 | 16277.8 | 6253.4 | 30999.7 |

The lower learning rate has the best average score, but the recommended
configuration shows that 0.001 can converge quickly when paired with the
45-cycle window and 64 hidden units.

## Next Step

Train the production candidate with:

```powershell
turbofan-train-sequence-gru
```

using these config values:

```yaml
sequence:
  window_size: 45
  hidden_size: 64
  learning_rate: 0.001
```

Keep the top-ranked `hidden_size=128`, `learning_rate=0.0001` configuration as
the accuracy-oriented fallback if final test-set performance is materially
better.
