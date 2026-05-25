# Baseline Feature Comparison Report

## Summary

The finalized tabular baseline is:

```yaml
model:
  name: ridge
  alpha: 100.0
  feature_set: rolling
  windows:
    - 10
```

Selection is based on the PHM08 score, where lower is better. PHM08 is the
primary selection metric here because it penalizes late RUL predictions more
strongly than early predictions, matching the maintenance-risk framing better
than symmetric RMSE.

## Experiment Setup

The comparison was run with:

- dataset subset: `FD001`
- validation split: engine-level holdout from `configs/default.yaml`
- model: Ridge regression with `alpha=100.0`
- feature sets: `raw`, `raw_plus_rolling`, `rolling`
- rolling windows: `5`, `10`, `20`
- source result file: `results/baseline_feature_comparison_4294012.csv`

All model inputs exclude `engine_id`, `cycle`, and operational setting columns.
`engine_id` is used only for rolling feature grouping. `op_1`, `op_2`, and
`op_3` are used only for operational-condition normalization.

## Results Ranked by PHM08

| Rank | Feature set | Window | Features | RMSE | MAE | PHM08 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `rolling` | 10 | 56 | 24.3304 | 18.2856 | 159,969.58 |
| 2 | `rolling` | 20 | 56 | 24.4620 | 18.2805 | 166,276.79 |
| 3 | `raw_plus_rolling` | 10 | 70 | 23.9231 | 17.8796 | 183,191.62 |
| 4 | `raw_plus_rolling` | 20 | 70 | 23.5166 | 17.3502 | 226,456.70 |
| 5 | `raw_plus_rolling` | 5 | 70 | 23.8334 | 17.8009 | 469,005.33 |
| 6 | `rolling` | 5 | 56 | 23.9576 | 17.8974 | 477,216.18 |
| 7 | `raw` | none | 14 | 24.5316 | 18.2198 | 510,878.25 |

## Decision

The RMSE-best model was `raw_plus_rolling` with window `20`, but it was not the
PHM08-best model. The finalized baseline uses `rolling` with window `10`
because it achieved the lowest PHM08 score.

This choice trades a small RMSE increase for materially better asymmetric RUL
scoring:

- PHM08-best: `rolling_w10`, RMSE `24.3304`, PHM08 `159,969.58`
- RMSE-best: `raw_plus_rolling_w20`, RMSE `23.5166`, PHM08 `226,456.70`

The selected model also uses fewer estimator features, `56` instead of `70`,
which makes it a simpler baseline while improving the selected domain metric.

## Final Training Run

The finalized baseline was trained with `configs/default.yaml` after updating
the default model settings.

Artifact directory:

```text
artifacts/models/baseline/20260525-151312
```

Persisted validation metrics:

| Split | RMSE | MAE | PHM08 |
| --- | ---: | ---: | ---: |
| validation | 24.3304 | 18.2856 | 159,969.58 |
| official_test | 26.0387 | 20.5264 | 2,829.00 |

The final artifact config records:

```json
{
  "feature_set": "rolling",
  "windows": [10],
  "alpha": 100.0
}
```

## Follow-up

Raw prediction ranges are still wider than the valid RUL range before clipping.
The training and comparison pipelines clip predictions to `[0, max_rul]` before
metrics, but the raw ranges should remain visible in reports as a calibration
diagnostic.
