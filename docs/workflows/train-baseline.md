# Workflow: `turbofan-train-baseline`

What happens when you train the Ridge tabular baseline, from invocation to a
registered model version. The sibling of
[`train-sequence`](train-sequence.md) on the tabular (non-windowed) path.

```mermaid
flowchart TD
    START(["$ turbofan-train-baseline --config configs/default.yaml"])

    subgraph SETUP["Setup"]
        direction TB
        CFG["Load project config (YAML + _base_)"]
        CFG2["Resolve ridge feature config + seed"]
        CFG --> CFG2
    end

    subgraph PREP["Preprocessing"]
        direction TB
        SPLIT["Load raw train files + compute RUL labels<br/>engine-level train / validation split"]
        XY["Split features / target for train + val"]
        SPLIT --> XY
    end

    subgraph TRAIN["Training"]
        direction TB
        BUILD["Build Ridge estimator (feature pipeline + Ridge)"]
        FIT["estimator.fit(X_train, y_train)"]
        BUILD --> FIT
    end

    subgraph EVAL["Evaluation"]
        direction TB
        VALP["Predict validation (clipped to [0, max_rul])<br/>RMSE, MAE"]
        OFFICIAL{"Official test<br/>files present?"}
        OFFEVAL["Predict final-cycle official test<br/>RMSE, MAE, PHM08 score"]
        SKIP["Skip official eval"]
        VALP --> OFFICIAL
        OFFICIAL -->|yes| OFFEVAL
        OFFICIAL -->|no| SKIP
    end

    subgraph RUN["Logging & registration"]
        direction TB
        MLF["Start MLflow run<br/>log params, metrics, tags"]
        ARTIFACTS["Save run artifacts<br/>metrics.json, config.json, predictions.csv"]
        REGISTER["Register model version<br/>turbofan-ridge-<subset>"]
        ARTIFACTS --> REGISTER
    end

    START --> CFG
    CFG2 --> SPLIT
    XY --> BUILD
    FIT --> VALP --> MLF
    MLF --> OFFICIAL
    OFFEVAL --> ARTIFACTS
    SKIP --> ARTIFACTS
    REGISTER --> DONE(["Registered model version<br/>printed metrics"])
```

## Step reference

| Step | Function | Module |
|---|---|---|
| Load config | `schema.load_config` | `config/schema.py` |
| Split | `split.load_and_split` | `training/split.py` |
| Features/target | `evaluate.split_features_target` | `evaluation/evaluate.py` |
| Build estimator | `baseline.build_ridge_estimator` | `models/baseline.py` |
| Validation eval | `evaluate.predict_with_clipping` + `metrics.regression_metrics` | `evaluation/` |
| Official-test eval | `evaluate.predict_ridge_official` | `evaluation/evaluate.py` |
| Log run | `registry.tracking.log_*` + `mlflow.*` | `cli/train_baseline.py`, `registry/tracking.py` |
| Register | `registry.log_and_register` | `registry/__init__.py` |
