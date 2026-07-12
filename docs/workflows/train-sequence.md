# Workflow: `turbofan-train-sequence`

What happens when you run the sequence-training command, from invocation to a
registered model version.

```mermaid
flowchart TD
    START(["$ turbofan-train-sequence --config configs/default.yaml"])

    subgraph SETUP["Setup"]
        direction TB
        CFG["Load project config (YAML + _base_)<br/>resolve architecture, seeds, hyperparameters"]
        DEVICE["Resolve torch device (cpu / cuda)"]
        CFG --> DEVICE
    end

    subgraph PREP["Preprocessing"]
        direction TB
        SPLIT["Load raw train files + compute RUL labels<br/>engine-level train / validation split"]
        FEAT["Fit feature pipeline on train, transform val<br/>SensorDropper -> OperatingModeNormalizer -><br/>SensorColumnSelector -> FeatureEngineer -> StandardScaler"]
        WIN["Build sliding windows per engine"]
        LOADER["Build train (shuffled) + val (sequential) loaders"]
        SPLIT --> FEAT --> WIN --> LOADER
    end

    subgraph TRAIN["Training"]
        direction TB
        SEED["Seed everything (model seed)"]
        BUILD["Build GRU / LSTM SequenceRULRegressor"]
        LOOP["Training loop: packed sequences,<br/>early stopping, restore best epoch"]
        SEED --> BUILD --> LOOP
    end

    subgraph EVAL["Evaluation"]
        direction TB
        EVALV["Evaluate validation windows<br/>RMSE, MAE"]
        OFFICIAL{"Official test<br/>files present?"}
        OFFEVAL["Predict final-cycle official test<br/>RMSE, MAE, PHM08 score"]
        SKIP["Skip official eval"]
        EVALV --> OFFICIAL
        OFFICIAL -->|yes| OFFEVAL
        OFFICIAL -->|no| SKIP
    end

    subgraph RUN["Logging & registration"]
        direction TB
        MLF["Start MLflow run<br/>log params, metrics, history, tags"]
        ARTIFACTS["Save run artifacts<br/>metrics.json, config.json,<br/>training_history.csv, predictions.csv"]
        PAYLOAD["Build model checkpoint payload<br/>(state dict, feature pipeline, normalizer,<br/>sequence config, max_rul, seed)"]
        REGISTER["Register model version<br/>turbofan-<arch>-<subset>"]
        ARTIFACTS --> PAYLOAD --> REGISTER
    end

    START --> CFG
    DEVICE --> SPLIT
    LOADER --> SEED
    LOOP --> EVALV --> MLF
    MLF --> OFFICIAL
    OFFEVAL --> ARTIFACTS
    SKIP --> ARTIFACTS
    REGISTER --> DONE(["Registered model version<br/>printed metrics"])
```

## Step reference

| Step | Function | Module |
|---|---|---|
| Load config | `schema.load_config` | `config/schema.py` |
| Resolve device | `sequence_training.resolve_device` | `training/sequence_training.py` |
| Prepare data | `sequence_pipeline.prepare_sequence_data` | `training/sequence_pipeline.py` |
| Train | `sequence_pipeline.train_prepared_sequence` | `training/sequence_pipeline.py` |
| Validation eval | `sequence_pipeline.evaluate_window_metrics` | `training/sequence_pipeline.py` |
| Official-test eval | `sequence_pipeline.predict_sequence_official` | `training/sequence_pipeline.py` |
| Log run | `registry.tracking.log_*` + `mlflow.*` | `cli/train_sequence.py`, `registry/tracking.py` |
| Register | `registry.log_and_register` | `registry/__init__.py` |
