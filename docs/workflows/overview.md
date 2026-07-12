# Workflow Overview: End-to-End Pipeline

End-to-end view of the turbofan RUL workflow, from raw C-MAPSS data loading
through training, the MLflow registry, and inference/serving. The Docker
deployment path is intentionally omitted.

This is the big picture the per-command workflow docs zoom into: training maps to
[`train-baseline`](train-baseline.md) / [`train-sequence`](train-sequence.md),
registry promotion to [`promote-model`](promote-model.md), and inference to
[`predict`](predict.md) / [`serve-api`](serve-api.md). See the
[index](README.md) for all commands.

The shared preprocessing contract is
`SensorDropper → OperatingModeNormalizer → SensorColumnSelector → FeatureEngineer → StandardScaler`
(`build_feature_pipeline`), used identically by the Ridge baseline, the
sequence models, and inference.

```mermaid
flowchart TD
    %% ============ Data loading ============
    subgraph DATA["Data loading (turbofan.data)"]
        RAW["C-MAPSS raw .txt files<br/>train / test / RUL<br/>FD001-FD004"]
        LOADTR["load_raw_train<br/>engine_id, cycle, op_1-3, s_1-21"]
        LOADTE["load_raw_test"]
        LOADRUL["load_rul_labels<br/>one RUL per test engine"]
        LABEL["compute_rul_labels / add_rul_column<br/>RUL = min(max_rul, max_cycle - cycle)<br/>piecewise-linear, capped at 125"]
        RAW --> LOADTR --> LABEL
        RAW --> LOADTE
        RAW --> LOADRUL
    end

    %% ============ Config ============
    CFG["ProjectConfig (YAML + _base_)<br/>data / features / model / sequence<br/>per-subset + per-architecture"]

    %% ============ Split ============
    SPLIT["split_by_engine<br/>engine-level train/val split<br/>seed = data.random_seed"]
    LABEL --> SPLIT

    %% ============ Feature pipeline ============
    subgraph FEAT["Shared feature pipeline (build_feature_pipeline)"]
        F1["SensorDropper<br/>drop EDA-flagged sensors"]
        F2["OperatingModeNormalizer<br/>KMeans op-mode clusters,<br/>per-mode z-score of s_*"]
        F3["SensorColumnSelector<br/>keep s_* + engine_id"]
        F4["FeatureEngineer<br/>raw / rolling_* / lag families<br/>per engine, no boundary crossing"]
        F5["StandardScaler"]
        F1 --> F2 --> F3 --> F4 --> F5
    end
    SPLIT -->|fit_transform train<br/>transform val| FEAT
    CFG -.->|feature families,<br/>windows, lag, n_modes| FEAT

    %% ============ Branch on model type ============
    BRANCH{"model type<br/>(config)"}
    FEAT --> BRANCH

    %% ---- Ridge path ----
    subgraph RIDGE["Ridge baseline path"]
        RTRAIN["build_ridge_estimator + Ridge.fit<br/>tabular features"]
    end
    BRANCH -->|ridge| RTRAIN

    %% ---- Sequence path ----
    subgraph SEQ["Sequence path (GRU / LSTM)"]
        W["build_sliding_windows<br/>window_size cycles per sample"]
        LD["build_sequence_loader<br/>train: shuffled / val: sequential"]
        BUILD["build_sequence_model<br/>SequenceRULRegressor<br/>(GRU or LSTM cell)"]
        STRAIN["train_sequence_model<br/>packed sequences, early stopping<br/>seed_everything(model_seed)"]
        W --> LD --> STRAIN
        BUILD --> STRAIN
    end
    BRANCH -->|gru / lstm| W
    CFG -.->|architecture,<br/>hyperparameters| BUILD

    %% ============ Evaluation ============
    subgraph EVAL["Evaluation"]
        VAL["validation windows / rows<br/>RMSE, MAE"]
        OFF["official C-MAPSS test<br/>final cycle per engine<br/>RMSE, MAE, PHM08"]
    end
    RTRAIN --> VAL
    STRAIN --> VAL
    LOADTE --> OFF
    LOADRUL --> OFF

    %% ============ Tracking & Registry ============
    subgraph TRACK["Experiment tracking & registry"]
        MLF["MLflow run<br/>params, metrics, history, artifacts<br/>(SQLite store)"]
        REG["log_and_register<br/>turbofan-&lt;arch&gt;-&lt;subset&gt;<br/>register a version"]
        VERSION["Registered model version"]
        PROMO["turbofan-promote<br/>explicitly repoint alias -> version"]
        REG --> VERSION
        VERSION -.->|operator selects a version| PROMO
    end
    VAL --> MLF --> OFF
    OFF --> REG

    %% ============ Inference / Serving ============
    subgraph INFER["Inference & serving"]
        RESOLVE["registry.load_predictor<br/>models:/&lt;name&gt;@&lt;alias&gt;"]
        PRED["PyfuncPredictor.predict<br/>validate -> feature pipeline -><br/>ridge_engine / sequence_final_window"]
        BATCH["turbofan-predict<br/>batch CSV in/out"]
        API["turbofan-serve-api (FastAPI)<br/>GET /health, POST /predict"]
        PRED --> BATCH
        PRED --> API
    end
    PROMO --> RESOLVE --> PRED

    %% New engine sensor records at inference time
    NEW["New engine sensor records<br/>(CSV / JSON request)"]
    NEW --> PRED
    PRED --> OUT["RUL predictions per engine"]
```

## Stage summary

| Stage | Key modules | What happens |
|---|---|---|
| Data loading | `data/loader.py`, `data/labels.py` | Read raw `.txt` files; compute capped piecewise-linear RUL labels. |
| Split | `training/split.py` | Engine-level train/validation split seeded by `data.random_seed`. |
| Features | `features/pipeline.py`, `features/engineering.py` | Shared 5-step sklearn pipeline; same fitted pipeline reused at inference. |
| Ridge | `models/baseline.py` | Linear baseline over engineered tabular features. |
| Sequence | `sequences/windowing.py`, `models/sequence_models.py`, `training/sequence_*` | Sliding windows → loaders → GRU/LSTM `SequenceRULRegressor`. |
| Evaluation | `evaluation/evaluate.py`, `evaluation/metrics.py` | Validation RMSE/MAE; official-test RMSE/MAE/PHM08 at final cycle. |
| Tracking/Registry | `registry/tracking.py`, `registry/` | Log to MLflow; register and promote model versions by alias. |
| Inference | `predictions/*`, `serving/*`, `cli/predict.py`, `cli/serve_api.py` | Resolve model by name/alias; batch CSV or FastAPI `/predict`. |
