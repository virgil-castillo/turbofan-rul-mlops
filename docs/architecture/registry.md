# Registry Architecture

`turbofan.registry` is the MLflow infrastructure adapter. Its public facade
re-exports package operations; model packaging lives in `pyfunc.py`, and
registry I/O lives in `store.py`. Registering a model creates a version only.
Promotion is a separate, explicit alias operation.

```mermaid
flowchart TD
    subgraph FACADE["registry/__init__.py (facade)"]
        EXPORTS["__all__ re-exports:<br/>RidgeEngineModel, SequenceFinalWindowModel,<br/>log_and_register, load, load_predictor,<br/>promote, list_registered, model_name, ..."]
    end

    subgraph PYFUNC["registry/pyfunc.py - model packaging"]
        RIDGEM["RidgeEngineModel<br/>(PythonModel)"]
        SEQM["SequenceFinalWindowModel<br/>(PythonModel)"]
        LOGREG["log_and_register<br/>log pyfunc model -> register version"]
        SIG["_signature / _sample_input<br/>infer MLflow ModelSignature"]
        LOGRIDGE["_log_ridge_model"]
        LOGSEQ["_log_sequence_model"]
        LOGREG --> LOGRIDGE
        LOGREG --> LOGSEQ
        LOGRIDGE -.->|wraps| RIDGEM
        LOGSEQ -.->|wraps| SEQM
        SIG -.-> LOGRIDGE
        SIG -.-> LOGSEQ
    end

    subgraph STORE["registry/store.py - registry I/O"]
        MNAME["model_name<br/>turbofan-&lt;arch&gt;-&lt;subset&gt;"]
        LATEST["latest_version<br/>read registered versions"]
        PROMOTE["promote<br/>version -> alias"]
        RESOLVE["resolve_uri<br/>models:/&lt;name&gt;@&lt;alias&gt;"]
        LOAD["load<br/>mlflow.pyfunc.load_model"]
        LOADPRED["load_predictor /<br/>load_predictor_from_uri"]
        LISTREG["list_registered<br/>+ RegisteredModelInfo"]
        RESOLVE --> LOAD --> LOADPRED
        PROMOTE -.->|alias is later resolved by| RESOLVE
    end

    TRAIN["turbofan-train-*<br/>(inside an active MLflow run)"] -->|registry.log_and_register| LOGREG
    LOGREG -->|registers a new version| MLREG[("MLflow registry<br/>(SQLite store)")]
    MLREG -.-> LATEST
    MLREG -.-> LOAD

    PROMOTECLI["turbofan-promote"] -->|registry.promote| PROMOTE
    PREDICTCLI["turbofan-predict"] -->|registry.load_predictor| LOADPRED
    SERVECLI["turbofan-serve-api"] -->|registry.load_predictor| LOADPRED
    LOADPRED -->|wraps loaded pyfunc model| PRED["PyfuncPredictor"]

    EXPORTS -.->|re-exports| LOGREG
    EXPORTS -.->|re-exports| PROMOTE
    EXPORTS -.->|re-exports| LOADPRED
    EXPORTS -.->|re-exports| LISTREG
    EXPORTS -.->|re-exports| MNAME
```

## Module-level ownership

```mermaid
flowchart TD
    subgraph store_py["store.py"]
        S1["model_name() / resolve_uri()<br/>naming and models: URI construction"]
        S2["load() / load_predictor() / load_predictor_from_uri()<br/>load a pyfunc model and wrap it in PyfuncPredictor"]
        S3["model_type_from_name() / parse_models_uri()<br/>parse model names and URIs"]
        S4["latest_version()<br/>read the highest registered version"]
        S5["promote()<br/>repoint a registered-model alias"]
        S6["list_registered()<br/>versions, @production, val_rmse, and run provenance"]
    end

    subgraph pyfunc_py["pyfunc.py"]
        P1["RidgeEngineModel(PythonModel)<br/>validate raw records -> ridge_engine_predictions"]
        P2["SequenceFinalWindowModel(PythonModel)<br/>validate raw records -> sequence_final_window_predictions"]
        P4["log_and_register()<br/>package a Ridge or sequence model<br/>and register a new version"]
        P5["_signature() / _sample_input()<br/>canonical MLflow ModelSignature"]
    end

    subgraph init_py["__init__.py"]
        IN1["Public facade<br/>re-exports store.py + pyfunc.py"]
    end

    P1 -->|engine-scope predictions via| ComputeNote["turbofan.predictions.compute"]
    P2 -->|final-window predictions via| ComputeNote
    P1 -->|validate_raw_records| ValidationNote["turbofan.predictions.validation"]
    P2 -->|validate_raw_records| ValidationNote
    P4 -->|calls| S1
    P4 -->|reads registered version with| S4
    P5 -.->|used by model logging| P4
    S2 -->|wraps loaded pyfunc model in| PyfuncPredictorNote["turbofan.predictions.predictor.PyfuncPredictor"]
    init_py -->|re-exports| store_py
    init_py -->|re-exports| pyfunc_py
```
