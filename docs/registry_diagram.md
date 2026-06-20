# Registry Package

`registry/__init__.py` is a pure re-export facade preserving the historical
`turbofan.registry` import path; the real logic lives in `pyfunc.py` (model
packaging) and `store.py` (registry I/O). Callers outside the package import
only from `registry`, never from the submodules directly.

```mermaid
flowchart TD
    subgraph FACADE["registry/__init__.py (facade)"]
        EXPORTS["__all__ re-exports:<br/>RidgeEngineModel, SequenceFinalWindowModel,<br/>log_and_register, load, load_predictor,<br/>promote, list_registered, model_name, ..."]
    end

    subgraph PYFUNC["registry/pyfunc.py — model packaging"]
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
        SIG -.-> LOGREG
    end

    subgraph STORE["registry/store.py — registry I/O"]
        MNAME["model_name<br/>turbofan-&lt;arch&gt;-&lt;subset&gt;"]
        LATEST["latest_version"]
        PROMOTE["promote<br/>version -> alias"]
        RESOLVE["resolve_uri<br/>models:/&lt;name&gt;@&lt;alias&gt;"]
        LOAD["load<br/>mlflow.pyfunc.load_model"]
        LOADPRED["load_predictor /<br/>load_predictor_from_uri"]
        LISTREG["list_registered<br/>+ RegisteredModelInfo"]
        MNAME --> LATEST --> PROMOTE
        RESOLVE --> LOAD --> LOADPRED
        PROMOTE -.-> RESOLVE
    end

    TRAIN["turbofan-train-*<br/>(end of training run)"] -->|EXPORTS.log_and_register| LOGREG
    LOGREG -->|registers version| MLREG[("MLflow registry<br/>(SQLite store)")]
    MLREG -.-> LATEST
    MLREG -.-> LOAD

    CLI["turbofan-promote / turbofan-predict /<br/>turbofan-serve-api"] -->|EXPORTS.promote, .load_predictor| PROMOTE
    CLI -->|EXPORTS.load_predictor| LOADPRED
    LOADPRED -->|wraps loaded pyfunc model| PRED["PyfuncPredictor"]

    EXPORTS -.->|re-exports| LOGREG
    EXPORTS -.->|re-exports| PROMOTE
    EXPORTS -.->|re-exports| LOADPRED
    EXPORTS -.->|re-exports| LISTREG
    EXPORTS -.->|re-exports| MNAME
```
