# Workflow: `turbofan-serve-api`

What happens when you start the FastAPI inference service: build the app, load
one registered model at startup, then serve prediction requests over the same
inference core used by [`predict`](predict.md).

```mermaid
flowchart TD
    START(["$ turbofan-serve-api --model turbofan-gru-fd001 --port 8000"])

    subgraph SETUP["Setup"]
        direction TB
        ARGS["Parse args (host, port, model, alias)"]
    end

    subgraph BUILD["Build app"]
        direction TB
        RESOLVE["Resolve model name/alias<br/>(args or TURBOFAN_MODEL_* env)"]
        CONF["registry.tracking.configure_mlflow<br/>load_predictor -> PyfuncPredictor"]
        APP["Create FastAPI app<br/>register /health + /predict routes"]
        RESOLVE --> CONF --> APP
    end

    subgraph RUN["Run server"]
        direction TB
        UVICORN["uvicorn.run(app)<br/>listen on host:port"]
    end

    subgraph REQUEST["Request handling (per call)"]
        direction TB
        POST["POST /predict (validated body)"]
        PREDICT["loaded_predictor.predict"]
        OK["Serialize response (200)"]
        ERR422["SchemaValidationError -> 422"]
        ERR500["other error -> 500"]
        POST --> PREDICT
        PREDICT --> OK
        PREDICT --> ERR422
        PREDICT --> ERR500
    end

    START --> ARGS --> RESOLVE
    APP --> UVICORN
    UVICORN -.serves.-> POST
```

## Step reference

| Step | Function | Module |
|---|---|---|
| Create app | `service.create_app` | `serving/service.py` |
| Resolve model | `registry.load_predictor` | `registry/` |
| Predict (per request) | `PyfuncPredictor.predict` | `predictions/predictor.py` |
| Validation error → 422 | `validation.SchemaValidationError` | `predictions/validation.py` |
| Serialize response | `serialization.prediction_result_to_dict` | `predictions/serialization.py` |
