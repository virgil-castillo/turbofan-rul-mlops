# Workflow: `turbofan-predict`

What happens during batch inference, from invocation to a written predictions
CSV. Resolves the production model from the registry and runs it over a CSV/JSON
input file.

```mermaid
flowchart TD
    START(["$ turbofan-predict --model turbofan-gru-fd001 --input x.csv ..."])

    subgraph SETUP["Setup"]
        direction TB
        ARGS["Parse args + setup logging"]
    end

    subgraph READ["Read input"]
        direction TB
        DETECT{".csv or .json?"}
        CSV["Read CSV, coerce identifier / feature cells"]
        JSON["Read JSON record list"]
        DETECT -->|csv| CSV
        DETECT -->|json| JSON
    end

    subgraph LOAD["Load model"]
        direction TB
        CONF["registry.tracking.configure_mlflow"]
        RESOLVE["Resolve model:/<name>@<alias><br/>load_predictor -> PyfuncPredictor"]
        CONF --> RESOLVE
    end

    subgraph PREDICT["Predict"]
        direction TB
        VALIDATE["Validate raw records (strict schema)"]
        RUN["pyfunc model.predict<br/>(feature pipeline + model inside)"]
        ROWS["Build PredictionRows + metadata"]
        VALIDATE --> RUN --> ROWS
    end

    subgraph EVAL["Evaluate (optional)"]
        direction TB
        CHECK{"RUL labels<br/>available?"}
        METRICS["metrics.official_test_metrics<br/>RMSE, MAE, PHM08"]
        SKIP["Skip evaluation"]
        CHECK -->|yes| METRICS
        CHECK -->|no| SKIP
    end

    subgraph WRITE["Write output"]
        direction TB
        PREDOUT["Serialize + write predictions CSV"]
        METAOUT["Write metadata JSON (+ evaluation)"]
        PREDOUT --> METAOUT
    end

    START --> ARGS --> DETECT
    CSV --> CONF
    JSON --> CONF
    RESOLVE --> VALIDATE
    ROWS --> PREDOUT
    METAOUT --> CHECK
    METRICS --> DONE(["predictions.csv + metadata.json<br/>printed summary"])
    SKIP --> DONE
```

## Step reference

| Step | Function | Module |
|---|---|---|
| Read records | `_read_records` | `cli/predict.py` |
| Resolve model | `registry.load_predictor` / `load_predictor_from_uri` | `registry/` |
| Validate | `validation.validate_raw_records` | `predictions/validation.py` |
| Predict | `PyfuncPredictor.predict` | `predictions/predictor.py` |
| Serialize | `serialization.prediction_result_to_dict` | `predictions/serialization.py` |
| Evaluate | `metrics.official_test_metrics` | `evaluation/metrics.py` |
