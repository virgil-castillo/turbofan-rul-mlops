# Predictions Architecture

`predictions/` is the inward-facing inference core: contracts, validation,
RUL-compute math, the loaded-model predictor adapter, and result serialization.
It is transport- and MLflow-free. The registry, FastAPI service, and batch CLI
consume it.

```mermaid
flowchart TD
    subgraph contracts_py["contracts.py"]
        C1["ModelType / PredictionScope / RawRecords"]
        C2["dataclasses: PredictionRow, PredictionMetadata,\nPredictionResult, ValidationResult"]
    end

    subgraph validation_py["validation.py"]
        V1["validate_raw_records()\ncoerce + sort + dedupe rows,\noptionally skip invalid rows (partial)"]
        V2["SchemaValidationError"]
    end

    subgraph compute_py["compute.py"]
        CP1["ridge_engine_predictions()\nlast-cycle per engine, clip [0, max_rul]"]
        CP2["sequence_final_window_predictions()\nrebuild architecture, normalize/feature-pipeline,\nfinal window per engine, rescale by max_rul"]
    end

    subgraph predictor_py["predictor.py"]
        P1["PyfuncPredictor\nadapts a loaded pyfunc model to PredictionResult"]
        P2["_MODEL_SCOPES (ridge->engine, gru/lstm->final_window)"]
    end

    subgraph serialization_py["serialization.py"]
        S1["prediction_result_to_dict()\nJSON-compatible response dict"]
    end

    validation_py -->|"uses CANONICAL/FEATURE columns"| DataNote["turbofan.data.contracts"]
    validation_py --> contracts_py
    predictor_py -->|"validate + build rows"| validation_py
    predictor_py --> contracts_py
    serialization_py --> contracts_py
    compute_py -->|"build_sequence_model"| ModelsNote["turbofan.models.sequence_models"]

    Consumers["Consumers:\nregistry.pyfunc, serving.service, cli.predict"]
    Consumers --> contracts_py
    Consumers --> validation_py
    Consumers --> predictor_py
    Consumers --> serialization_py
    RegistryPyfunc["registry.pyfunc"] -->|"RUL math"| compute_py
```
