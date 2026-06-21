# `turbofan.serving` Package Modules

`serving/` is the FastAPI transport adapter only. Inference contracts,
validation, the pyfunc predictor adapter, and result serialization live inward
in `turbofan.predictions`.

```mermaid
flowchart TD
    subgraph service_py["service.py"]
        SV1["create_app()\nFastAPI factory; lazily resolves a predictor\nfrom turbofan.registry by name/alias"]
        SV2["PredictRequest (pydantic)\nPOST /predict, GET /health routes"]
        SV3["_resolve_predictor()\nregistry.load_predictor(name, alias)"]
    end

    subgraph init_py["__init__.py"]
        I1["Public surface: create_app only"]
    end

    service_py -->|"PredictionResult/Metadata, RawRecords"| ContractsNote["turbofan.predictions.contracts"]
    service_py -->|"SchemaValidationError"| ValidationNote["turbofan.predictions.validation"]
    service_py -->|"prediction_result_to_dict()"| SerializationNote["turbofan.predictions.serialization"]
    service_py -.->|"lazy import: load_predictor"| RegistryNote["turbofan.registry"]
    init_py -->|"re-exports create_app"| service_py
```
