# `turbofan.predictions` module usage

```mermaid
flowchart TD
    compute_py["compute.py<br/>ridge_engine_predictions(),<br/>sequence_final_window_predictions(),<br/>DEFAULT_MAX_RUL"]
    registry_pyfunc_py["registry/pyfunc.py<br/>RidgeEngineModel, SequenceFinalWindowModel"]

    registry_pyfunc_py -->|calls to compute RUL from a trained model + data frame| compute_py
```

`compute.py` is pure, framework-free math: it takes an already-trained
model object and a data frame and returns RUL predictions. It has no
MLflow or FastAPI knowledge. Its only consumer is `registry/pyfunc.py`,
which uses it to implement the `predict()` methods of its MLflow
`PythonModel` wrappers (`RidgeEngineModel`, `SequenceFinalWindowModel`).
