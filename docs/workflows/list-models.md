# Workflow: `turbofan-models`

What happens when listing registered models: query the registry and print a
greppable table (name, versions, `@production` version, production `val_rmse`,
run link).

```mermaid
flowchart TD
    START(["$ turbofan-models"])

    subgraph SETUP["Setup"]
        direction TB
        ARGS["Parse args + setup logging"]
        CONF["registry.tracking.configure_mlflow"]
        ARGS --> CONF
    end

    subgraph QUERY["Query"]
        direction TB
        LIST["registry.list_registered()<br/>-> RegisteredModelInfo[]"]
        EMPTY{"Any models?"}
        LIST --> EMPTY
    end

    subgraph RENDER["Render"]
        direction TB
        CELLS["Build row cells per model<br/>(versions, production, val_rmse, run_link)"]
        TABLE["Align columns + print table"]
        CELLS --> TABLE
    end

    START --> ARGS
    CONF --> LIST
    EMPTY -->|no| NONE(["print 'No registered models.'"])
    EMPTY -->|yes| CELLS
    TABLE --> DONE(["printed model table"])
```

## Step reference

| Step | Function | Module |
|---|---|---|
| Configure MLflow | `registry.tracking.configure_mlflow` | `registry/tracking.py` |
| List models | `registry.list_registered` | `registry/` |
| Resolve run link | `registry.resolve_uri` | `registry/` |
