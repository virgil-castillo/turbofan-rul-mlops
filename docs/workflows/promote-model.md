# Workflow: `turbofan-promote`

What happens when promoting a registered model version to an alias (e.g.
`@production`). Repoints the alias immediately with no approval gate; rollback is
the same operation against an earlier version.

```mermaid
flowchart TD
    START(["$ turbofan-promote turbofan-gru-fd001 3 --to production"])

    subgraph SETUP["Setup"]
        direction TB
        ARGS["Parse args (name, version, --to alias)"]
        CONF["registry.tracking.configure_mlflow"]
        ARGS --> CONF
    end

    subgraph PROMOTE["Promote"]
        direction TB
        SET["registry.promote(name, version, alias)<br/>repoint alias -> version"]
        ERR{"Error?"}
        SET --> ERR
    end

    subgraph CONFIRM["Confirm"]
        direction TB
        URI["registry.resolve_uri(name, alias)"]
        PRINT["Print alias -> version + resolved URI"]
        URI --> PRINT
    end

    START --> ARGS
    CONF --> SET
    ERR -->|no| URI
    ERR -->|yes| FAIL(["log error, exit 1"])
    PRINT --> DONE(["alias repointed"])
```

## Step reference

| Step | Function | Module |
|---|---|---|
| Configure MLflow | `registry.tracking.configure_mlflow` | `registry/tracking.py` |
| Promote | `registry.promote` | `registry/` |
| Resolve URI | `registry.resolve_uri` | `registry/` |
