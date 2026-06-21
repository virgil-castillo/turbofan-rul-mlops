# Workflow: `turbofan-feature-screen`

What happens during the feature-family sweep: enumerate a grid of
(subset × architecture × feature config × seed) cells and, for each, run the
shared sequence pipeline and append a result row. Resumable — completed cells
are skipped.

```mermaid
flowchart TD
    START(["$ turbofan-feature-screen --subsets FD001 ... --architectures gru lstm"])

    subgraph SETUP["Setup"]
        direction TB
        ARGS["Parse sweep params + setup logging"]
        DELEGATE["feature_family_screen.run_screen(...)"]
        ARGS --> DELEGATE
    end

    subgraph GRID["Build grid"]
        direction TB
        ENUM["enumerate_cells<br/>(subset x arch x feature config x seed)"]
        RESUME["Load completed cell keys from results CSV"]
        ENUM --> RESUME
    end

    subgraph SWEEP["Run sweep (per cell)"]
        direction TB
        SKIPQ{"Cell already<br/>completed?"}
        PREP["prepare_sequence_data<br/>(split, features, windows, loaders)"]
        TRAIN["train_prepared_sequence<br/>(build + train GRU/LSTM)"]
        EVALC["Evaluate official test<br/>RMSE, MAE, PHM08"]
        APPEND["append_row to results CSV"]
        SKIP["Skip cell"]
        SKIPQ -->|no| PREP --> TRAIN --> EVALC --> APPEND
        SKIPQ -->|yes| SKIP
    end

    START --> ARGS
    DELEGATE --> ENUM
    RESUME --> SKIPQ
    APPEND --> DONE(["per-cell result CSVs"])
    SKIP --> DONE
```

## Step reference

| Step | Function | Module |
|---|---|---|
| Enumerate grid | `feature_family_grid.enumerate_cells` | `experiments/feature_family_grid.py` |
| Resume keys | `feature_family_results.completed_keys` | `experiments/feature_family_results.py` |
| Orchestrate | `feature_family_screen.run_screen` | `experiments/feature_family_screen.py` |
| Prepare/train | `sequence_pipeline.prepare_sequence_data` / `train_prepared_sequence` | `training/sequence_pipeline.py` |
| Append row | `feature_family_results.append_row` | `experiments/feature_family_results.py` |
