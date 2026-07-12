# Workflow: `turbofan-regenerate-baselines`

What happens when regenerating the official-evaluation snapshots: for every
production config, train the model and evaluate it on the official C-MAPSS test
set, writing per-run and summary CSVs. Resumable, and **no MLflow run is created
and the registry is never touched.**

```mermaid
flowchart TD
    START(["$ turbofan-regenerate-baselines --models ridge gru lstm --seeds 42 ..."])

    subgraph SETUP["Setup"]
        direction TB
        ARGS["Parse args + setup logging"]
        MKDIR["Create output dir; resolve per-run + summary paths"]
        ARGS --> MKDIR
    end

    subgraph JOBS["Build jobs"]
        direction TB
        BUILD["official_jobs.build_jobs<br/>(model x subset x seed)"]
        DONE_KEYS["completed_keys (existing per-run CSV)"]
        FILTER["Filter out completed jobs"]
        BUILD --> DONE_KEYS --> FILTER
    end

    subgraph RUN["Run jobs (per remaining job)"]
        direction TB
        TRAINEVAL["official_jobs.run_job<br/>train model + evaluate official test"]
        APPEND["append_record to per-run CSV (incremental)"]
        TRAINEVAL --> APPEND
    end

    subgraph SUMMARY["Summarize"]
        direction TB
        READ["read_records (full per-run CSV)"]
        AGG["build_summary_frame (per model + subset)"]
        WRITE["Write summary CSV"]
        READ --> AGG --> WRITE
    end

    START --> ARGS
    MKDIR --> BUILD
    FILTER --> TRAINEVAL
    APPEND --> READ
    WRITE --> OUT(["per-run + summary CSVs"])
```

## Step reference

| Step | Function | Module |
|---|---|---|
| Build jobs | `official_jobs.build_jobs` | `benchmarks/official_jobs.py` |
| Resume filter | `official_results.completed_keys` | `benchmarks/official_results.py` |
| Train + eval | `official_jobs.run_job` | `benchmarks/official_jobs.py` |
| Append record | `official_results.append_record` | `benchmarks/official_results.py` |
| Summary | `official_results.build_summary_frame` | `benchmarks/official_results.py` |
