# Benchmarks Architecture

`benchmarks/` regenerates the official C-MAPSS benchmark CSVs. `official_jobs.py`
enumerates and runs train+evaluate jobs (delegating the actual train/evaluate
work to the shared pipeline modules documented in
[the shared train/evaluate pipeline diagram](../workflows/shared-pipeline-usage.md));
`official_results.py` persists `RunRecord`s to CSV and aggregates them into the
summary frame. The only external entrypoint is the
`turbofan-regenerate-baselines` CLI.

```mermaid
flowchart TD
    subgraph OJ["official_jobs.py"]
        BJ["build_jobs"]
        RJ["run_job"]
        JK["job_key"]
        ER["_evaluate_ridge"]
        ES["_evaluate_sequence"]
        BR["_build_record"]
        JW["join_windows"]
        JL["join_lag_steps"]
        CPF["config_path_for"]
        BJ -->|per model/subset| CPF
        RJ -->|model == ridge| ER
        RJ -->|model != ridge| ES
        ER --> BR
        ES --> BR
        BR --> JW
        BR --> JL
    end

    subgraph WF["training (split / sequence_pipeline) + models.baseline + evaluation.evaluate"]
        LAS["training.split.load_and_split /<br/>models.baseline.build_ridge_estimator /<br/>evaluation.evaluate.predict_with_clipping /<br/>evaluation.evaluate.predict_ridge_official"]
        SEQ["training.sequence_pipeline:<br/>prepare_sequence_data /<br/>train_prepared_sequence /<br/>evaluate_window_metrics /<br/>predict_sequence_official"]
    end
    ER -->|trains + evaluates Ridge| LAS
    ES -->|trains + evaluates sequence model| SEQ

    subgraph OR["official_results.py"]
        CK["completed_keys"]
        AR["append_record"]
        RR["read_records"]
        BSF["build_summary_frame"]
        R2R["record_to_row"]
        RFR["record_from_row"]
        SSD["sample_sd"]
        RK["record_key"]
        RSK["record_sort_key"]
        GSK["group_sort_key"]
        AR -->|writes row| R2R
        RR -->|parses rows| RFR
        CK -->|reads + keys| RR
        CK -.-> RK
        RR -.->|sort| RSK
        BSF -->|per group mean/sd| SSD
        BSF -.->|sort| GSK
    end
    OR -.->|imports RunRecord, MODEL_ORDER| OJ

    CLI["cli/regenerate_official_baselines.py<br/>main"]
    CLI -->|build_jobs| BJ
    CLI -->|job_key, completed_keys: resume filter| JK
    CLI -.-> CK
    CLI -->|run_job per remaining job| RJ
    CLI -->|append_record per result| AR
    CLI -->|read_records + build_summary_frame at the end| RR
    CLI -.-> BSF
    CLI -->|writes| PERRUN[("outputs/results/<br/>latest_official_eval_per_run.csv")]
    CLI -->|writes| SUMMARY[("outputs/results/<br/>latest_official_eval_summary.csv")]
    AR --> PERRUN
    BSF --> SUMMARY
```

## Notes

- **Resumability**: `main` reads `completed_keys` from the existing per-run CSV
  and filters jobs whose `job_key` (model, subset, seed) is already present, so
  an interrupted sweep can restart without recomputing finished jobs.
- **Ridge vs. sequence**: `run_job` dispatches purely on `job.model == "ridge"`;
  both branches funnel into `_build_record`, which fills in the
  sequence-specific columns (`sequence_window`, `hidden_size`,
  `learning_rate`) as empty strings for Ridge.
- `official_results.py` depends on `official_jobs.py` only for the `RunRecord`
  dataclass and `MODEL_ORDER` (display/sort order) — it has no knowledge of
  how a record was produced.
