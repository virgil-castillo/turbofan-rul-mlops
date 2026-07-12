# Evaluation Architecture

`evaluation/` owns evaluation primitives: regression metrics, prediction
selection, official-label alignment, and the Ridge/sequence official-test
helpers. Reproducible benchmark job execution and reporting live in
`turbofan.benchmarks`.

```mermaid
flowchart TD
    subgraph metrics_py["metrics.py"]
        M1["rmse() / mae()\nregression_metrics() = {rmse, mae}"]
        M2["phm08_score()\nasymmetric early/late penalty"]
        M3["official_test_metrics()\n= regression_metrics + phm08_score"]
    end

    subgraph evaluate_py["evaluate.py"]
        E1["split_features_target() / evaluate_rows()\nfeature/target split, clip >= 0, metrics"]
        E2["select_last_cycle_per_engine()\nofficial-test row selection"]
        E3["align_official_test_labels()\nalign RUL labels to last-cycle rows"]
        E4["predict_with_clipping() / clip_rul_predictions()\npredict_ridge_official() (OfficialRidgePredictions)"]
    end

    subgraph sequence_official_py["sequence_official.py"]
        SO1["align_labels_to_eligible_engines()\nmap official labels to engines that\nproduced a final sequence window"]
    end

    evaluate_py -->|"computes metrics via"| metrics_py
    sequence_official_py -->|"reuses align_official_test_labels"| evaluate_py

    TrainingPipeline["training.sequence_pipeline"]
    TrainingLoop["training.sequence_training"]
    Benchmarks["benchmarks.official_jobs"]
    BaselineCli["cli.train_baseline"]
    SequenceCli["cli.train_sequence"]
    PredictCli["cli.predict"]
    TrainingPipeline --> metrics_py
    TrainingPipeline --> sequence_official_py
    TrainingLoop --> metrics_py
    Benchmarks --> metrics_py
    Benchmarks --> evaluate_py
    BaselineCli --> metrics_py
    BaselineCli --> evaluate_py
    SequenceCli --> metrics_py
    PredictCli --> metrics_py
```
