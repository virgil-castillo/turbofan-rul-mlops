# `turbofan.training` Package Modules

`training/` holds the application-level train/evaluate use cases that compose
the data, feature, sequence, model, and evaluation layers. It is consumed by the
train CLIs, the benchmark jobs, and the feature-screen experiment.

```mermaid
flowchart TD
    subgraph split_py["split.py"]
        SP1["load_and_split()\nload raw train, label via data.labels.add_rul_column,\nsplit engines into train/validation"]
        SP2["split_by_engine()\nshuffle engine_ids, partition rows by whole engine"]
    end

    subgraph sequence_training_py["sequence_training.py"]
        ST1["resolve_device() / seed_everything()"]
        ST2["train_sequence_model()\nepoch loop, MSE on normalized targets, Adam,\nvalidation RMSE, early stopping, restore best"]
        ST3["predict_windows()\nbatched inference over a loader, rescale by max_rul"]
    end

    subgraph sequence_pipeline_py["sequence_pipeline.py"]
        SQ1["prepare_sequence_data()\nsplit + fit feature pipeline + build windows/loaders"]
        SQ2["train_prepared_sequence()\nseed, build model, train"]
        SQ3["evaluate_window_metrics() / predict_sequence_official()"]
    end

    subgraph artifacts_py["artifacts.py"]
        A1["create_run_dir()\ntimestamped, collision-safe run directory"]
        A2["save_model() / save_json() / save_predictions()"]
    end

    split_py -->|"labels via"| LabelsNote["turbofan.data.labels"]
    sequence_pipeline_py --> split_py
    sequence_pipeline_py --> sequence_training_py
    sequence_pipeline_py -->|"build_sequence_model"| ModelsNote["turbofan.models.sequence_models"]
    sequence_pipeline_py -->|"metrics + sequence_official"| EvalNote["turbofan.evaluation"]
    sequence_training_py -->|"regression_metrics"| EvalNote
```
