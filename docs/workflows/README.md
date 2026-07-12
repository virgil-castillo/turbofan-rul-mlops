# CLI Workflows

Start with the [**end-to-end overview**](overview.md) for the big picture, then
drill into one doc per `turbofan-*` command below — each tracing what happens
from invocation to result through named phases (Setup, Preprocessing, Training,
Evaluation, …) with a step-reference table. For the static folder-dependency map
underneath these flows, see the
[architecture diagrams](../architecture/README.md).

| Command | Workflow | Summary |
|---|---|---|
| `turbofan-download-data` | [download-data](download-data.md) | Download or verify the C-MAPSS dataset |
| `turbofan-train-baseline` | [train-baseline](train-baseline.md) | Train + register the Ridge tabular baseline |
| `turbofan-train-sequence` | [train-sequence](train-sequence.md) | Train + register a GRU/LSTM sequence model |
| `turbofan-predict` | [predict](predict.md) | Batch inference from a CSV/JSON file |
| `turbofan-serve-api` | [serve-api](serve-api.md) | FastAPI service over the inference core |
| `turbofan-promote` | [promote-model](promote-model.md) | Repoint a model alias to a version |
| `turbofan-models` | [list-models](list-models.md) | List registered models + production aliases |
| `turbofan-feature-screen` | [feature-screen](feature-screen.md) | Feature-family sweep over the sequence pipeline |
| `turbofan-regenerate-baselines` | [regenerate-baselines](regenerate-baselines.md) | Regenerate official-eval snapshot CSVs |
