# Architecture Diagrams

These diagrams describe stable package ownership and dependency direction.
They complement the command-oriented diagrams in
[`../workflows/README.md`](../workflows/README.md), which describe runtime
flows rather than package boundaries.

| Area | Diagram |
| --- | --- |
| Benchmark jobs and result persistence | [benchmarks](benchmarks.md) |
| Reusable metrics and official-test helpers | [evaluation](evaluation.md) |
| Ridge and sequence model definitions | [models](models.md) |
| Inference contracts, validation, and compute | [predictions](predictions.md) |
| Operating-mode normalization | [preprocessing](preprocessing.md) |
| MLflow model packaging and registry operations | [registry](registry.md) |
| Window construction and datasets | [sequences](sequences.md) |
| FastAPI transport adapter | [serving](serving.md) |
| Splitting, training, and local artifacts | [training](training.md) |
| Logging utilities | [utilities](utils.md) |

When a package move changes an import boundary, update the relevant diagram,
the [architecture boundary assessment](../architecture_boundary_assessment.md),
and `tests/architecture/test_boundaries.py` together.
