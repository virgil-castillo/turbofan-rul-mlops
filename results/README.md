# Result Data Convention

`results/baselines/` contains reviewed, version-controlled benchmark snapshots
used by the reports in `docs/`.

Generated experiment output should use the ignored runtime tree under
`outputs/results/`. The sweep and official-evaluation CLIs use that path by
default. Write to `results/baselines/` only when intentionally refreshing the
committed benchmark snapshot, then review the CSV diff before committing.
