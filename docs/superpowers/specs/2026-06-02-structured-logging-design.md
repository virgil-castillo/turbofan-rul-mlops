# Structured Logging — Design

**Date:** 2026-06-02
**Stage:** MLOps Infrastructure (step 3 of 4: structured logging)
**Status:** Approved design, pending implementation plan

## Goal

Replace the ad-hoc `print()` diagnostics in the project's surviving entrypoints
with leveled standard-library `logging`, using the existing (currently unused)
`turbofan.utils.logging` helper. Adopt a clean **logs → stderr, results →
stdout** split, add a verbosity switch, and capture a per-run `run.log` for
production training runs. On-disk result artifacts (`metrics.json`,
`config.json`, `model_manifest.json`, prediction/sweep CSVs,
`training_history.csv`) are **results, not logs** and are unchanged.

**Prerequisite — exploration scripts are retired first.** The project is moving
from the exploration phase into operationalization; the one-shot sweep/comparison
scripts have deposited their conclusions into `configs/subsets/fd00*.yaml` and the
reports, so they are deleted (see *Prerequisite Cleanup* below) **before** this
step. Logging is therefore applied only to the surviving production entrypoints —
no effort is spent log-polishing code that is about to be removed.

## Decisions (settled during brainstorming)

| Decision | Choice |
|---|---|
| Logging style | Human-readable, leveled stdlib `logging` (not JSON / structured-for-aggregation) |
| stdout vs stderr | **Clean split**: diagnostics → `logging` (stderr); genuine result output → `print` (stdout) |
| Prerequisite | **Retire 7 exploration scripts** (all sweeps + baseline comparisons) before logging; their decisions live in configs/reports |
| Scope | The **4 surviving entrypoints** that `print()`: `train_baseline`, `train_sequence_gru`, `predict`, `download_data` |
| Verbosity | `--log-level` CLI arg (default `INFO`) with `LOG_LEVEL` env-var fallback |
| Per-run `run.log` | **Included**, for the **2 production training CLIs only**; logged as an **MLflow artifact** on the run (not a run-dir file — the registry step retires run dirs) |
| `run.log` capture timing | File handler attached at the **start** of `main()` writing to a temp file (captures data load → train → eval → save), logged as the `run.log` MLflow artifact before the run closes, then detached |
| Progress (`run i/N`) | Stays as `INFO` log lines; **tqdm not adopted** (would corrupt non-TTY/HPC logs) |
| Result artifacts | Unchanged — untouched by this step |

## Prerequisite Cleanup — retire exploration scripts

Done as a **separate change before this step** (ideally a follow-up branch after
the MLflow-tracking branch merges, so one branch does not both add and delete the
step-2 sweep logging). All seven are one-shot exploration scaffolding whose
conclusions are already frozen in `configs/subsets/fd00*.yaml` and the reports; a
single purpose-built sweep will be written later when the model lineup is settled.

**Delete (source + tests + console entrypoint):**
`experiments/baseline_alpha.py`, `experiments/baseline_feature_comparison.py`,
`experiments/gru_temporal_sweep.py`, `experiments/gru_capacity_sweep.py`,
`experiments/feature_gru_sweep.py`, `experiments/sequence_gru_sweep.py`,
`experiments/feature_sweep.py`.

Footprint to clean up alongside:
- **Tests:** the matching `tests/**` files for each script.
- **`pyproject.toml`:** remove the 7 `turbofan-sweep-*` / `turbofan-compare-*`
  console-script entrypoints.
- **SLURM drivers:** `jobs/slurm/run_gru_temporal_sweep_stage1.sh`,
  `jobs/slurm/run_gru_capacity_sweep_stage2.sh`.
- **Docs:** drop README/roadmap mentions (e.g. the "two-stage GRU … SLURM
  drivers" line).

**Keep, do not touch:** the historical reports (`docs/gru_capacity_sweep_report.md`,
`docs/feature_sweep_ridge_vs_gru.md`, `docs/archive/*`) — per the "historical
numbers are not rewritten" convention, they remain as a record even though they
reference now-deleted scripts.

Note: the step-2 MLflow nested-run logging added to the three sweeps is removed
with them. Production-training tracking is unaffected.

## Architecture

### `turbofan/utils/logging.py` (extend the existing helper)

The module already exposes `setup_logging(level)` and `get_logger(name)`. Extend
it with:

- `setup_logging(level: str = "INFO") -> None` — configure the root logger with
  the existing timestamped console formatter writing to **stderr**. Additionally
  quiet noisy third-party loggers (`mlflow`, `matplotlib`) to `WARNING` so they
  do not drown project output. Idempotent (`force=True`, already present).
- `get_logger(name: str) -> logging.Logger` — unchanged; each module does
  `logger = get_logger(__name__)`.
- **New** `run_file_logging(log_path: Path) -> Iterator[None]` — a context
  manager that attaches a `logging.FileHandler` (same formatter) to the root
  logger for its duration and **detaches + closes it on exit**. The detach is
  essential so that in-process callers (tests invoke `main()` repeatedly in one
  process) never bleed one run's diagnostics into another run's file.

`run.log` mirrors the console level (the single `--log-level`). A verbose-file /
concise-console split (file always `DEBUG`) is a deferred refinement, not in v1.

### The split rule (applied at every call site)

- **Diagnostics → `logger.*` (stderr):** data-loading notices, per-epoch and
  `run i/N` progress, raw prediction min/max, "official test skipped".
- **Results → `print(...)` (stdout):** the final `run_dir`, final
  `validation rmse`/`mae`, the official-metrics summary, and the sweep results
  table (`results.to_string()`). These remain pipeable/greppable on stdout.

### Level convention

- **INFO** — lifecycle + progress: "loading FD001", `run 3/12: …`, "saved run
  to …".
- **DEBUG** — verbose internals: raw prediction min/max, intermediate shapes.
- **WARNING** — recoverable surprises: "official test evaluation skipped: files
  not found", fallbacks.

### Verbosity wiring

Each entrypoint's `main()`:
1. parses a new `--log-level` arg (choices `DEBUG/INFO/WARNING/ERROR`, default
   `INFO`), falling back to the `LOG_LEVEL` env var when the flag is absent;
2. calls `setup_logging(level)` once at startup;
3. uses a module-level `logger = get_logger(__name__)`.

### `run.log` capture (production training CLIs — MLflow artifact)

For `cli/train_baseline.py` and `cli/train_sequence_gru.py`, wrap `main()` in
`run_file_logging(<temp path>)` from the start so the temp file captures the whole
run (data load → train → eval → save). Before the production MLflow run closes,
the temp file is logged as the `run.log` artifact:

```python
def main() -> None:
    args = _parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    with run_file_logging(tmp_run_log) as run_log_path:
        ...                                   # data load, train, eval
        with mlflow.start_run():
            ...                               # tracking (+ registry, later)
            mlflow.log_artifact(str(run_log_path), artifact_path="logs")
```

The file handler is detached on exit (essential so in-process test runs do not
bleed one run's log into the next). `run.log` thus travels with the run in
MLflow and is viewable in the UI — no run-dir dependency. Trade-off: a run that
fails *before* the artifact is logged is not uploaded; its narration is still on
stderr (and SLURM/`2>` capture). Resolving the production model URI is unrelated;
this only concerns where the diagnostic log lands.

## Per-path integration

| Entrypoint | `print()`s | `run.log`? | Notes |
|---|---|---|---|
| `cli/train_baseline.py` | 5 | ✅ | Create run dir early; wrap in `run_file_logging`. Keep `run_dir`/final metrics on stdout. |
| `cli/train_sequence_gru.py` | 7 | ✅ | Same; per-epoch/eval narration to stderr + `run.log`. |
| `cli/download_data.py` | 12 | — | Diagnostics → logging; any final "downloaded to …" summary may stay stdout. Kaggle's own progress bar is left alone. |
| `cli/predict.py` | 9 | — | Diagnostics → logging; predicted-value result stays stdout. |

`predict`/`download_data` have no production MLflow run, so they log to stderr
only (no `run.log`). (~33 `print()`s across these 4 files; the other ~12 lived in
the retired exploration scripts.)

## Testing

- **Split assertions:** tests that currently assert *diagnostics* on
  `result.stdout` (e.g. `"official test … skipped"`) move those assertions to
  `result.stderr`; genuine *result* assertions (`"validation rmse"`, the
  `run_dir` line, predicted values) **stay** on `result.stdout`.
- **Verbosity:** a focused test that `--log-level DEBUG` surfaces a debug line
  and `--log-level WARNING` suppresses INFO lines (assert via `caplog` for
  in-process or `result.stderr` for subprocess).
- **`run.log`:** after a production training run, assert `run_dir/run.log` exists
  and contains the training narration (e.g. a known INFO line). Assert the file
  handler is detached afterward (a second in-process `main()` run does not append
  to the first run's file).
- All tests remain data-independent (synthetic fixtures only); no C-MAPSS
  download required.

## Documentation

- Update README's run-output description to note: diagnostics go to stderr via
  leveled logging (`--log-level`), results to stdout, and production runs write a
  `run.log` into their artifact folder.
- Mark step 3 done in `docs/roadmap.md` once implemented.

## Out of scope (deferred)

- **JSON / structured-for-aggregation logs** (Design B) — no log aggregator in
  use; no payoff for a single-user laptop + HPC workflow.
- **`run.log` for `predict` / `download_data`** — these have no production MLflow
  run; out of scope.
- **tqdm progress bars** — an interactive-only convenience. If adopted later, do
  it as an optional `--progress` flag using
  `tqdm(..., disable=not sys.stderr.isatty())` (auto-disable on non-TTY so it
  never corrupts SLURM/CI logs) plus `tqdm.contrib.logging.logging_redirect_tqdm`
  so INFO lines and the bar coexist. tqdm remains an unused transitive dependency
  until then.
- **Verbose-file / concise-console split** (file always `DEBUG`) — possible
  `run.log` refinement; not in v1.
- **Step 4** (model registry) of the MLOps stage.

## Conventions honored

Google-style docstrings; full type annotations; `mypy --strict` clean; ruff
`E,F,W,I,UP,ANN` at line length 88; `--no-ff` merge; reports cite local data
only.
