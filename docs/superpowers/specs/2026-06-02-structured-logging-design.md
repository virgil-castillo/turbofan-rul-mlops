# Structured Logging — Design

**Date:** 2026-06-02
**Stage:** MLOps Infrastructure (step 3 of 4: structured logging)
**Status:** Approved design, pending implementation plan

## Goal

Replace the project's ~45 ad-hoc `print()` diagnostics with leveled standard-
library `logging`, using the existing (currently unused) `turbofan.utils.logging`
helper. Adopt a clean **logs → stderr, results → stdout** split, add a
verbosity switch, and capture a per-run `run.log` for production training runs.
On-disk result artifacts (`metrics.json`, `config.json`, `model_manifest.json`,
prediction/sweep CSVs, `training_history.csv`) are **results, not logs** and are
unchanged.

## Decisions (settled during brainstorming)

| Decision | Choice |
|---|---|
| Logging style | Human-readable, leveled stdlib `logging` (not JSON / structured-for-aggregation) |
| stdout vs stderr | **Clean split**: diagnostics → `logging` (stderr); genuine result output → `print` (stdout) |
| Scope | **All 11 entrypoints** that currently `print()`, not just the 5 touched by MLflow |
| Verbosity | `--log-level` CLI arg (default `INFO`) with `LOG_LEVEL` env-var fallback |
| Per-run `run.log` | **Included**, for the **2 production training CLIs only** (the only entrypoints with a run dir) |
| `run.log` capture timing | **Option A** — create the run dir at the start of the run and attach the file handler immediately, so `run.log` captures the whole run (data load → train → eval → save) |
| Progress (`run i/N`) | Stays as `INFO` log lines; **tqdm not adopted** (would corrupt non-TTY/HPC logs) |
| Result artifacts | Unchanged — untouched by this step |

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

### `run.log` capture (production training CLIs — Option A)

For `cli/train_baseline.py` and `cli/train_sequence_gru.py`, restructure `main()`
so the run directory is created **at the start of the run** (after config load
and cheap validation, before data loading/training), then wrap the body in
`run_file_logging(run_dir / "run.log")`:

```python
def main() -> None:
    args = _parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    # (cheap validation, e.g. GRU architecture check)
    run_dir = create_run_dir(...)            # moved to the start of the run
    with run_file_logging(run_dir / "run.log"):
        ...                                   # data load, train, eval, MLflow, save
```

Consequence (accepted): a run that fails early now leaves a run folder
containing only `run.log` (and no model). This is treated as a **feature** — a
failed run leaves a diagnosable log behind — at the cost of the previous "every
run folder has a model" invariant. The MLflow `start_run()` context is unchanged
and still wraps the eval/save/log section; `run_dir` (already created) is logged
as the `run_dir` tag as before.

## Per-path integration

| Entrypoint | `print()`s | `run.log`? | Notes |
|---|---|---|---|
| `cli/train_baseline.py` | 5 | ✅ | Create run dir early; wrap in `run_file_logging`. Keep `run_dir`/final metrics on stdout. |
| `cli/train_sequence_gru.py` | 7 | ✅ | Same; per-epoch/eval narration to stderr + `run.log`. |
| `cli/download_data.py` | 12 | — | Diagnostics → logging; any final "downloaded to …" summary may stay stdout. |
| `cli/predict.py` | 9 | — | Diagnostics → logging; predicted-value result stays stdout. |
| `experiments/feature_sweep.py` | 2 | — | `run i/N` → INFO; results table stays stdout. |
| `experiments/feature_gru_sweep.py` | 2 | — | As above. |
| `experiments/sequence_gru_sweep.py` | 2 | — | As above. |
| `experiments/baseline_alpha.py` | 1 | — | Diagnostics → logging; result table stdout. |
| `experiments/baseline_feature_comparison.py` | 1 | — | As above. |
| `experiments/gru_temporal_sweep.py` | 2 | — | As above. |
| `experiments/gru_capacity_sweep.py` | 2 | — | As above. |

Sweeps and `predict`/`download_data` have no per-run artifact folder, so they log
to stderr only (no `run.log`).

## Testing

- **Split assertions:** tests that currently assert diagnostics on
  `result.stdout` (e.g. `"run 1/1"`, `"official test … skipped"`,
  `validation/sweep progress`) move those assertions to `result.stderr`; genuine
  result assertions (`"validation rmse"`, the results table) **stay** on
  `result.stdout`.
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
- **`run.log` for sweeps / `predict` / `download_data`** — these have no per-run
  folder; out of scope.
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
