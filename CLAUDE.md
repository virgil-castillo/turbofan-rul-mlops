# CLAUDE.md

Dev contract for Claude Code when working in this repository.

## Environment

Use the `mlops` Conda environment from `environment.yml` for development and verification.

## Commands

```text
ruff check src/ tests/                 # lint
mypy src/turbofan                       # type-check

pytest                                  # all tests; coverage-gated (fails <93%); temp in workspace-local .pytest_tmp/
pytest tests/test_file.py               # single file
pytest -k "test_name"                   # single test by name
```

## Pre-commit

To run all hooks manually against every file:

```text
pre-commit run --all-files
```

## Commit style
Use conventional commits: `type(scope): message` (e.g. `feat(features): add rolling window`, `fix(train): correct loss accumulation`).

## Coding style
- **Docstrings.** Every public module, class, and function gets a
  Google-style docstring with `Args:`, `Returns:`, and `Raises:` sections
  where applicable.
- **Type annotations.** All function signatures must be fully annotated;
  `mypy --strict` must pass.
- **Ruff** enforces `E`, `F`, `W`, `I`, `UP`, `ANN`, `B`, `BLE`, `ARG`,
  `C4`, and `SIM` rules at line length 88. Fix lint errors before committing.
