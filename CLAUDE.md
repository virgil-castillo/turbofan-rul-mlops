# CLAUDE.md

Dev contract for Claude Code when working in this repository.

## Environment

Conda lives at `$env:USERPROFILE\miniconda3`. Claude Code does not inherit a login shell, so the `mlops` env must be activated explicitly before running Python commands.

**PowerShell tool:**
```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate mlops
```

**Bash tool:**
```bash
source ~/miniconda3/Scripts/activate
conda activate mlops
```

## Commands

```bash
ruff check src/ tests/                 # lint
mypy src/turbofan                       # type-check

pytest                                  # all tests; reports line coverage for `turbofan` and fails if total drops below 93%
pytest tests/test_file.py               # single file
pytest -k "test_name"                   # single test by name
```

## Pre-commit

Run once after cloning to enable the git hook:

```bash
pre-commit install
```

To run all hooks manually against every file:

```bash
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
