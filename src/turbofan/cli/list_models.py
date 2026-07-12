"""List registered models with versions, production alias, and provenance.

Prints a plain-text, greppable table to stdout: one row per registered model
with its name, all versions, the ``@production`` version, the production run's
``val_rmse`` metric, and a run link (the run id, falling back to the resolved
``models:/<name>@production`` URI). Diagnostics use leveled logging on stderr.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from turbofan import registry
from turbofan.registry import RegisteredModelInfo
from turbofan.utils import logging as turbofan_logging

logger = turbofan_logging.get_logger(__name__)

_HEADERS = ("name", "versions", "production", "val_rmse", "run_link")
_PLACEHOLDER = "-"


def main(argv: Sequence[str] | None = None) -> int:
    """List registered models and their production aliases.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    turbofan_logging.setup_logging(args.log_level)
    registry.tracking.configure_mlflow()
    try:
        models = registry.list_registered()
    except Exception as exc:  # noqa: BLE001 - CLI boundary surfaces any failure as exit 1
        logger.error(str(exc))
        logger.debug("Traceback for the error above:", exc_info=True)
        return 1

    if not models:
        print("No registered models.")
        return 0

    for line in _format_table(models):
        print(line)
    return 0


def _format_table(models: Sequence[RegisteredModelInfo]) -> list[str]:
    """Render registered models as aligned, ``|``-separated table rows.

    Args:
        models: Registered-model listing records.

    Returns:
        Header row followed by one formatted row per model.
    """
    rows = [_HEADERS, *(_row_cells(model) for model in models)]
    widths = [max(len(row[col]) for row in rows) for col in range(len(_HEADERS))]
    return [
        " | ".join(cell.ljust(widths[col]) for col, cell in enumerate(row))
        for row in rows
    ]


def _row_cells(model: RegisteredModelInfo) -> tuple[str, str, str, str, str]:
    """Build the formatted cells for one registered model.

    Args:
        model: Registered-model listing record.

    Returns:
        Cells for the name, versions, production, val_rmse, and run_link columns.
    """
    versions = ",".join(str(version) for version in model.versions)
    production = (
        str(model.production_version)
        if model.production_version is not None
        else _PLACEHOLDER
    )
    val_rmse = (
        f"{model.val_rmse:.4f}" if model.val_rmse is not None else _PLACEHOLDER
    )
    run_link = model.run_id or registry.resolve_uri(model.name)
    return (model.name, versions, production, val_rmse, run_link)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the listing CLI.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging verbosity (falls back to the LOG_LEVEL env var or INFO).",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
