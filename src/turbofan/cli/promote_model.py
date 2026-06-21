"""Promote a registered model version to an alias (e.g. ``@production``).

Repoints a registered-model alias at a specific version immediately, with no
approval gate. Rollback is the same operation against an earlier version.
Confirmation is printed to stdout; diagnostics use leveled logging on stderr.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from turbofan import registry
from turbofan.utils import logging as turbofan_logging

logger = turbofan_logging.get_logger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Promote a registered model version to an alias.

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
        registry.promote(args.name, args.version, alias=args.to)
    except Exception as exc:  # noqa: BLE001 - CLI boundary surfaces any failure as exit 1
        logger.error(str(exc))
        logger.debug("Traceback for the error above:", exc_info=True)
        return 1

    uri = registry.resolve_uri(args.name, args.to)
    print(f"{args.name}@{args.to} -> version {args.version}")
    print(f"resolved uri: {uri}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the promote CLI.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", type=str, help="Registered-model name.")
    parser.add_argument("version", type=int, help="Model version to alias.")
    parser.add_argument(
        "--to",
        type=str,
        default="production",
        help="Alias to repoint (defaults to 'production').",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging verbosity (falls back to the LOG_LEVEL env var or INFO).",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
