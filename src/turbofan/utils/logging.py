"""Logging configuration for the turbofan package."""
from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with timestamped console output.

    Args:
        level: Logging level string (e.g., "INFO", "DEBUG", "WARNING").
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a module.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)
