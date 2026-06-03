"""Logging configuration for the turbofan package."""
from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with timestamped output on stderr.

    Diagnostics are written to ``stderr`` so that genuine results printed on
    ``stdout`` stay cleanly separable. Noisy third-party loggers (``mlflow``,
    ``matplotlib``) are quieted to ``WARNING``. Idempotent: repeated calls
    reconfigure the root logger via ``force=True``.

    Args:
        level: Logging level string (e.g., "INFO", "DEBUG", "WARNING").

    Returns:
        None.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True,
    )
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a module.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)


@contextmanager
def run_file_logging(log_path: Path) -> Iterator[None]:
    """Mirror console log records into a per-run file for the context's duration.

    Attaches a :class:`logging.FileHandler` (sharing the console formatter and
    the root logger's level) to the root logger on enter, and removes and closes
    it on exit. Detaching on exit is essential: callers that invoke a training
    ``main()`` repeatedly in one process must not bleed one run's diagnostics
    into the next run's file.

    Args:
        log_path: Destination path for the per-run log file.

    Yields:
        None. The file handler is active for the duration of the ``with`` block.
    """
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
