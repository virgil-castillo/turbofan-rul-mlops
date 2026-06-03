"""Tests for turbofan.utils.logging."""
from __future__ import annotations

import logging
from pathlib import Path

from turbofan.utils.logging import (
    get_logger,
    run_file_logging,
    setup_logging,
)


def test_get_logger_returns_logger() -> None:
    """get_logger returns a logging.Logger with the given name."""
    logger = get_logger("turbofan.test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "turbofan.test_module"


def test_get_logger_different_names_are_distinct() -> None:
    """Two loggers with different names are different objects."""
    logger_a = get_logger("turbofan.module_a")
    logger_b = get_logger("turbofan.module_b")
    assert logger_a is not logger_b


def test_setup_logging_does_not_raise() -> None:
    """setup_logging completes without raising for valid level strings."""
    setup_logging("DEBUG")
    setup_logging("INFO")
    setup_logging("WARNING")


def test_setup_logging_quiets_third_party_loggers() -> None:
    """setup_logging lowers noisy third-party loggers to WARNING."""
    setup_logging("DEBUG")
    assert logging.getLogger("mlflow").level == logging.WARNING
    assert logging.getLogger("matplotlib").level == logging.WARNING


def test_run_file_logging_captures_records(tmp_path: Path) -> None:
    """run_file_logging writes records emitted inside the context to the file."""
    setup_logging("INFO")
    log_path = tmp_path / "run.log"
    logger = get_logger("turbofan.test_file_logging")

    with run_file_logging(log_path):
        logger.info("captured narration line")

    contents = log_path.read_text()
    assert "captured narration line" in contents


def test_run_file_logging_detaches_handler_on_exit(tmp_path: Path) -> None:
    """Each context only captures records emitted while it is open."""
    setup_logging("INFO")
    logger = get_logger("turbofan.test_file_detach")
    path_a = tmp_path / "a.log"
    path_b = tmp_path / "b.log"

    with run_file_logging(path_a):
        logger.info("alpha")
    with run_file_logging(path_b):
        logger.info("beta")

    contents_a = path_a.read_text()
    contents_b = path_b.read_text()
    assert "alpha" in contents_a
    assert "beta" not in contents_a
    assert "beta" in contents_b
    assert "alpha" not in contents_b


def test_run_file_logging_does_not_append_after_close(tmp_path: Path) -> None:
    """Records logged outside any context do not append to a closed file."""
    setup_logging("INFO")
    logger = get_logger("turbofan.test_file_closed")
    log_path = tmp_path / "run.log"

    with run_file_logging(log_path):
        logger.info("inside")
    logger.info("outside")

    contents = log_path.read_text()
    assert "inside" in contents
    assert "outside" not in contents
