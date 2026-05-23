"""Tests for turbofan.utils.logging."""
from __future__ import annotations

import logging

from turbofan.utils.logging import get_logger, setup_logging


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
