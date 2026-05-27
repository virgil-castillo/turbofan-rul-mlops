"""Tests for turbofan.preprocessing.normalization."""
from __future__ import annotations

import pytest

from turbofan.preprocessing.normalization import (
    CMAPSS_SUBSET_MODE_COUNTS,
    mode_count_for_subset,
)


def test_all_subset_mode_counts_are_positive_integers() -> None:
    """Every entry in CMAPSS_SUBSET_MODE_COUNTS is a positive integer."""
    for subset, count in CMAPSS_SUBSET_MODE_COUNTS.items():
        assert isinstance(count, int), f"{subset}: expected int, got {type(count)}"
        assert count > 0, f"{subset}: mode count must be positive"


def test_single_condition_subsets_have_mode_count_one() -> None:
    """FD001 and FD003 are treated as single-condition subsets."""
    assert mode_count_for_subset("FD001") == 1
    assert mode_count_for_subset("FD003") == 1


def test_multi_condition_subsets_have_mode_count_six() -> None:
    """FD002 and FD004 are treated as six-condition subsets."""
    assert mode_count_for_subset("FD002") == 6
    assert mode_count_for_subset("FD004") == 6


def test_unsupported_subset_raises_value_error() -> None:
    """mode_count_for_subset raises ValueError for unknown subset names."""
    with pytest.raises(ValueError, match="FD999"):
        mode_count_for_subset("FD999")
