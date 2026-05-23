"""Tests for turbofan.data.labels."""
from __future__ import annotations

import pandas as pd

from turbofan.data.labels import compute_rul_labels


def _make_df(engine_cycles: dict[int, int]) -> pd.DataFrame:
    """Build a minimal DataFrame from {engine_id: n_cycles}."""
    rows = []
    for engine_id, n_cycles in engine_cycles.items():
        for cycle in range(1, n_cycles + 1):
            rows.append({"engine_id": engine_id, "cycle": cycle})
    return pd.DataFrame(rows)


def test_rul_is_zero_at_last_cycle() -> None:
    """RUL is 0 at the final cycle of each engine."""
    df = _make_df({1: 10})
    rul = compute_rul_labels(df, max_rul=125)
    last_cycle_rul = rul[df["cycle"] == 10].iloc[0]
    assert last_cycle_rul == 0


def test_rul_capped_at_max_rul() -> None:
    """RUL never exceeds max_rul for any cycle."""
    df = _make_df({1: 200})
    rul = compute_rul_labels(df, max_rul=125)
    assert (rul <= 125).all()


def test_rul_equals_max_rul_for_early_cycles() -> None:
    """Early cycles where cycles_remaining > max_rul receive max_rul."""
    df = _make_df({1: 200})
    rul = compute_rul_labels(df, max_rul=125)
    first_cycle_rul = rul[df["cycle"] == 1].iloc[0]
    assert first_cycle_rul == 125


def test_rul_custom_max_rul() -> None:
    """Custom max_rul value is respected."""
    df = _make_df({1: 50})
    rul = compute_rul_labels(df, max_rul=30)
    first_cycle_rul = rul[df["cycle"] == 1].iloc[0]
    assert first_cycle_rul == 30
    last_cycle_rul = rul[df["cycle"] == 50].iloc[0]
    assert last_cycle_rul == 0


def test_rul_multiple_engines_independent() -> None:
    """Engines with different lifespans each reach RUL=0 at their own last cycle."""
    df = _make_df({1: 10, 2: 20})
    rul = compute_rul_labels(df, max_rul=125)
    engine1_last = rul[(df["engine_id"] == 1) & (df["cycle"] == 10)].iloc[0]
    engine2_last = rul[(df["engine_id"] == 2) & (df["cycle"] == 20)].iloc[0]
    assert engine1_last == 0
    assert engine2_last == 0


def test_rul_index_aligned_to_input() -> None:
    """Returned Series has the same index as the input DataFrame."""
    df = _make_df({1: 5})
    rul = compute_rul_labels(df, max_rul=125)
    assert list(rul.index) == list(df.index)


def test_rul_series_name() -> None:
    """Returned Series is named 'rul'."""
    df = _make_df({1: 5})
    rul = compute_rul_labels(df, max_rul=125)
    assert rul.name == "rul"
