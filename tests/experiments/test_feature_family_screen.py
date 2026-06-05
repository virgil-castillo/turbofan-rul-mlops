"""Tests for the feature-family screen scaffold.

Covers cell enumeration, cell_key sentinels, CSV append/resume logic,
and malformed-row tolerance.
"""
from __future__ import annotations

from pathlib import Path

from turbofan.experiments.feature_family_screen import (
    CSV_COLUMNS,
    ScreenCell,
    append_row,
    cell_key,
    completed_keys,
    csv_path,
    enumerate_cells,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_KWARGS: dict[str, object] = {
    "architectures": ["gru", "lstm"],
    "subsets": ["FD001", "FD002", "FD003", "FD004"],
    "sequence_windows": [30, 60],
    "rolling_windows": [5, 20],
    "lag_steps": [1, 5],
    "seeds": [42],
}


def _default_cells() -> list[ScreenCell]:
    """Return the full default 240-cell grid."""
    return enumerate_cells(**_DEFAULT_KWARGS)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# enumerate_cells — total count
# ---------------------------------------------------------------------------


def test_enumerate_cells_default_total_is_240() -> None:
    """enumerate_cells with the spec defaults must produce exactly 240 cells."""
    cells = _default_cells()
    assert len(cells) == 240


# ---------------------------------------------------------------------------
# enumerate_cells — 15 cells per (arch, subset, sequence_window)
# ---------------------------------------------------------------------------


def test_enumerate_cells_15_per_arch_subset_seqwin() -> None:
    """Each (arch, subset, sequence_window) slice must contain exactly 15 cells."""
    cells = _default_cells()
    for arch in ["gru", "lstm"]:
        for subset in ["FD001", "FD002", "FD003", "FD004"]:
            for sw in [30, 60]:
                slice_ = [
                    c
                    for c in cells
                    if c.architecture == arch
                    and c.subset == subset
                    and c.sequence_window == sw
                ]
                assert len(slice_) == 15, (
                    f"Expected 15 cells for ({arch},{subset},{sw}), got {len(slice_)}"
                )


# ---------------------------------------------------------------------------
# enumerate_cells — raw cell structure
# ---------------------------------------------------------------------------


def test_enumerate_cells_raw_cell_has_none_sentinels() -> None:
    """The raw cell must have rolling_window=None and lag_step=None."""
    cells = _default_cells()
    raw_cells = [c for c in cells if c.feature_config == "raw"]
    assert len(raw_cells) > 0
    for cell in raw_cells:
        assert cell.rolling_window is None, (
            f"raw cell rolling_window should be None: {cell}"
        )
        assert cell.lag_step is None, f"raw cell lag_step should be None: {cell}"
        assert cell.feature_families == ["raw"]


# ---------------------------------------------------------------------------
# enumerate_cells — rolling cells
# ---------------------------------------------------------------------------


def test_enumerate_cells_rolling_cells_carry_window_and_no_lag() -> None:
    """Rolling cells must have rolling_window set and lag_step=None."""
    cells = _default_cells()
    rolling_cells = [c for c in cells if c.feature_config.startswith("raw+rolling_")]
    assert len(rolling_cells) > 0
    for cell in rolling_cells:
        assert cell.rolling_window is not None, (
            f"rolling cell should have rolling_window set: {cell}"
        )
        assert cell.rolling_window in [5, 20]
        assert cell.lag_step is None, (
            f"rolling cell lag_step should be None: {cell}"
        )


def test_enumerate_cells_rolling_slope_families_correct() -> None:
    """raw+rolling_slope cells must have feature_families=['raw','rolling_slope']."""
    cells = _default_cells()
    slope_cells = [c for c in cells if c.feature_config == "raw+rolling_slope"]
    assert len(slope_cells) > 0
    for cell in slope_cells:
        assert cell.feature_families == ["raw", "rolling_slope"]


# ---------------------------------------------------------------------------
# enumerate_cells — lag cells
# ---------------------------------------------------------------------------


def test_enumerate_cells_lag_cells_carry_lag_and_no_rolling() -> None:
    """Lag cells must have lag_step set and rolling_window=None."""
    cells = _default_cells()
    lag_cells = [c for c in cells if c.feature_config == "raw+lag"]
    assert len(lag_cells) > 0
    for cell in lag_cells:
        assert cell.lag_step is not None, (
            f"lag cell should have lag_step set: {cell}"
        )
        assert cell.lag_step in [1, 5]
        assert cell.rolling_window is None, (
            f"lag cell rolling_window should be None: {cell}"
        )
        assert cell.feature_families == ["raw", "lag"]


# ---------------------------------------------------------------------------
# cell_key — sentinels
# ---------------------------------------------------------------------------


def test_cell_key_raw_cell_sentinels() -> None:
    """raw cell key must use empty strings for rolling_window and lag_step."""
    cells = _default_cells()
    raw_cell = next(
        c for c in cells if c.feature_config == "raw" and c.sequence_window == 30
    )
    key = cell_key(raw_cell)
    assert key == ("raw", "", "", "30", "42")


def test_cell_key_rolling_cell_sentinels() -> None:
    """rolling cell key must use str(rolling_window) and '' for lag_step."""
    cells = _default_cells()
    cell = next(
        c
        for c in cells
        if c.feature_config == "raw+rolling_min"
        and c.rolling_window == 5
        and c.sequence_window == 30
        and c.architecture == "gru"
        and c.subset == "FD001"
    )
    key = cell_key(cell)
    assert key == ("raw+rolling_min", "5", "", "30", "42")


def test_cell_key_lag_cell_sentinels() -> None:
    """lag cell key must use '' for rolling_window and str(lag_step)."""
    cells = _default_cells()
    cell = next(
        c
        for c in cells
        if c.feature_config == "raw+lag"
        and c.lag_step == 5
        and c.sequence_window == 30
        and c.architecture == "gru"
        and c.subset == "FD001"
    )
    key = cell_key(cell)
    assert key == ("raw+lag", "", "5", "30", "42")


# ---------------------------------------------------------------------------
# csv_path
# ---------------------------------------------------------------------------


def test_csv_path_returns_correct_filename(tmp_path: Path) -> None:
    """csv_path must encode arch and subset verbatim in the filename."""
    p = csv_path(tmp_path, "gru", "FD001")
    assert p == tmp_path / "feature_family_screen_gru_FD001.csv"


# ---------------------------------------------------------------------------
# append_row + completed_keys — round-trip
# ---------------------------------------------------------------------------


def _make_row(feature_config: str = "raw", rolling_window: str = "",
              lag_step: str = "", sequence_window: str = "30",
              seed: str = "42") -> dict[str, object]:
    """Build a minimal valid row dict for testing."""
    return {
        "architecture": "gru",
        "subset": "FD001",
        "feature_config": feature_config,
        "rolling_window": rolling_window,
        "lag_step": lag_step,
        "sequence_window": sequence_window,
        "hidden_size": 64,
        "learning_rate": 0.001,
        "seed": seed,
        "n_features": 21,
        "n_train_windows": 100,
        "n_val_windows": 20,
        "best_epoch": 10,
        "val_rmse": 15.3,
        "val_mae": 12.1,
        "training_duration_seconds": 42.0,
    }


def test_append_row_then_completed_keys_round_trip(tmp_path: Path) -> None:
    """Writing a row must make its cell_key appear in completed_keys."""
    p = csv_path(tmp_path, "gru", "FD001")
    row = _make_row()
    append_row(p, row)
    keys = completed_keys(p)
    assert ("raw", "", "", "30", "42") in keys


def test_append_row_writes_header_once(tmp_path: Path) -> None:
    """Header must appear exactly once even after two appends."""
    p = csv_path(tmp_path, "gru", "FD001")
    append_row(p, _make_row(feature_config="raw"))
    append_row(p, _make_row(feature_config="raw+lag", lag_step="1"))
    lines = p.read_text().splitlines()
    header_lines = [ln for ln in lines if ln.startswith("architecture")]
    assert len(header_lines) == 1


def test_completed_keys_returns_both_rows(tmp_path: Path) -> None:
    """completed_keys must return one key per appended row."""
    p = csv_path(tmp_path, "gru", "FD001")
    append_row(p, _make_row(feature_config="raw"))
    append_row(p, _make_row(feature_config="raw+lag", lag_step="1"))
    keys = completed_keys(p)
    assert ("raw", "", "", "30", "42") in keys
    assert ("raw+lag", "", "1", "30", "42") in keys


# ---------------------------------------------------------------------------
# completed_keys — missing file
# ---------------------------------------------------------------------------


def test_completed_keys_empty_for_missing_file(tmp_path: Path) -> None:
    """completed_keys must return an empty set when the CSV does not exist."""
    p = tmp_path / "does_not_exist.csv"
    assert completed_keys(p) == set()


# ---------------------------------------------------------------------------
# completed_keys — malformed trailing row
# ---------------------------------------------------------------------------


def test_completed_keys_skips_malformed_trailing_row(tmp_path: Path) -> None:
    """completed_keys must skip rows with wrong field count and not raise."""
    p = csv_path(tmp_path, "gru", "FD001")
    append_row(p, _make_row())
    # Append a junk line with too few fields
    with p.open("a", newline="") as fh:
        fh.write("junk,broken\n")
    keys = completed_keys(p)
    # Good row still parses
    assert ("raw", "", "", "30", "42") in keys
    # No exception raised; junk row just dropped


# ---------------------------------------------------------------------------
# CSV_COLUMNS
# ---------------------------------------------------------------------------


def test_csv_columns_order() -> None:
    """CSV_COLUMNS must match the spec-mandated 16-column order."""
    expected = [
        "architecture",
        "subset",
        "feature_config",
        "rolling_window",
        "lag_step",
        "sequence_window",
        "hidden_size",
        "learning_rate",
        "seed",
        "n_features",
        "n_train_windows",
        "n_val_windows",
        "best_epoch",
        "val_rmse",
        "val_mae",
        "training_duration_seconds",
    ]
    assert CSV_COLUMNS == expected
