"""Tests for the feature-family screen scaffold.

Covers cell enumeration, cell_key sentinels, CSV append/resume logic,
malformed-row tolerance, and run_cell/run_screen execution harness.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

import turbofan.experiments.feature_family_screen as _screen_module
from turbofan.experiments.feature_family_screen import (
    CSV_COLUMNS,
    ScreenCell,
    append_row,
    cell_key,
    completed_keys,
    csv_path,
    enumerate_cells,
)
from turbofan.models.sequence_training import TrainingResult

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


# ---------------------------------------------------------------------------
# Helpers for run_cell / run_screen tests
# ---------------------------------------------------------------------------

# 3 fake feature cols returned by the fake pipeline
_FAKE_FEATURE_COLS: list[str] = ["s_2", "s_3", "s_4"]
_N_FEATURES = len(_FAKE_FEATURE_COLS)  # 3

# Minimal fake DataFrames (content doesn't matter — nothing real runs)
_FAKE_DF = pd.DataFrame(
    {"engine_id": [1, 1], "cycle": [1, 2], "rul": [10, 9], "s_2": [1.0, 2.0]}
)


def _make_fake_windows(n: int) -> object:
    """Return a fake WindowedSequences-like object with shape (n,)."""
    ws = MagicMock()
    ws.X = np.zeros((n, 5, _N_FEATURES))
    ws.y = np.ones(n, dtype=np.float64) * 20.0
    return ws


def _fake_subset_cfg(n_modes: int = 6) -> object:
    """Fake ProjectConfig with inherited sensor/mode fields."""
    cfg = SimpleNamespace()
    cfg.data = SimpleNamespace(
        raw_dir=Path("data/raw"),
        processed_dir=Path("data/processed"),
        interim_dir=Path("data/interim"),
    )
    cfg.features = SimpleNamespace(
        sensor_cols_to_drop=["s_1", "s_18"],
        n_modes=n_modes,
    )
    return cfg


def _make_rolling_cell(seed: int = 44) -> ScreenCell:
    """Return a rolling-window cell with seed != 42."""
    return ScreenCell(
        architecture="gru",
        subset="FD002",
        feature_config="raw+rolling_mean",
        feature_families=["raw", "rolling_mean"],
        rolling_window=5,
        lag_step=None,
        sequence_window=30,
        seed=seed,
    )


def _make_raw_cell(seed: int = 44) -> ScreenCell:
    """Return a raw cell with seed != 42."""
    return ScreenCell(
        architecture="gru",
        subset="FD001",
        feature_config="raw",
        feature_families=["raw"],
        rolling_window=None,
        lag_step=None,
        sequence_window=30,
        seed=seed,
    )


def _make_lag_cell(seed: int = 44) -> ScreenCell:
    """Return a lag cell with seed != 42."""
    return ScreenCell(
        architecture="gru",
        subset="FD001",
        feature_config="raw+lag",
        feature_families=["raw", "lag"],
        rolling_window=None,
        lag_step=5,
        sequence_window=30,
        seed=seed,
    )


def _install_run_cell_patches(
    monkeypatch: pytest.MonkeyPatch,
    cell: ScreenCell,
    *,
    n_modes: int = 6,
) -> dict[str, list[object]]:
    """Install all module-level monkeypatches for run_cell and return call logs."""
    calls: dict[str, list[object]] = {
        "load_config": [],
        "load_raw_train": [],
        "add_rul_column": [],
        "split_by_engine": [],
        "build_feature_pipeline": [],
        "build_sliding_windows": [],
        "build_sequence_loader": [],
        "seed_everything": [],
        "build_sequence_model": [],
        "train_sequence_model": [],
        "resolve_device": [],
    }

    fake_subset_cfg = _fake_subset_cfg(n_modes=n_modes)
    fake_train_windows = _make_fake_windows(80)
    fake_val_windows = _make_fake_windows(20)

    class _FakePipeline:
        named_steps: dict[str, object]

        def __init__(self) -> None:
            fe = SimpleNamespace(feature_cols_=_FAKE_FEATURE_COLS)
            self.named_steps = {"feature_engineer": fe}

        def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    fake_model = MagicMock(spec=torch.nn.Module)
    fake_history = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "validation_windows_rmse": [15.0, 13.0, 12.5],
            "validation_windows_mae": [11.0, 10.0, 9.0],
        }
    )
    fake_result = TrainingResult(
        model=fake_model,
        history=fake_history,
        best_epoch=3,
        best_metric=12.5,
    )

    def fake_load_config(path: Path) -> object:
        calls["load_config"].append(path)
        return fake_subset_cfg

    def fake_load_raw_train(cfg: object) -> pd.DataFrame:
        calls["load_raw_train"].append(cfg)
        return _FAKE_DF.copy()

    def fake_add_rul_column(df: pd.DataFrame, *, max_rul: int) -> pd.DataFrame:
        calls["add_rul_column"].append(max_rul)
        return df

    def fake_split_by_engine(
        df: pd.DataFrame, *, test_size: float, random_seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls["split_by_engine"].append(
            {"test_size": test_size, "random_seed": random_seed}
        )
        return df, df

    def fake_build_feature_pipeline(**kwargs: object) -> _FakePipeline:
        calls["build_feature_pipeline"].append(kwargs)
        return _FakePipeline()

    def fake_build_sliding_windows(
        df: pd.DataFrame, *, feature_cols: list[str], window_size: int
    ) -> object:
        calls["build_sliding_windows"].append(
            {"feature_cols": feature_cols, "window_size": window_size}
        )
        # Return train or val windows based on call count
        if len(calls["build_sliding_windows"]) == 1:
            return fake_train_windows
        return fake_val_windows

    def fake_build_sequence_loader(windows: object, **kwargs: object) -> object:
        calls["build_sequence_loader"].append(kwargs)
        return MagicMock()

    def fake_seed_everything(seed: int) -> None:
        calls["seed_everything"].append(seed)

    def fake_build_sequence_model(arch: str, **kwargs: object) -> torch.nn.Module:
        calls["build_sequence_model"].append({"arch": arch, **kwargs})
        return fake_model

    def fake_train_sequence_model(**kwargs: object) -> TrainingResult:
        calls["train_sequence_model"].append(kwargs)
        return fake_result

    def fake_resolve_device(requested: object) -> torch.device:
        calls["resolve_device"].append(requested)
        return torch.device("cpu")

    monkeypatch.setattr(_screen_module, "load_config", fake_load_config)
    monkeypatch.setattr(_screen_module, "load_raw_train", fake_load_raw_train)
    monkeypatch.setattr(_screen_module, "add_rul_column", fake_add_rul_column)
    monkeypatch.setattr(_screen_module, "split_by_engine", fake_split_by_engine)
    monkeypatch.setattr(
        _screen_module, "build_feature_pipeline", fake_build_feature_pipeline
    )
    monkeypatch.setattr(
        _screen_module, "build_sliding_windows", fake_build_sliding_windows
    )
    monkeypatch.setattr(
        _screen_module, "build_sequence_loader", fake_build_sequence_loader
    )
    monkeypatch.setattr(_screen_module, "seed_everything", fake_seed_everything)
    monkeypatch.setattr(
        _screen_module, "build_sequence_model", fake_build_sequence_model
    )
    monkeypatch.setattr(
        _screen_module, "train_sequence_model", fake_train_sequence_model
    )
    monkeypatch.setattr(_screen_module, "resolve_device", fake_resolve_device)

    return calls


# ---------------------------------------------------------------------------
# run_cell — seed decoupling: split always uses 42
# ---------------------------------------------------------------------------


def test_run_cell_split_always_uses_seed_42(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """split_by_engine must receive random_seed=42 and test_size=0.2, not cell.seed."""
    from turbofan.experiments.feature_family_screen import run_cell

    cell = _make_rolling_cell(seed=44)
    calls = _install_run_cell_patches(monkeypatch, cell)
    run_cell(cell)

    assert len(calls["split_by_engine"]) == 1
    kw = calls["split_by_engine"][0]
    assert kw["random_seed"] == 42, "split seed must be fixed at 42, not cell.seed"
    assert kw["test_size"] == 0.2


# ---------------------------------------------------------------------------
# run_cell — seed decoupling: pipeline always uses random_state=42
# ---------------------------------------------------------------------------


def test_run_cell_pipeline_always_uses_random_state_42(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_feature_pipeline must receive random_state=42, not cell.seed."""
    from turbofan.experiments.feature_family_screen import run_cell

    cell = _make_rolling_cell(seed=44)
    calls = _install_run_cell_patches(monkeypatch, cell)
    run_cell(cell)

    assert len(calls["build_feature_pipeline"]) == 1
    kw = calls["build_feature_pipeline"][0]
    assert kw["random_state"] == 42


# ---------------------------------------------------------------------------
# run_cell — seed decoupling: model/training use cell.seed
# ---------------------------------------------------------------------------


def test_run_cell_model_uses_cell_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """seed_everything and train_sequence_model must use cell.seed (44), not 42."""
    from turbofan.experiments.feature_family_screen import run_cell

    cell = _make_rolling_cell(seed=44)
    calls = _install_run_cell_patches(monkeypatch, cell)
    run_cell(cell)

    assert calls["seed_everything"] == [44]
    assert len(calls["train_sequence_model"]) == 1
    assert calls["train_sequence_model"][0]["random_seed"] == 44


# ---------------------------------------------------------------------------
# run_cell — feature families / windows / lag_steps per cell type
# ---------------------------------------------------------------------------


def test_run_cell_rolling_cell_pipeline_gets_correct_families_and_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rolling cell must pass feature_families and windows=[5], lag_steps=None."""
    from turbofan.experiments.feature_family_screen import run_cell

    cell = _make_rolling_cell(seed=44)
    calls = _install_run_cell_patches(monkeypatch, cell)
    run_cell(cell)

    kw = calls["build_feature_pipeline"][0]
    assert kw["feature_families"] == ["raw", "rolling_mean"]
    assert kw["windows"] == [5]
    assert kw["lag_steps"] is None


def test_run_cell_raw_cell_pipeline_gets_none_windows_and_lag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw cell must pass windows=None and lag_steps=None to build_feature_pipeline."""
    from turbofan.experiments.feature_family_screen import run_cell

    cell = _make_raw_cell(seed=44)
    calls = _install_run_cell_patches(monkeypatch, cell, n_modes=1)
    run_cell(cell)

    kw = calls["build_feature_pipeline"][0]
    assert kw["feature_families"] == ["raw"]
    assert kw["windows"] is None
    assert kw["lag_steps"] is None


def test_run_cell_lag_cell_pipeline_gets_lag_steps_and_none_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lag cell must pass lag_steps=[5] and windows=None to build_feature_pipeline."""
    from turbofan.experiments.feature_family_screen import run_cell

    cell = _make_lag_cell(seed=44)
    calls = _install_run_cell_patches(monkeypatch, cell, n_modes=1)
    run_cell(cell)

    kw = calls["build_feature_pipeline"][0]
    assert kw["feature_families"] == ["raw", "lag"]
    assert kw["lag_steps"] == [5]
    assert kw["windows"] is None


# ---------------------------------------------------------------------------
# run_cell — n_modes and sensor_cols_to_drop come from subset config
# ---------------------------------------------------------------------------


def test_run_cell_pipeline_inherits_n_modes_from_subset_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_feature_pipeline must receive n_modes from the subset config (6)."""
    from turbofan.experiments.feature_family_screen import run_cell

    cell = _make_rolling_cell(seed=44)
    calls = _install_run_cell_patches(monkeypatch, cell, n_modes=6)
    run_cell(cell)

    kw = calls["build_feature_pipeline"][0]
    assert kw["n_modes"] == 6, "n_modes must be inherited from subset config"


# ---------------------------------------------------------------------------
# run_cell — returned row has exactly CSV_COLUMNS keys
# ---------------------------------------------------------------------------


def test_run_cell_returns_dict_with_exactly_csv_columns_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_cell must return a dict whose key-set equals set(CSV_COLUMNS)."""
    from turbofan.experiments.feature_family_screen import run_cell

    cell = _make_rolling_cell(seed=44)
    _install_run_cell_patches(monkeypatch, cell)
    row = run_cell(cell)

    assert set(row.keys()) == set(CSV_COLUMNS)


# ---------------------------------------------------------------------------
# run_cell — fixed hyperparameter values in returned row
# ---------------------------------------------------------------------------


def test_run_cell_row_fixed_hyperparams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returned row must carry the spec-fixed hyperparameters and cell.seed."""
    from turbofan.experiments.feature_family_screen import run_cell

    cell = _make_rolling_cell(seed=44)
    _install_run_cell_patches(monkeypatch, cell)
    row = run_cell(cell)

    assert row["hidden_size"] == 64
    assert row["learning_rate"] == pytest.approx(1e-3)
    assert row["seed"] == 44
    assert row["n_features"] == _N_FEATURES


# ---------------------------------------------------------------------------
# run_cell — None sentinels render as "" in rolling/lag inapplicable fields
# ---------------------------------------------------------------------------


def test_run_cell_rolling_cell_lag_step_sentinel_is_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rolling cell row must have lag_step='' (not None) for CSV round-trip."""
    from turbofan.experiments.feature_family_screen import run_cell

    cell = _make_rolling_cell(seed=44)
    _install_run_cell_patches(monkeypatch, cell)
    row = run_cell(cell)

    assert row["lag_step"] == "", "lag_step must be '' for rolling cell"
    assert row["rolling_window"] == 5


def test_run_cell_raw_cell_both_sentinels_are_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw cell row must have rolling_window='' and lag_step='' for CSV round-trip."""
    from turbofan.experiments.feature_family_screen import run_cell

    cell = _make_raw_cell(seed=44)
    _install_run_cell_patches(monkeypatch, cell, n_modes=1)
    row = run_cell(cell)

    assert row["rolling_window"] == ""
    assert row["lag_step"] == ""


# ---------------------------------------------------------------------------
# run_cell — round-trip: append_row then completed_keys contains cell_key
# ---------------------------------------------------------------------------


def test_run_cell_row_round_trips_through_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Appending run_cell's row then calling completed_keys must find cell_key."""
    from turbofan.experiments.feature_family_screen import run_cell

    cell = _make_rolling_cell(seed=44)
    _install_run_cell_patches(monkeypatch, cell)
    row = run_cell(cell)

    p = csv_path(tmp_path, cell.architecture, cell.subset)
    append_row(p, row)
    keys = completed_keys(p)
    assert cell_key(cell) in keys


# ---------------------------------------------------------------------------
# run_cell — val_rmse and val_mae come from TrainingResult, not extra inference
# ---------------------------------------------------------------------------


def test_run_cell_metrics_come_from_training_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """val_rmse must equal result.best_metric; val_mae from history at best_epoch."""
    from turbofan.experiments.feature_family_screen import run_cell

    cell = _make_rolling_cell(seed=44)
    _install_run_cell_patches(monkeypatch, cell)
    row = run_cell(cell)

    assert row["val_rmse"] == pytest.approx(12.5), (
        "val_rmse must equal result.best_metric (12.5)"
    )
    assert row["val_mae"] == pytest.approx(9.0), (
        "val_mae must come from history row at best_epoch=3 (9.0)"
    )
    assert row["best_epoch"] == 3


# ---------------------------------------------------------------------------
# run_screen — completed cells are skipped
# ---------------------------------------------------------------------------


def _minimal_row(cell: ScreenCell) -> dict[str, Any]:
    """Build a valid run_cell-style row for use in fake_run_cell."""
    return {
        "architecture": cell.architecture,
        "subset": cell.subset,
        "feature_config": cell.feature_config,
        "rolling_window": "" if cell.rolling_window is None else cell.rolling_window,
        "lag_step": "" if cell.lag_step is None else cell.lag_step,
        "sequence_window": cell.sequence_window,
        "hidden_size": 64,
        "learning_rate": 1e-3,
        "seed": cell.seed,
        "n_features": 3,
        "n_train_windows": 80,
        "n_val_windows": 20,
        "best_epoch": 5,
        "val_rmse": 10.0,
        "val_mae": 8.0,
        "training_duration_seconds": 1.0,
    }


def test_run_screen_skips_already_completed_cell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """run_screen must skip a cell whose key is already in the CSV."""
    from turbofan.experiments.feature_family_screen import run_screen

    # Pre-write one cell's row so it appears completed
    pre_cell = ScreenCell(
        architecture="gru",
        subset="FD001",
        feature_config="raw",
        feature_families=["raw"],
        rolling_window=None,
        lag_step=None,
        sequence_window=30,
        seed=42,
    )
    p = csv_path(tmp_path / "results", "gru", "FD001")
    append_row(p, _minimal_row(pre_cell))

    called_cells: list[ScreenCell] = []

    def fake_run_cell(cell: ScreenCell, **kwargs: object) -> dict[str, Any]:
        called_cells.append(cell)
        return _minimal_row(cell)

    monkeypatch.setattr(_screen_module, "run_cell", fake_run_cell)

    run_screen(
        architectures=["gru"],
        subsets=["FD001"],
        sequence_windows=[30],
        rolling_windows=[5],
        lag_steps=[1],
        seeds=[42],
        results_dir=tmp_path / "results",
        configs_dir=tmp_path / "configs",
    )

    # The raw/sw=30/seed=42 cell must NOT have been called
    skipped_key = cell_key(pre_cell)
    called_keys = {cell_key(c) for c in called_cells}
    assert skipped_key not in called_keys, (
        "Completed cell must be skipped by run_screen"
    )


# ---------------------------------------------------------------------------
# run_screen — non-completed cells ARE called and appended
# ---------------------------------------------------------------------------


def test_run_screen_calls_run_cell_for_non_completed_cells(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """run_screen must call run_cell for every cell not in the CSV."""
    from turbofan.experiments.feature_family_screen import run_screen

    called_cells: list[ScreenCell] = []

    def fake_run_cell(cell: ScreenCell, **kwargs: object) -> dict[str, Any]:
        called_cells.append(cell)
        return _minimal_row(cell)

    monkeypatch.setattr(_screen_module, "run_cell", fake_run_cell)

    run_screen(
        architectures=["gru"],
        subsets=["FD001"],
        sequence_windows=[30],
        rolling_windows=[5],
        lag_steps=[1],
        seeds=[42],
        results_dir=tmp_path / "results",
        configs_dir=tmp_path / "configs",
    )

    # 1 raw + 6 rolling configs * 1 window + 1 lag config * 1 step = 8 cells
    assert len(called_cells) == 8


# ---------------------------------------------------------------------------
# run_screen — full resume: second call runs zero cells
# ---------------------------------------------------------------------------


def test_run_screen_full_resume_calls_run_cell_zero_times(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """After a complete run, a second run_screen call must skip all cells."""
    from turbofan.experiments.feature_family_screen import run_screen

    call_count = [0]

    def fake_run_cell(cell: ScreenCell, **kwargs: object) -> dict[str, Any]:
        call_count[0] += 1
        return _minimal_row(cell)

    monkeypatch.setattr(_screen_module, "run_cell", fake_run_cell)

    kwargs: dict[str, Any] = {
        "architectures": ["gru"],
        "subsets": ["FD001"],
        "sequence_windows": [30],
        "rolling_windows": [5],
        "lag_steps": [1],
        "seeds": [42],
        "results_dir": tmp_path / "results",
        "configs_dir": tmp_path / "configs",
    }
    run_screen(**kwargs)
    first_run_calls = call_count[0]
    assert first_run_calls > 0

    # Second run: all cells already in CSV — run_cell must not be called
    call_count[0] = 0
    run_screen(**kwargs)
    assert call_count[0] == 0, (
        "Second run_screen call must skip all cells (full resume)"
    )
