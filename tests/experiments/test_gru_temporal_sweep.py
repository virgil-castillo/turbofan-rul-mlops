"""Tests for the Stage 1 GRU temporal-context sweep CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from turbofan.experiments.gru_temporal_sweep import (
    build_stage1_specs,
    run_stage1_sweep,
)


def test_build_stage1_specs_emits_rolling_grid_plus_raw_control() -> None:
    specs = build_stage1_specs(
        rolling_feature_set="rolling_mean",
        rolling_windows=[10, 15, 20],
        sequence_window_sizes=[30, 45, 60],
    )
    rolling = [s for s in specs if s.feature_set == "rolling_mean"]
    raw = [s for s in specs if s.feature_set == "raw"]
    assert len(rolling) == 9  # 3 rolling windows x 3 sequence windows
    assert len(raw) == 3      # raw has no rolling window axis
    rolling_pairs = {(s.windows[0], s.sequence_window_size) for s in rolling}
    assert rolling_pairs == {
        (w, sw) for w in (10, 15, 20) for sw in (30, 45, 60)
    }
    assert {s.sequence_window_size for s in raw} == {30, 45, 60}
    assert all(s.lag_steps == () for s in specs)


def test_run_stage1_sweep_writes_csv_with_required_columns(
    tmp_path: Path, tiny_config_path: Path
) -> None:
    output_path = tmp_path / "stage1_fd001.csv"
    run_stage1_sweep(
        config_path=tiny_config_path,
        rolling_feature_set="rolling_mean",
        rolling_windows=[5],
        sequence_window_sizes=[15, 20],
        device="cpu",
        output_path=output_path,
    )
    assert output_path.exists()
    df = pd.read_csv(output_path)
    for column in (
        "feature_set",
        "windows",
        "sequence_window_size",
        "hidden_size",
        "learning_rate",
        "best_epoch",
        "rmse",
        "n_engines_total",
        "n_engines_padded",
        "n_engines_full",
    ):
        assert column in df.columns
    # 1 rolling window x 2 sequence sizes = 2, plus 2 raw = 4 total
    assert len(df) == 4
