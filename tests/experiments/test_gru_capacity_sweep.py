"""Tests for the Stage 2 GRU capacity sweep CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from turbofan.experiments.gru_capacity_sweep import (
    build_stage2_specs,
    select_top_k_from_stage1,
)


def test_select_top_k_from_stage1_picks_best_by_phm08(tmp_path: Path) -> None:
    csv = tmp_path / "stage1.csv"
    pd.DataFrame(
        {
            "feature_set": ["raw", "rolling_mean", "rolling_mean"],
            "windows": ["()", "(15,)", "(20,)"],
            "sequence_window_size": [30, 45, 60],
            "hidden_size": [64, 64, 64],
            "learning_rate": [0.001, 0.001, 0.001],
            "phm08_score": [300.0, 250.0, 275.0],
            "rmse": [25.0, 22.0, 23.0],
        }
    ).to_csv(csv, index=False)

    top = select_top_k_from_stage1(csv, k=2)
    assert len(top) == 2
    assert top[0]["feature_set"] == "rolling_mean"
    assert top[0]["windows"] == "(15,)"
    assert top[1]["windows"] == "(20,)"


def test_build_stage2_specs_crosses_hidden_and_lr() -> None:
    bases = [
        {
            "feature_set": "rolling_mean",
            "windows": "(15,)",
            "sequence_window_size": 45,
        }
    ]
    specs = build_stage2_specs(
        bases, hidden_sizes=[32, 64, 128], learning_rates=[0.001, 0.0003]
    )
    assert len(specs) == 6
    assert {(s.hidden_size, s.learning_rate) for s in specs} == {
        (32, 0.001), (32, 0.0003),
        (64, 0.001), (64, 0.0003),
        (128, 0.001), (128, 0.0003),
    }
