"""Tests for turbofan.models.test_evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from turbofan.config.schema import DataConfig
from turbofan.models.gru import GRURULRegressor
from turbofan.models.test_evaluation import (
    align_labels_to_eligible_engines,
    evaluate_official_test,
    evaluate_test_from_df,
)
from turbofan.sequences.normalize import SequenceNormalizer


class TestAlignLabelsToEligibleEngines:
    """Tests for align_labels_to_eligible_engines."""

    def test_selects_labels_for_eligible_engines(self) -> None:
        """Labels for eligible engine IDs are selected and aligned."""
        metadata = pd.DataFrame({"engine_id": [1, 3], "cycle": [10, 20]})
        rul_labels = pd.Series([100, 200, 300], name="rul")

        result = align_labels_to_eligible_engines(metadata, rul_labels)

        assert list(result) == [100.0, 300.0]
        assert result.name == "rul"
        assert result.dtype == np.float64

    def test_raises_on_out_of_range_engine_id(self) -> None:
        """Engine IDs beyond available labels raise ValueError."""
        metadata = pd.DataFrame({"engine_id": [1, 5], "cycle": [10, 20]})
        rul_labels = pd.Series([100, 200, 300], name="rul")

        with pytest.raises(ValueError, match="eligible test engine"):
            align_labels_to_eligible_engines(metadata, rul_labels)


class TestEvaluateTestFromDf:
    """Tests for evaluate_test_from_df."""

    def test_returns_test_metrics(self) -> None:
        """Returns dict with test_rmse, test_mae, test_phm08_score keys."""
        n_engines = 3
        n_cycles = 5
        feature_cols = ["op_1", "s_1", "s_2"]

        rows = []
        for eid in range(1, n_engines + 1):
            for cycle in range(1, n_cycles + 1):
                rows.append({
                    "engine_id": eid,
                    "cycle": cycle,
                    "op_1": 0.0,
                    "s_1": float(cycle),
                    "s_2": float(cycle * 2),
                })
        test_df = pd.DataFrame(rows)
        rul_labels = pd.Series([10, 20, 30], name="rul")

        normalizer = SequenceNormalizer(feature_cols=feature_cols)
        normalizer.fit_transform(test_df.copy())

        model = GRURULRegressor(
            input_size=len(feature_cols),
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
        )
        device = torch.device("cpu")

        result = evaluate_test_from_df(
            test_df=test_df,
            rul_labels=rul_labels,
            model=model,
            normalizer=normalizer,
            feature_cols=feature_cols,
            device=device,
            window_size=3,
            batch_size=8,
            max_rul=125,
        )

        assert set(result.keys()) == {"test_rmse", "test_mae", "test_phm08_score"}
        assert all(isinstance(v, float) for v in result.values())
        assert all(v >= 0.0 for v in result.values())


class TestEvaluateOfficialTest:
    """Tests for evaluate_official_test."""

    def test_returns_none_when_test_files_missing(self, tmp_path: Path) -> None:
        """Returns None when test or RUL files do not exist."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        data_config = DataConfig(
            raw_dir=raw_dir,
            processed_dir=tmp_path / "processed",
            interim_dir=tmp_path / "interim",
        )
        feature_cols = ["op_1", "s_1"]
        normalizer = SequenceNormalizer(feature_cols=feature_cols)

        rows = []
        for eid in range(1, 3):
            for cycle in range(1, 6):
                rows.append({
                    "engine_id": eid,
                    "cycle": cycle,
                    "op_1": 0.0,
                    "s_1": float(cycle),
                })
        normalizer.fit_transform(pd.DataFrame(rows))

        model = GRURULRegressor(
            input_size=len(feature_cols),
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
        )

        result = evaluate_official_test(
            data_config=data_config,
            model=model,
            normalizer=normalizer,
            feature_cols=feature_cols,
            device=torch.device("cpu"),
            window_size=3,
            batch_size=8,
        )

        assert result is None

    def test_returns_metrics_when_files_exist(self, tmp_path: Path) -> None:
        """Returns test metrics dict when test and RUL files exist."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        n_engines = 3
        n_cycles = 5
        lines = []
        for eid in range(1, n_engines + 1):
            for cycle in range(1, n_cycles + 1):
                op_cols = [0.0, 0.0, 0.0]
                sensors = [float(cycle + si) for si in range(1, 22)]
                values = [eid, cycle, *op_cols, *sensors]
                lines.append(" ".join(str(v) for v in values))
        (raw_dir / "test_FD001.txt").write_text("\n".join(lines))
        (raw_dir / "RUL_FD001.txt").write_text("10\n20\n30\n")

        train_lines = list(lines)
        (raw_dir / "train_FD001.txt").write_text("\n".join(train_lines))

        data_config = DataConfig(
            raw_dir=raw_dir,
            processed_dir=tmp_path / "processed",
            interim_dir=tmp_path / "interim",
            fd_subset="FD001",
            max_rul=125,
        )
        feature_cols = [
            "op_1", "op_2", "op_3",
            *[f"s_{i}" for i in range(1, 22)],
        ]
        normalizer = SequenceNormalizer(feature_cols=feature_cols)

        from turbofan.data.loader import load_raw_train
        train_raw = load_raw_train(data_config)
        normalizer.fit_transform(train_raw)

        model = GRURULRegressor(
            input_size=len(feature_cols),
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
        )

        result = evaluate_official_test(
            data_config=data_config,
            model=model,
            normalizer=normalizer,
            feature_cols=feature_cols,
            device=torch.device("cpu"),
            window_size=3,
            batch_size=8,
        )

        assert result is not None
        assert set(result.keys()) == {"test_rmse", "test_mae", "test_phm08_score"}
        assert all(isinstance(v, float) for v in result.values())
