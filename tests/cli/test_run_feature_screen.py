"""Tests for the turbofan-feature-screen CLI."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from turbofan.cli import run_feature_screen


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_parser().parse_args([]) yields the spec defaults."""
    monkeypatch.delenv("DEVICE", raising=False)
    parser = run_feature_screen._build_parser()
    args = parser.parse_args([])

    assert args.subsets == ["FD001", "FD002", "FD003", "FD004"]
    assert args.architectures == ["gru", "lstm"]
    assert args.seeds == [42]
    assert args.rolling_windows == [5, 20]
    assert args.lag_steps == [1, 5]
    assert args.sequence_windows == [30, 60]
    assert args.results_dir == Path("outputs/results")
    assert args.configs_dir == Path("configs/subsets")
    assert args.device == "auto"


def test_device_override() -> None:
    """--device accepts cpu/cuda/auto and rejects anything else."""
    parser = run_feature_screen._build_parser()
    assert parser.parse_args(["--device", "cuda"]).device == "cuda"
    assert parser.parse_args(["--device", "cpu"]).device == "cpu"
    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "tpu"])


def test_overrides() -> None:
    """Parsing explicit flags overrides all defaults correctly."""
    parser = run_feature_screen._build_parser()
    args = parser.parse_args(
        [
            "--seeds",
            "43",
            "44",
            "45",
            "46",
            "--subsets",
            "FD001",
            "--architectures",
            "gru",
        ]
    )

    assert args.seeds == [43, 44, 45, 46]
    assert args.subsets == ["FD001"]
    assert args.architectures == ["gru"]


def test_invalid_architecture_rejected() -> None:
    """Parsing --architectures with an invalid choice raises SystemExit."""
    parser = run_feature_screen._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--architectures", "cnn"])


def test_invalid_subset_rejected() -> None:
    """Parsing --subsets with an invalid choice raises SystemExit."""
    parser = run_feature_screen._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--subsets", "FD999"])


def test_main_wiring() -> None:
    """main() calls run_screen with parsed kwargs and returns 0."""
    captured: dict[str, object] = {}

    def fake_run_screen(**kwargs: object) -> None:
        captured.update(kwargs)  # type: ignore[arg-type]

    with patch(
        "turbofan.experiments.feature_family_screen.run_screen", fake_run_screen
    ):
        code = run_feature_screen.main(
            [
                "--subsets",
                "FD001",
                "--architectures",
                "gru",
                "--seeds",
                "42",
                "--sequence-windows",
                "30",
                "--device",
                "cuda",
            ]
        )

    assert code == 0
    assert captured["architectures"] == ["gru"]
    assert captured["subsets"] == ["FD001"]
    assert captured["seeds"] == [42]
    assert captured["sequence_windows"] == [30]
    assert captured["rolling_windows"] == [5, 20]
    assert captured["lag_steps"] == [1, 5]
    assert captured["results_dir"] == Path("outputs/results")
    assert captured["configs_dir"] == Path("configs/subsets")
    assert captured["device"] == "cuda"
