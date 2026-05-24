"""Smoke tests for scripts/train_sequence_gru.py."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _write_cmapps_file(path: Path, n_engines: int, n_cycles: int) -> None:
    """Write a small C-MAPSS-style whitespace-delimited file.

    Args:
        path: Destination file path.
        n_engines: Number of synthetic engines to write.
        n_cycles: Number of cycles per engine.
    """
    lines = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, n_cycles + 1):
            op_cols = [0.0, 0.0, 0.0]
            sensors = [
                float((engine_id * 0.1) + (cycle * 0.2) + sensor_idx)
                for sensor_idx in range(1, 22)
            ]
            values = [engine_id, cycle, *op_cols, *sensors]
            lines.append(" ".join(str(value) for value in values))
    path.write_text("\n".join(lines))


def _write_config(
    path: Path,
    raw_dir: Path,
    artifact_dir: Path,
    tmp_path: Path,
) -> None:
    """Write a tiny GRU training config.

    Args:
        path: Destination YAML path.
        raw_dir: Raw C-MAPSS data directory.
        artifact_dir: Artifact root.
        tmp_path: Temporary test directory for other configured paths.
    """
    path.write_text(
        "\n".join(
            [
                "project_name: test",
                "data:",
                f"  raw_dir: {raw_dir.as_posix()}",
                f"  processed_dir: {(tmp_path / 'processed').as_posix()}",
                f"  interim_dir: {(tmp_path / 'interim').as_posix()}",
                "  fd_subset: FD001",
                "  max_rul: 30",
                "  test_size: 0.25",
                "  random_seed: 42",
                "sequence:",
                "  architecture: gru",
                "  window_size: 3",
                "  batch_size: 4",
                "  hidden_size: 4",
                "  num_layers: 1",
                "  dropout: 0.0",
                "  learning_rate: 0.001",
                "  epochs: 2",
                "  patience: 2",
                "  device: cpu",
                f"  artifact_dir: {artifact_dir.as_posix()}",
            ]
        )
    )


def _run_cli(cfg_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the sequence GRU CLI with the worktree source path.

    Args:
        cfg_path: YAML config path.

    Returns:
        Completed subprocess result.
    """
    project_root = Path(__file__).parent.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    return subprocess.run(
        [sys.executable, "scripts/train_sequence_gru.py", "--config", str(cfg_path)],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_train_sequence_gru_cli_writes_artifacts_with_official_test(
    tmp_path: Path,
) -> None:
    """CLI trains a tiny GRU and writes validation plus official artifacts."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)
    _write_cmapps_file(raw_dir / "test_FD001.txt", n_engines=2, n_cycles=5)
    (raw_dir / "RUL_FD001.txt").write_text("10\n20\n")

    artifact_dir = tmp_path / "artifacts"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_dir, artifact_dir, tmp_path)

    result = _run_cli(cfg_path)

    assert "validation_final_window rmse" in result.stdout
    run_dirs = list((artifact_dir / "sequence_gru").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "training_history.csv").exists()
    assert (run_dir / "validation_final_window_predictions.csv").exists()
    assert (run_dir / "validation_window_predictions.csv").exists()
    assert (run_dir / "official_test_predictions.csv").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {
        "validation_final_window",
        "validation_windows",
        "official_test",
    }
    assert set(metrics["validation_final_window"]) == {
        "rmse",
        "mae",
        "phm08_score",
    }


def test_train_sequence_gru_cli_skips_missing_official_test(
    tmp_path: Path,
) -> None:
    """CLI skips official evaluation when test or RUL files are absent."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_cmapps_file(raw_dir / "train_FD001.txt", n_engines=4, n_cycles=6)

    artifact_dir = tmp_path / "artifacts"
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, raw_dir, artifact_dir, tmp_path)

    result = _run_cli(cfg_path)

    assert "validation_final_window rmse" in result.stdout
    assert "official test evaluation skipped" in result.stdout
    run_dir = next((artifact_dir / "sequence_gru").iterdir())
    assert not (run_dir / "official_test_predictions.csv").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics) == {"validation_final_window", "validation_windows"}
