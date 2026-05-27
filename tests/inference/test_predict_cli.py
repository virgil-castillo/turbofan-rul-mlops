"""Tests for turbofan.cli.predict."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
from sklearn.dummy import DummyRegressor

from turbofan.inference.schemas import FEATURE_COLUMNS


def _record(
    *,
    engine_id: int = 1,
    cycle: int = 1,
    feature_value: float = 1.0,
) -> dict[str, object]:
    """Build one canonical inference record.

    Args:
        engine_id: Engine identifier.
        cycle: Cycle identifier.
        feature_value: Value used for all feature columns.

    Returns:
        Canonical record mapping.
    """
    return {
        "engine_id": engine_id,
        "cycle": cycle,
        **{column: feature_value for column in FEATURE_COLUMNS},
    }


def _write_ridge_artifact(tmp_path: Path) -> Path:
    """Write a synthetic Ridge artifact for CLI tests.

    Args:
        tmp_path: Temporary test directory.

    Returns:
        Path to the model manifest.
    """
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    model = DummyRegressor(strategy="constant", constant=42.0)
    model.fit([[0.0], [1.0]], [42.0, 42.0])
    joblib.dump(model, artifact_dir / "model.joblib")
    manifest_path = artifact_dir / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "ridge",
                "artifact_id": "ridge-cli-test",
                "prediction_scope": "engine",
                "model_path": "model.joblib",
            }
        )
    )
    return manifest_path


def _run_predict(
    tmp_path: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    """Run the predict command module with the worktree src on PYTHONPATH.

    Args:
        tmp_path: Temporary test directory.
        *args: Additional command-line arguments.

    Returns:
        Completed subprocess result.
    """
    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root / "src"),
    }
    return subprocess.run(
        [sys.executable, "-m", "turbofan.cli.predict", *args],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_predict_cli_reads_csv_and_writes_predictions_and_metadata(
    tmp_path: Path,
) -> None:
    """CLI writes prediction CSV, metadata JSON, and a useful summary."""
    artifact_path = _write_ridge_artifact(tmp_path)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "predictions.csv"
    metadata_path = tmp_path / "metadata.json"
    with input_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_record().keys()))
        writer.writeheader()
        writer.writerow(_record(engine_id=2, cycle=1))
        writer.writerow(_record(engine_id=1, cycle=3))

    result = _run_predict(
        tmp_path,
        "--artifact",
        str(artifact_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--metadata-output",
        str(metadata_path),
    )

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output_path.open()))
    assert [(row["engine_id"], row["cycle"]) for row in rows] == [
        ("1", "3"),
        ("2", "1"),
    ]
    assert rows[0]["prediction"] == "42.0"
    metadata = json.loads(metadata_path.read_text())
    assert metadata == {
        "model_type": "ridge",
        "artifact_id": "ridge-cli-test",
        "prediction_scope": "engine",
        "input_rows": 2,
        "prediction_rows": 2,
        "warnings": [],
    }
    assert "ridge-cli-test" in result.stdout
    assert "ridge" in result.stdout
    assert "2 prediction" in result.stdout
    assert str(output_path) in result.stdout
    assert str(metadata_path) in result.stdout


def test_predict_cli_reads_json_records_object_and_allows_partial_rows(
    tmp_path: Path,
) -> None:
    """CLI accepts JSON records envelopes and forwards allow_partial."""
    artifact_path = _write_ridge_artifact(tmp_path)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "predictions.csv"
    metadata_path = tmp_path / "metadata.json"
    invalid_record = _record(engine_id=1, cycle=2)
    invalid_record["s_21"] = "bad"
    input_path.write_text(
        json.dumps({"records": [_record(engine_id=1, cycle=1), invalid_record]})
    )

    result = _run_predict(
        tmp_path,
        "--artifact",
        str(artifact_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--metadata-output",
        str(metadata_path),
        "--allow-partial",
    )

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output_path.open()))
    assert len(rows) == 1
    metadata = json.loads(metadata_path.read_text())
    assert metadata["input_rows"] == 2
    assert metadata["prediction_rows"] == 1
    assert len(metadata["warnings"]) == 1
    assert "row 1" in metadata["warnings"][0]


def test_predict_cli_allows_partial_csv_rows_with_one_bad_numeric_cell(
    tmp_path: Path,
) -> None:
    """CLI scores valid CSV rows when one row has a bad numeric feature."""
    artifact_path = _write_ridge_artifact(tmp_path)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "predictions.csv"
    metadata_path = tmp_path / "metadata.json"
    invalid_record = _record(engine_id=1, cycle=2)
    invalid_record["s_21"] = "bad"
    with input_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_record().keys()))
        writer.writeheader()
        writer.writerow(_record(engine_id=1, cycle=1))
        writer.writerow(invalid_record)

    result = _run_predict(
        tmp_path,
        "--artifact",
        str(artifact_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--metadata-output",
        str(metadata_path),
        "--allow-partial",
    )

    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output_path.open()))
    assert len(rows) == 1
    assert rows[0]["engine_id"] == "1"
    assert rows[0]["cycle"] == "1"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["input_rows"] == 2
    assert metadata["prediction_rows"] == 1
    assert len(metadata["warnings"]) == 1
    assert "row 1" in metadata["warnings"][0]


def test_predict_cli_exits_nonzero_for_missing_input(tmp_path: Path) -> None:
    """CLI reports missing input paths on stderr and exits non-zero."""
    artifact_path = _write_ridge_artifact(tmp_path)

    result = _run_predict(
        tmp_path,
        "--artifact",
        str(artifact_path),
        "--input",
        str(tmp_path / "missing.csv"),
        "--output",
        str(tmp_path / "predictions.csv"),
        "--metadata-output",
        str(tmp_path / "metadata.json"),
    )

    assert result.returncode != 0
    assert "Input path does not exist" in result.stderr


def _write_rul_labels(data_dir: Path, subset: str, labels: list[int]) -> Path:
    """Write a synthetic RUL labels file.

    Args:
        data_dir: Directory for the labels file.
        subset: C-MAPSS subset name (e.g. "FD001").
        labels: RUL values, one per engine.

    Returns:
        Path to the written labels file.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"RUL_{subset}.txt"
    path.write_text("\n".join(str(label) for label in labels) + "\n")
    return path


def test_predict_cli_evaluates_against_rul_labels_when_available(
    tmp_path: Path,
) -> None:
    """CLI computes and prints metrics when RUL labels file is found."""
    artifact_path = _write_ridge_artifact(tmp_path)
    data_dir = tmp_path / "data"
    _write_rul_labels(data_dir, "FD001", [40, 45])
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "predictions.csv"
    metadata_path = tmp_path / "metadata.json"
    with input_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_record().keys()))
        writer.writeheader()
        writer.writerow(_record(engine_id=1, cycle=1))
        writer.writerow(_record(engine_id=2, cycle=1))

    result = _run_predict(
        tmp_path,
        "--artifact",
        str(artifact_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--metadata-output",
        str(metadata_path),
        "--data-dir",
        str(data_dir),
        "--subset",
        "FD001",
    )

    assert result.returncode == 0, result.stderr
    assert "RMSE" in result.stdout
    assert "MAE" in result.stdout
    assert "PHM08" in result.stdout
    metadata = json.loads(metadata_path.read_text())
    assert "evaluation" in metadata
    assert "rmse" in metadata["evaluation"]
    assert "mae" in metadata["evaluation"]
    assert "phm08_score" in metadata["evaluation"]


def test_predict_cli_skips_evaluation_when_rul_labels_missing(
    tmp_path: Path,
) -> None:
    """CLI skips evaluation silently when no RUL labels file exists."""
    artifact_path = _write_ridge_artifact(tmp_path)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "predictions.csv"
    metadata_path = tmp_path / "metadata.json"
    with input_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_record().keys()))
        writer.writeheader()
        writer.writerow(_record(engine_id=1, cycle=1))

    result = _run_predict(
        tmp_path,
        "--artifact",
        str(artifact_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--metadata-output",
        str(metadata_path),
        "--data-dir",
        str(tmp_path / "no_data"),
        "--subset",
        "FD001",
    )

    assert result.returncode == 0, result.stderr
    assert "RMSE" not in result.stdout
    metadata = json.loads(metadata_path.read_text())
    assert "evaluation" not in metadata


def test_predict_cli_exits_nonzero_for_invalid_artifact(tmp_path: Path) -> None:
    """CLI reports artifact loading errors on stderr and exits non-zero."""
    input_path = tmp_path / "input.csv"
    with input_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_record().keys()))
        writer.writeheader()
        writer.writerow(_record())

    result = _run_predict(
        tmp_path,
        "--artifact",
        str(tmp_path / "missing_manifest.json"),
        "--input",
        str(input_path),
        "--output",
        str(tmp_path / "predictions.csv"),
        "--metadata-output",
        str(tmp_path / "metadata.json"),
    )

    assert result.returncode != 0
    assert "Manifest path does not exist" in result.stderr
