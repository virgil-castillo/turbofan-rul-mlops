"""Tests for turbofan.cli.predict."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from turbofan import registry
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


def _constant_ridge_pipeline(constant: float = 42.0) -> Pipeline:
    """Build a Ridge-shaped pipeline that always predicts a constant.

    The pipeline consumes the full canonical frame and selects the feature
    columns internally, exactly like ``build_baseline_pipeline``, so the logged
    pyfunc Ridge wrapper scores it correctly. A ``DummyRegressor`` keeps the
    prediction deterministic for the metadata assertions.

    Args:
        constant: Constant RUL value every prediction returns.

    Returns:
        A fitted sklearn ``Pipeline`` mapping canonical rows to ``constant``.
    """
    rows = [
        _record(engine_id=engine_id, cycle=cycle)
        for engine_id in (1, 2)
        for cycle in range(1, 4)
    ]
    frame = pd.DataFrame(rows)
    pipeline = Pipeline(
        [
            (
                "features",
                ColumnTransformer([("keep", "passthrough", FEATURE_COLUMNS)]),
            ),
            ("model", DummyRegressor(strategy="constant", constant=constant)),
        ]
    )
    pipeline.fit(frame, [constant] * len(frame))
    return pipeline


def _register_ridge_model(subset: str = "FD001") -> str:
    """Register and promote a tiny Ridge model in the per-test MLflow store.

    Args:
        subset: C-MAPSS subset identifier for the registered-model name.

    Returns:
        The registered-model name (e.g. ``turbofan-ridge-fd001``).
    """
    pipeline = _constant_ridge_pipeline()
    name = registry.model_name("ridge", subset)
    with mlflow.start_run():
        version = registry.log_and_register(
            pipeline, model_type="ridge", subset=subset
        )
    registry.promote(name, version)
    return name


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
    name = _register_ridge_model()
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
        "--model",
        name,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--metadata-output",
        str(metadata_path),
    )

    assert result.returncode == 0, result.stderr
    with output_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["engine_id"], row["cycle"]) for row in rows] == [
        ("1", "3"),
        ("2", "1"),
    ]
    assert rows[0]["prediction"] == "42.0"
    metadata = json.loads(metadata_path.read_text())
    assert metadata == {
        "model_type": "ridge",
        "artifact_id": f"{name}/1",
        "prediction_scope": "engine",
        "input_rows": 2,
        "prediction_rows": 2,
        "warnings": [],
    }
    assert f"{name}/1" in result.stdout
    assert "ridge" in result.stdout
    assert "2 prediction" in result.stdout
    assert str(output_path) in result.stdout
    assert str(metadata_path) in result.stdout


def test_predict_cli_reads_json_records_object(
    tmp_path: Path,
) -> None:
    """CLI accepts JSON records envelopes and predicts per engine."""
    name = _register_ridge_model()
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "predictions.csv"
    metadata_path = tmp_path / "metadata.json"
    input_path.write_text(
        json.dumps(
            {
                "records": [
                    _record(engine_id=1, cycle=1),
                    _record(engine_id=1, cycle=2),
                ]
            }
        )
    )

    result = _run_predict(
        tmp_path,
        "--model",
        name,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--metadata-output",
        str(metadata_path),
    )

    assert result.returncode == 0, result.stderr
    with output_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["engine_id"] == "1"
    assert rows[0]["cycle"] == "2"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["input_rows"] == 2
    assert metadata["prediction_rows"] == 1
    assert metadata["warnings"] == []


def test_predict_cli_resolves_explicit_models_uri(tmp_path: Path) -> None:
    """CLI accepts an explicit models:/<name>@<alias> URI."""
    name = _register_ridge_model()
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "predictions.csv"
    metadata_path = tmp_path / "metadata.json"
    with input_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_record().keys()))
        writer.writeheader()
        writer.writerow(_record(engine_id=1, cycle=1))

    result = _run_predict(
        tmp_path,
        "--model",
        f"models:/{name}@production",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--metadata-output",
        str(metadata_path),
    )

    assert result.returncode == 0, result.stderr
    metadata = json.loads(metadata_path.read_text())
    assert metadata["artifact_id"] == f"{name}/production"
    assert metadata["model_type"] == "ridge"


def test_predict_cli_exits_nonzero_for_missing_input(tmp_path: Path) -> None:
    """CLI reports missing input paths on stderr and exits non-zero."""
    name = _register_ridge_model()

    result = _run_predict(
        tmp_path,
        "--model",
        name,
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
    name = _register_ridge_model()
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
        "--model",
        name,
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
    name = _register_ridge_model()
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "predictions.csv"
    metadata_path = tmp_path / "metadata.json"
    with input_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_record().keys()))
        writer.writeheader()
        writer.writerow(_record(engine_id=1, cycle=1))

    result = _run_predict(
        tmp_path,
        "--model",
        name,
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


def test_predict_cli_exits_nonzero_for_unknown_model(tmp_path: Path) -> None:
    """CLI reports model resolution errors on stderr and exits non-zero."""
    input_path = tmp_path / "input.csv"
    with input_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_record().keys()))
        writer.writeheader()
        writer.writerow(_record())

    result = _run_predict(
        tmp_path,
        "--model",
        "turbofan-ridge-does-not-exist",
        "--input",
        str(input_path),
        "--output",
        str(tmp_path / "predictions.csv"),
        "--metadata-output",
        str(tmp_path / "metadata.json"),
    )

    assert result.returncode != 0
    assert result.stderr.strip() != ""
