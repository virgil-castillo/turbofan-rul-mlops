"""Tests for turbofan.cli.predict."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import NamedTuple

import mlflow
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from turbofan import registry
from turbofan.cli.predict import main as predict_main
from turbofan.data.contracts import FEATURE_COLUMNS


class _CliResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


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
        **dict.fromkeys(FEATURE_COLUMNS, feature_value),
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
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> _CliResult:
    """Run the predict CLI in-process and return captured output.

    Args:
        capsys: pytest capsys fixture for stdout/stderr capture.
        *args: Command-line arguments forwarded to main().

    Returns:
        CLI result with returncode, stdout, and stderr.
    """
    returncode = predict_main(list(args))
    captured = capsys.readouterr()
    return _CliResult(returncode=returncode, stdout=captured.out, stderr=captured.err)


def test_predict_cli_reads_csv_and_writes_predictions_and_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
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
        capsys,
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
    capsys: pytest.CaptureFixture[str],
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
        capsys,
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


def test_predict_cli_resolves_explicit_models_uri(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
        capsys,
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


def test_predict_cli_exits_nonzero_for_missing_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI reports missing input paths on stderr and exits non-zero."""
    name = _register_ridge_model()

    result = _run_predict(
        capsys,
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
    capsys: pytest.CaptureFixture[str],
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
        capsys,
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
    capsys: pytest.CaptureFixture[str],
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
        capsys,
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


def _make_normalizer_payload(feature_cols: list[str]) -> dict[str, object]:
    """Fit a minimal ``OperatingModeNormalizer`` and return its payload dict.

    Args:
        feature_cols: Feature columns to include in the normalizer.

    Returns:
        Payload dictionary produced by ``OperatingModeNormalizer.to_payload``.
    """
    from turbofan.preprocessing.normalization import OperatingModeNormalizer

    normalizer = OperatingModeNormalizer(feature_cols=list(feature_cols))
    fit_df = pd.DataFrame({col: [0.0, 1.0] for col in feature_cols})
    fit_df["op_1"] = [0.0, 0.0]
    fit_df["op_2"] = [0.0, 0.0]
    fit_df["op_3"] = [0.0, 0.0]
    normalizer.fit(fit_df)
    return normalizer.to_payload()


def _register_lstm_model(subset: str = "FD001", *, window_size: int = 3) -> str:
    """Register and promote a tiny LSTM model in the per-test MLflow store.

    Args:
        subset: C-MAPSS subset identifier for the registered-model name.
        window_size: Sequence window size stored in the checkpoint payload.

    Returns:
        The registered-model name (e.g. ``turbofan-lstm-fd001``).
    """
    import torch

    from turbofan.models.sequence_models import build_sequence_model

    torch.manual_seed(0)
    model = build_sequence_model(
        "lstm",
        input_size=len(FEATURE_COLUMNS),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    payload: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "feature_cols": list(FEATURE_COLUMNS),
        "sequence_config": {
            "architecture": "lstm",
            "window_size": window_size,
            "hidden_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "normalizer_type": "operating_mode",
        "normalizer_payload": _make_normalizer_payload(FEATURE_COLUMNS),
        "max_rul": 125,
    }
    name = registry.model_name("lstm", subset)
    with mlflow.start_run():
        version = registry.log_and_register(payload, model_type="lstm", subset=subset)
    registry.promote(name, version)
    return name


def test_predict_cli_serves_registered_lstm_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI resolves a turbofan-lstm-<subset> model and writes final-window preds.

    Drives the real predict CLI end-to-end against a promoted LSTM registry model
    (an engine with enough cycles for the window), asserting one final-window
    prediction per engine and LSTM-typed metadata.
    """
    name = _register_lstm_model("FD001", window_size=3)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "predictions.csv"
    metadata_path = tmp_path / "metadata.json"
    with input_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_record().keys()))
        writer.writeheader()
        for cycle in range(1, 6):
            writer.writerow(_record(engine_id=1, cycle=cycle))

    result = _run_predict(
        capsys,
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
    assert float(rows[0]["prediction"]) >= 0.0
    metadata = json.loads(metadata_path.read_text())
    assert metadata["model_type"] == "lstm"
    assert metadata["prediction_scope"] == "final_window"
    assert metadata["artifact_id"] == f"{name}/1"


def test_predict_cli_exits_nonzero_for_unknown_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI reports model resolution errors on stderr and exits non-zero."""
    input_path = tmp_path / "input.csv"
    with input_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_record().keys()))
        writer.writeheader()
        writer.writerow(_record())

    result = _run_predict(
        capsys,
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
