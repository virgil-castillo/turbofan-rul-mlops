"""Batch prediction CLI resolving the production model from the registry."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from turbofan import registry
from turbofan.data.contracts import CANONICAL_COLUMNS, FEATURE_COLUMNS
from turbofan.evaluation import metrics
from turbofan.predictions import serialization
from turbofan.predictions.contracts import RawRecords
from turbofan.predictions.predictor import PyfuncPredictor
from turbofan.utils import logging as turbofan_logging

logger = turbofan_logging.get_logger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Run batch prediction from a CSV or JSON input file.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    turbofan_logging.setup_logging(args.log_level)
    evaluation: dict[str, float] | None = None
    try:
        records = _read_records(args.input)
        predictor = _resolve_predictor(args.model, args.alias)
        result = predictor.predict(records, allow_partial=args.allow_partial)
        payload = serialization.prediction_result_to_dict(result)
        _write_predictions(args.output, payload)
        predictions_list = payload["predictions"]
        if not isinstance(predictions_list, list):
            raise ValueError("Serialized predictions must be a list.")
        evaluation = _try_evaluate(predictions_list, args.data_dir, args.subset)
        if evaluation is not None:
            meta = payload["metadata"]
            if isinstance(meta, dict):
                meta["evaluation"] = evaluation
        _write_metadata(args.metadata_output, payload)
    except Exception as exc:  # noqa: BLE001 - CLI boundary surfaces any failure as exit 1
        logger.error(str(exc))
        logger.debug("Traceback for the error above:", exc_info=True)
        return 1

    metadata = result.metadata
    print(f"Artifact ID: {metadata.artifact_id}")
    print(f"Model type: {metadata.model_type}")
    print(f"Prediction count: {metadata.prediction_rows} predictions")
    print(f"Predictions output: {args.output}")
    print(f"Metadata output: {args.metadata_output}")
    if evaluation is not None:
        print(f"RMSE: {evaluation['rmse']:.4f}")
        print(f"MAE: {evaluation['mae']:.4f}")
        print(f"PHM08 Score: {evaluation['phm08_score']:.4f}")
    return 0


def _resolve_predictor(model: str, alias: str) -> PyfuncPredictor:
    """Resolve the prediction model from the registry by name or models URI.

    Args:
        model: Registered-model name (e.g. ``turbofan-gru-fd001``) or an
            explicit ``models:/<name>@<alias>`` URI.
        alias: Alias to resolve when ``model`` is a bare name.

    Returns:
        A loaded predictor adapter for the resolved registry model.
    """
    registry.tracking.configure_mlflow()
    if model.startswith("models:/"):
        return registry.load_predictor_from_uri(model)
    return registry.load_predictor(model, alias)


def _try_evaluate(
    predictions: list[dict[str, object]],
    data_dir: Path | None,
    subset: str | None,
) -> dict[str, float] | None:
    """Evaluate predictions against official RUL labels if available.

    Predictions must be sorted by ascending engine_id to align with the
    label file order (one label per engine in engine-id order).

    Args:
        predictions: Serialized prediction rows sorted by engine_id.
        data_dir: Directory containing RUL label files.
        subset: C-MAPSS subset identifier.

    Returns:
        Metric dictionary or None when labels are unavailable.
    """
    if data_dir is None or subset is None:
        return None
    labels_path = data_dir / f"RUL_{subset}.txt"
    if not labels_path.exists():
        return None
    labels = pd.read_csv(labels_path, header=None).iloc[:, 0].to_numpy(
        dtype=np.float64,
    )
    if len(labels) != len(predictions):
        return None
    y_pred = np.array(
        [float(str(row["prediction"])) for row in predictions],
        dtype=np.float64,
    )
    return metrics.official_test_metrics(labels, y_pred)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help=(
            "Registered-model name (e.g. turbofan-gru-fd001) or an explicit "
            "models:/<name>@<alias> URI."
        ),
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="production",
        help="Alias to resolve for a bare model name (defaults to production).",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Skip individually invalid rows (and duplicate engine_id/cycle "
            "rows) instead of failing the request; each skipped row is "
            "reported as a warning in the prediction metadata."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging verbosity (falls back to the LOG_LEVEL env var or INFO).",
    )
    return parser


def _read_records(path: Path) -> RawRecords:
    if not path.exists():
        raise ValueError(f"Input path does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_records(path)
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return _records_from_json_value(payload)
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return _records_from_json_value(payload["records"])
        raise ValueError(
            "JSON input must be a record list or object with records list."
        )
    raise ValueError("Input path must have .csv or .json extension.")


def _read_csv_records(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path, dtype=object)
    records = frame.to_dict("records")
    return [
        {
            str(column): _coerce_csv_cell(str(column), value)
            for column, value in record.items()
        }
        for record in records
    ]


def _coerce_csv_cell(column: str, value: object) -> object:
    if column not in CANONICAL_COLUMNS:
        return value
    if value is None:
        return value
    if isinstance(value, float) and math.isnan(value):
        return value
    if column in FEATURE_COLUMNS:
        return _coerce_csv_feature(value)
    return _coerce_csv_identifier(value)


def _coerce_csv_identifier(value: object) -> object:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            number = float(stripped)
        except ValueError:
            return value
        if math.isfinite(number) and number.is_integer():
            return int(number)
        return number
    return value


def _coerce_csv_feature(value: object) -> object:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return float(stripped)
        except ValueError:
            return value
    return value


def _records_from_json_value(value: list[object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"JSON record at index {index} must be an object.")
        records.append({str(key): record_value for key, record_value in item.items()})
    return records


def _write_predictions(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = payload["predictions"]
    if not isinstance(predictions, list):
        raise ValueError("Serialized predictions must be a list.")
    pd.DataFrame(predictions).to_csv(path, index=False)


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = payload["metadata"]
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
