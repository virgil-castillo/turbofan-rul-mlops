"""Batch prediction CLI for local turbofan inference artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from turbofan.inference.predictors import load_predictor
from turbofan.inference.schemas import CANONICAL_COLUMNS, FEATURE_COLUMNS, RawRecords
from turbofan.inference.service import prediction_result_to_dict


def main(argv: Sequence[str] | None = None) -> int:
    """Run batch prediction from a CSV or JSON input file.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        records = _read_records(args.input)
        predictor = load_predictor(args.artifact)
        result = predictor.predict(records, allow_partial=args.allow_partial)
        payload = prediction_result_to_dict(result)
        _write_predictions(args.output, payload)
        _write_metadata(args.metadata_output, payload)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    metadata = result.metadata
    print(f"Artifact ID: {metadata.artifact_id}")
    print(f"Model type: {metadata.model_type}")
    print(f"Prediction count: {metadata.prediction_rows} predictions")
    print(f"Predictions output: {args.output}")
    print(f"Metadata output: {args.metadata_output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
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
    if pd.isna(value):
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
