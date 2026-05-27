"""Send test data to the running inference API and compare against true RUL labels.

Usage:
    python scripts/query_api.py
    python scripts/query_api.py --subset FD002 --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import pandas as pd

COLUMN_NAMES: list[str] = [
    "engine_id",
    "cycle",
    "op_1",
    "op_2",
    "op_3",
    *[f"s_{i}" for i in range(1, 22)],
]


def load_test(data_dir: Path, subset: str) -> pd.DataFrame:
    path = data_dir / f"test_{subset}.txt"
    df: pd.DataFrame = pd.read_csv(path, sep=r"\s+", header=None, index_col=False)
    df = df.iloc[:, : len(COLUMN_NAMES)]
    df.columns = pd.Index(COLUMN_NAMES)
    return df


def load_rul(data_dir: Path, subset: str) -> pd.Series[int]:
    path = data_dir / f"RUL_{subset}.txt"
    return pd.read_csv(path, header=None).iloc[:, 0].rename("rul")


def post_predict(url: str, records: list[dict[str, object]]) -> dict[str, object]:
    body = json.dumps({"records": records, "allow_partial": True}).encode()
    req = urllib.request.Request(
        f"{url}/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    data_dir = Path("data/raw")
    test_df = load_test(data_dir, args.subset)
    true_rul = load_rul(data_dir, args.subset)

    # Each engine gets all its rows sent as one batch.
    # The API returns one prediction per engine (the final window).
    records = test_df.to_dict("records")
    print(f"Sending {len(records)} rows ({test_df['engine_id'].nunique()} engines) to {args.url} ...")

    response = post_predict(args.url, records)
    predictions = response["predictions"]
    metadata = response["metadata"]

    if metadata["warnings"]:
        print(f"\nWarnings ({len(metadata['warnings'])}):")
        for w in metadata["warnings"]:
            print(f"  {w}")

    # Build results table: one row per engine, prediction vs true RUL
    pred_by_engine = {row["engine_id"]: row["prediction"] for row in predictions}
    rows = []
    for engine_id, true in enumerate(true_rul, start=1):
        pred = pred_by_engine.get(engine_id)
        if pred is not None:
            rows.append({"engine_id": engine_id, "true_rul": true, "predicted_rul": round(pred, 1), "error": round(pred - true, 1)})

    results = pd.DataFrame(rows)
    rmse = math.sqrt((results["error"] ** 2).mean())
    mae = results["error"].abs().mean()

    print(f"\n{'engine_id':>10} {'true_rul':>10} {'predicted_rul':>14} {'error':>8}")
    print("-" * 46)
    for _, row in results.iterrows():
        print(f"{int(row['engine_id']):>10} {int(row['true_rul']):>10} {row['predicted_rul']:>14} {row['error']:>+8.1f}")

    print("-" * 46)
    print(f"\nEngines predicted : {len(results)} / {test_df['engine_id'].nunique()}")
    print(f"RMSE              : {rmse:.2f}")
    print(f"MAE               : {mae:.2f}")


if __name__ == "__main__":
    main()
