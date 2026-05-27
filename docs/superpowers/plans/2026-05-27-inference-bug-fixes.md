# Inference Pipeline Bug Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three inference bugs — Ridge returns per-engine predictions, GRU rescales by max_rul, and the predict CLI auto-evaluates against official labels.

**Architecture:** RidgePredictor.predict groups scored rows by engine_id and keeps only the last cycle. GRUPredictor reads max_rul from the checkpoint payload and multiplies raw model output before clipping. The predict CLI looks for RUL_<subset>.txt alongside input data and computes regression metrics when label counts match predictions.

**Tech Stack:** Python, pandas, numpy, torch, joblib, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/turbofan/inference/predictors.py` | Modify | Ridge: group-by last-cycle logic. GRU: read max_rul from payload, rescale. Update `_load_torch_payload` to require `max_rul`. Update `RidgePredictor.__init__` to accept `prediction_scope="engine"`. |
| `src/turbofan/inference/manifest.py` | Modify | Add `"engine"` to `_require_prediction_scope` and `_validate_model_scope_pair`. |
| `src/turbofan/inference/schemas.py` | Modify | Add `"engine"` to `PredictionScope` literal. |
| `src/turbofan/cli/train_baseline.py` | Modify | Change manifest `prediction_scope` from `"row"` to `"engine"`. |
| `src/turbofan/cli/train_sequence_gru.py` | Modify | Add `max_rul` to `_model_payload`. |
| `src/turbofan/cli/predict.py` | Modify | Add `--data-dir` and `--subset` args, auto-evaluate against RUL labels. |
| `tests/inference/test_predictors.py` | Modify | Update Ridge tests for per-engine output and new prediction_scope. Add GRU max_rul rescaling test. Update `_gru_artifact` to include `max_rul`. |
| `tests/inference/test_predict_cli.py` | Modify | Add tests for CLI evaluation-present and evaluation-absent paths. Update existing tests for new Ridge per-engine behavior. |

---

### Task 1: Add "engine" to PredictionScope and manifest validation

**Files:**
- Modify: `src/turbofan/inference/schemas.py:15` (PredictionScope literal)
- Modify: `src/turbofan/inference/manifest.py:212-255` (scope validation)

- [ ] **Step 1: Update PredictionScope literal in schemas.py**

In `src/turbofan/inference/schemas.py`, change the `PredictionScope` type alias:

```python
# Old
PredictionScope = Literal["row", "final_window"]

# New
PredictionScope = Literal["row", "engine", "final_window"]
```

- [ ] **Step 2: Update `_require_prediction_scope` in manifest.py**

In `src/turbofan/inference/manifest.py`, add the `"engine"` case:

```python
def _require_prediction_scope(payload: dict[str, object]) -> PredictionScope:
    value = _require_string(payload, "prediction_scope")
    if value == "row":
        return "row"
    if value == "engine":
        return "engine"
    if value == "final_window":
        return "final_window"
    raise ManifestError(
        "Manifest field 'prediction_scope' must be one of: engine, final_window, row."
    )
```

- [ ] **Step 3: Update `_validate_model_scope_pair` in manifest.py**

```python
def _validate_model_scope_pair(
    model_type: ModelType,
    prediction_scope: PredictionScope,
) -> None:
    if (model_type, prediction_scope) in {
        ("ridge", "engine"),
        ("gru", "final_window"),
    }:
        return
    raise ManifestError(
        "Manifest model_type and prediction_scope are inconsistent; expected "
        "ridge with engine or gru with final_window."
    )
```

- [ ] **Step 4: Run type checker**

Run: `mypy src/turbofan/inference/schemas.py src/turbofan/inference/manifest.py`
Expected: PASS (no errors)

---

### Task 2: Update RidgePredictor to return one prediction per engine (last cycle)

**Files:**
- Modify: `src/turbofan/inference/predictors.py:59-127`
- Modify: `tests/inference/test_predictors.py`

- [ ] **Step 1: Update the failing test**

In `tests/inference/test_predictors.py`, update `_ridge_artifact` to use `prediction_scope="engine"`:

```python
def _ridge_artifact(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "ridge"
    artifact_dir.mkdir()
    joblib.dump(_NegativeRidgePipeline(), artifact_dir / "model.joblib")
    return _write_manifest(
        artifact_dir,
        model_type="ridge",
        artifact_id="ridge-test",
        prediction_scope="engine",
        model_path="model.joblib",
    )
```

Update `test_load_predictor_returns_ridge_predictions_for_each_valid_row` to test per-engine behavior. Rename it and rewrite the body to send multiple cycles per engine and assert one prediction per engine (the last cycle):

```python
def test_load_predictor_returns_ridge_prediction_per_engine_last_cycle(
    tmp_path: Path,
) -> None:
    """Ridge predictor returns one clipped prediction per engine (last cycle)."""
    from turbofan.inference.predictors import load_predictor

    predictor = load_predictor(_ridge_artifact(tmp_path))
    records = pd.DataFrame(
        [
            _record(engine_id=1, cycle=1),
            _record(engine_id=1, cycle=2),
            _record(engine_id=1, cycle=3),
            _record(engine_id=2, cycle=1),
            _record(engine_id=2, cycle=2),
        ]
    )

    result = predictor.predict(records)

    prediction_tuples = [
        (row.engine_id, row.cycle, row.prediction) for row in result.predictions
    ]
    assert prediction_tuples == [
        (1, 3, 0.0),
        (2, 2, 0.0),
    ]
    _assert_prediction_rows(
        result.predictions,
        artifact_id="ridge-test",
        model_type="ridge",
        prediction_scope="engine",
    )
    _assert_response_metadata(
        result,
        artifact_id="ridge-test",
        model_type="ridge",
        prediction_scope="engine",
        input_rows=5,
        prediction_rows=2,
        warnings=[],
    )
```

Also update the Ridge partial mode test `test_ridge_predictor_partial_mode_returns_validation_warnings` to use `prediction_scope="engine"` expectations where applicable.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/inference/test_predictors.py -v -k "ridge"`
Expected: FAIL — RidgePredictor.__init__ rejects `prediction_scope="engine"`, test expects per-engine output.

- [ ] **Step 3: Update RidgePredictor**

In `src/turbofan/inference/predictors.py`, update `RidgePredictor.__init__` to accept `"engine"` scope:

```python
def __init__(self, metadata: ModelMetadata) -> None:
    if metadata.model_type != "ridge" or metadata.prediction_scope != "engine":
        raise ValueError("RidgePredictor requires ridge engine metadata.")
    self._metadata = metadata
    self._pipeline = joblib.load(metadata.model_path)
```

Update the docstring on `RidgePredictor.predict` and add last-cycle grouping after scoring. Replace the predict method body:

```python
def predict(
    self,
    records: RawRecords,
    *,
    allow_partial: bool = False,
) -> PredictionResult:
    """Predict one non-negative RUL value per engine (last-cycle prediction).

    Args:
        records: Raw canonical inference records.
        allow_partial: Whether row-level validation errors may be skipped.

    Returns:
        Per-engine prediction response with one prediction per engine.

    Raises:
        ValueError: If the pipeline returns a mismatched number of predictions.
    """
    input_rows = _input_row_count(records)
    validation = validate_raw_records(records, partial=allow_partial)
    frame = validation.records
    raw_predictions = self._pipeline.predict(frame)
    predictions = _clip_predictions(raw_predictions)
    if len(predictions) != len(frame):
        raise ValueError("Ridge pipeline returned an unexpected prediction count.")
    frame = frame.copy()
    frame["_prediction"] = predictions
    last_cycle_idx = frame.groupby("engine_id")["cycle"].idxmax()
    last_rows = frame.loc[last_cycle_idx].sort_values("engine_id")
    prediction_rows = _prediction_rows(
        metadata=self._metadata,
        rows=last_rows[["engine_id", "cycle"]],
        predictions=last_rows["_prediction"].to_numpy(),
    )
    return _prediction_result(
        metadata=self._metadata,
        predictions=prediction_rows,
        input_rows=input_rows,
        warnings=validation.warnings,
    )
```

Also update the class docstring:

```python
class RidgePredictor:
    """Predict per-engine RUL using a fitted sklearn-compatible pipeline.

    Scores all input rows and returns the last-cycle prediction per engine.

    Args:
        metadata: Model metadata pointing at a joblib artifact.

    Raises:
        ValueError: If the metadata is not for a Ridge engine predictor.
    """
```

- [ ] **Step 4: Run Ridge tests to verify they pass**

Run: `pytest tests/inference/test_predictors.py -v -k "ridge"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/turbofan/inference/schemas.py src/turbofan/inference/manifest.py src/turbofan/inference/predictors.py tests/inference/test_predictors.py
git commit -m "fix: Ridge predictor returns one prediction per engine (last cycle)

Change RidgePredictor.predict to score all rows then select the
last-cycle prediction per engine. Add 'engine' prediction scope to the
manifest schema."
```

---

### Task 3: Update Ridge training manifest to use "engine" scope

**Files:**
- Modify: `src/turbofan/cli/train_baseline.py:73`

- [ ] **Step 1: Change prediction_scope in train_baseline.py**

```python
# Old
"prediction_scope": "row",

# New
"prediction_scope": "engine",
```

- [ ] **Step 2: Run lint and type check**

Run: `ruff check src/turbofan/cli/train_baseline.py && mypy src/turbofan/cli/train_baseline.py`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/turbofan/cli/train_baseline.py
git commit -m "fix: Ridge training manifest uses 'engine' prediction scope"
```

---

### Task 4: Fix GRU predictor to rescale by max_rul

**Files:**
- Modify: `src/turbofan/cli/train_sequence_gru.py:187-218` (_model_payload)
- Modify: `src/turbofan/inference/predictors.py:130-227,339-352` (GRUPredictor, _load_torch_payload)
- Modify: `tests/inference/test_predictors.py`

- [ ] **Step 1: Update the GRU test fixture to include max_rul**

In `tests/inference/test_predictors.py`, update `_gru_artifact` to add `max_rul` to the checkpoint and update the test that verifies GRU predictions to expect rescaled output:

```python
def _gru_artifact(tmp_path: Path, *, window_size: int = 3, max_rul: int = 125) -> Path:
    artifact_dir = tmp_path / "gru"
    artifact_dir.mkdir()
    model = GRURULRegressor(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    model.regressor.bias.data.fill_(-2.0)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sequence_config": {
                "architecture": "gru",
                "window_size": window_size,
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "feature_cols": FEATURE_COLUMNS,
            "normalizer_means": {column: 0.0 for column in FEATURE_COLUMNS},
            "normalizer_stds": {column: 1.0 for column in FEATURE_COLUMNS},
            "max_rul": max_rul,
        },
        artifact_dir / "model.pt",
    )
    return _write_manifest(
        artifact_dir,
        model_type="gru",
        artifact_id="gru-test",
        prediction_scope="final_window",
        model_path="model.pt",
    )
```

- [ ] **Step 2: Add a test for max_rul rescaling**

Add a test that uses a known max_rul and verifies the predictor output is rescaled (the model has bias=-2.0, sigmoid(-2.0) ≈ 0.119, multiplied by max_rul=125 ≈ 14.88):

```python
def test_gru_predictor_rescales_output_by_max_rul(tmp_path: Path) -> None:
    """GRU predictor multiplies raw sigmoid output by max_rul."""
    from turbofan.inference.predictors import load_predictor

    max_rul = 125
    predictor = load_predictor(_gru_artifact(tmp_path, window_size=3, max_rul=max_rul))
    records = pd.DataFrame(_records_for_engine(1, 3, feature_value=0.0))

    result = predictor.predict(records)

    assert len(result.predictions) == 1
    prediction = result.predictions[0].prediction
    # Model has zero weights + bias=-2.0 → sigmoid(-2.0) ≈ 0.119
    # Rescaled: 0.119 * 125 ≈ 14.88
    assert prediction > 10.0, f"Expected rescaled prediction > 10, got {prediction}"
    assert prediction < 20.0, f"Expected rescaled prediction < 20, got {prediction}"
```

- [ ] **Step 3: Add a test that checkpoint missing max_rul raises ValueError**

```python
def test_gru_predictor_rejects_checkpoint_without_max_rul(tmp_path: Path) -> None:
    """GRU predictor fails to load when checkpoint is missing max_rul."""
    from turbofan.inference.predictors import load_predictor

    artifact_dir = tmp_path / "gru_no_maxrul"
    artifact_dir.mkdir()
    model = GRURULRegressor(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sequence_config": {
                "architecture": "gru",
                "window_size": 3,
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "feature_cols": FEATURE_COLUMNS,
            "normalizer_means": {column: 0.0 for column in FEATURE_COLUMNS},
            "normalizer_stds": {column: 1.0 for column in FEATURE_COLUMNS},
        },
        artifact_dir / "model.pt",
    )
    _write_manifest(
        artifact_dir,
        model_type="gru",
        artifact_id="gru-no-maxrul",
        prediction_scope="final_window",
        model_path="model.pt",
    )

    with pytest.raises(ValueError, match="max_rul"):
        load_predictor(artifact_dir / "model_manifest.json")
```

- [ ] **Step 4: Run GRU tests to verify they fail**

Run: `pytest tests/inference/test_predictors.py -v -k "gru"`
Expected: FAIL — max_rul not in checkpoint, predictions not rescaled.

- [ ] **Step 5: Update `_load_torch_payload` to require max_rul**

In `src/turbofan/inference/predictors.py`, add `"max_rul"` to the required keys:

```python
def _load_torch_payload(path: Path) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("GRU checkpoint payload must be a mapping.")
    for key in [
        "model_state_dict",
        "sequence_config",
        "feature_cols",
        "normalizer_means",
        "normalizer_stds",
        "max_rul",
    ]:
        if key not in payload:
            raise ValueError(f"GRU checkpoint payload missing {key!r}.")
    return cast(Mapping[str, object], payload)
```

- [ ] **Step 6: Update GRUPredictor.__init__ to read max_rul**

In `GRUPredictor.__init__`, after loading the model, read max_rul from the payload:

```python
self._normalizer = _normalizer_from_payload(payload, self._feature_cols)
self._max_rul = _positive_int(payload, "max_rul")
```

(Note: `_positive_int` already exists and validates positive integer — `max_rul` fits.)

- [ ] **Step 7: Update GRUPredictor.predict to rescale before clipping**

In `GRUPredictor.predict`, change the line after raw model output:

```python
# Old
predictions = _clip_predictions(raw_predictions)

# New
rescaled = np.asarray(raw_predictions, dtype=np.float64).reshape(-1) * self._max_rul
predictions = _clip_predictions(rescaled)
```

- [ ] **Step 8: Update _model_payload in train_sequence_gru.py to include max_rul**

In `src/turbofan/cli/train_sequence_gru.py`, add `"max_rul"` to the checkpoint payload:

```python
def _model_payload(
    model: GRURULRegressor,
    cfg: ProjectConfig,
    feature_cols: list[str],
    normalizer: SequenceNormalizer,
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "sequence_config": cfg.sequence.model_dump(mode="json"),
        "normalizer_means": {
            str(key): float(value)
            for key, value in normalizer.means_.to_dict().items()
        },
        "normalizer_stds": {
            str(key): float(value)
            for key, value in normalizer.stds_.to_dict().items()
        },
        "fd_subset": cfg.data.fd_subset,
        "random_seed": cfg.data.random_seed,
        "max_rul": cfg.data.max_rul,
    }
```

- [ ] **Step 9: Run GRU tests to verify they pass**

Run: `pytest tests/inference/test_predictors.py -v -k "gru"`
Expected: PASS

- [ ] **Step 10: Run all predictor tests**

Run: `pytest tests/inference/test_predictors.py -v`
Expected: ALL PASS

- [ ] **Step 11: Commit**

```bash
git add src/turbofan/inference/predictors.py src/turbofan/cli/train_sequence_gru.py tests/inference/test_predictors.py
git commit -m "fix: GRU predictor rescales raw output by max_rul

Store max_rul in checkpoint payload during training and read it back
in GRUPredictor. Multiply raw sigmoid output by max_rul before
clipping to convert from normalized 0-1 range to actual RUL cycles."
```

---

### Task 5: Add auto-evaluation to the predict CLI

**Files:**
- Modify: `src/turbofan/cli/predict.py`
- Modify: `tests/inference/test_predict_cli.py`

- [ ] **Step 1: Write the test for evaluation-present path**

In `tests/inference/test_predict_cli.py`, add a helper to write a RUL labels file and a test for auto-evaluation. The Ridge artifact now produces 1 prediction per engine, so set up 2 engines with 1 cycle each:

```python
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
```

- [ ] **Step 2: Write the test for evaluation-absent path**

```python
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
```

- [ ] **Step 3: Run CLI tests to verify they fail**

Run: `pytest tests/inference/test_predict_cli.py -v -k "evaluation"`
Expected: FAIL — `--data-dir` and `--subset` are unknown arguments.

- [ ] **Step 4: Implement auto-evaluation in predict.py**

Add `--data-dir` and `--subset` arguments to `_build_parser`:

```python
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--subset", type=str, default=None)
    return parser
```

Add import at the top of `predict.py`:

```python
import numpy as np

from turbofan.models.metrics import regression_metrics
```

Add a private evaluation helper:

```python
def _try_evaluate(
    predictions: list[dict[str, object]],
    data_dir: Path | None,
    subset: str | None,
) -> dict[str, float] | None:
    """Evaluate predictions against official RUL labels if available.

    Args:
        predictions: Serialized prediction rows.
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
        [float(row["prediction"]) for row in predictions],
        dtype=np.float64,
    )
    return regression_metrics(labels, y_pred)
```

Update `main` to call the evaluation after writing outputs, and fold metrics into the metadata payload:

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        records = _read_records(args.input)
        predictor = load_predictor(args.artifact)
        result = predictor.predict(records, allow_partial=args.allow_partial)
        payload = prediction_result_to_dict(result)
        _write_predictions(args.output, payload)
        predictions_list = payload["predictions"]
        if not isinstance(predictions_list, list):
            raise ValueError("Serialized predictions must be a list.")
        evaluation = _try_evaluate(
            predictions_list,
            args.data_dir,
            args.subset,
        )
        if evaluation is not None:
            metadata_dict = payload["metadata"]
            if isinstance(metadata_dict, dict):
                metadata_dict["evaluation"] = evaluation
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
    if evaluation is not None:
        print(f"RMSE: {evaluation['rmse']:.4f}")
        print(f"MAE: {evaluation['mae']:.4f}")
        print(f"PHM08 Score: {evaluation['phm08_score']:.4f}")
    return 0
```

- [ ] **Step 5: Run CLI tests to verify they pass**

Run: `pytest tests/inference/test_predict_cli.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/turbofan/cli/predict.py tests/inference/test_predict_cli.py
git commit -m "feat: predict CLI auto-evaluates against official RUL labels

When --data-dir and --subset are provided and the RUL labels file
exists with matching count, compute RMSE, MAE, and PHM08 score.
Print metrics to stdout and include them in the metadata JSON."
```

---

### Task 6: Update existing CLI tests for new Ridge per-engine behavior

**Files:**
- Modify: `tests/inference/test_predict_cli.py`

- [ ] **Step 1: Update `_write_ridge_artifact` to use `prediction_scope="engine"`**

```python
def _write_ridge_artifact(tmp_path: Path) -> Path:
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
```

- [ ] **Step 2: Update the first CLI test for per-engine output**

In `test_predict_cli_reads_csv_and_writes_predictions_and_metadata`, the test sends 2 records (engine 2 cycle 1, engine 1 cycle 3). Each engine has only one cycle, so one prediction per engine. Update the expected metadata:

```python
metadata = json.loads(metadata_path.read_text())
assert metadata == {
    "model_type": "ridge",
    "artifact_id": "ridge-cli-test",
    "prediction_scope": "engine",
    "input_rows": 2,
    "prediction_rows": 2,
    "warnings": [],
}
```

- [ ] **Step 3: Update partial mode tests for per-engine scope**

In `test_predict_cli_reads_json_records_object_and_allows_partial_rows` and `test_predict_cli_allows_partial_csv_rows_with_one_bad_numeric_cell`, the output will still have 1 prediction (1 engine after filtering). The metadata `prediction_scope` in the metadata JSON will be `"engine"` instead of `"row"`. Verify these assertions match.

- [ ] **Step 4: Run all CLI tests**

Run: `pytest tests/inference/test_predict_cli.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tests/inference/test_predict_cli.py
git commit -m "test: update CLI tests for Ridge per-engine prediction scope"
```

---

### Task 7: Update manifest tests if needed and run full suite

**Files:**
- Modify: `tests/inference/test_manifest.py` (if assertions reference `"row"` for ridge)

- [ ] **Step 1: Check manifest tests for ridge/row assumptions**

Read `tests/inference/test_manifest.py` and update any hardcoded `prediction_scope="row"` assertions for ridge artifacts to use `"engine"`.

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: Run linting and type checking**

Run: `ruff check src/ tests/ && mypy src/turbofan`
Expected: PASS

- [ ] **Step 4: Commit any remaining fixes**

```bash
git add -u
git commit -m "fix: update manifest tests for ridge engine prediction scope"
```
