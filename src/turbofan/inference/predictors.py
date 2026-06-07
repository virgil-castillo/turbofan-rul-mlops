"""Shared inference compute and a registry-backed pyfunc predictor adapter.

The pure compute helpers (:func:`ridge_engine_predictions`,
:func:`sequence_final_window_predictions`) are reused by the MLflow pyfunc
wrappers in
:mod:`turbofan.registry`. The :class:`PyfuncPredictor` adapter wraps a loaded
pyfunc model so batch prediction and the serving API can consume the registry
model through the ``metadata`` + ``predict`` interface they already expect.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from turbofan.inference.schemas import (
    CANONICAL_COLUMNS,
    ModelType,
    PredictionMetadata,
    PredictionResult,
    PredictionRow,
    PredictionScope,
    RawRecords,
    validate_raw_records,
)
from turbofan.models.sequence_models import SequenceRULRegressor, build_sequence_model
from turbofan.preprocessing.normalization import OperatingModeNormalizer
from turbofan.sequences.windowing import build_final_windows

#: Inference prediction scope for each supported model family. Both recurrent
#: families (GRU and LSTM) share the final-window scope; Ridge is engine-scoped.
_MODEL_SCOPES: dict[ModelType, PredictionScope] = {
    "ridge": "engine",
    "gru": "final_window",
    "lstm": "final_window",
}


def ridge_engine_predictions(
    pipeline: object,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    """Score validated rows with a Ridge pipeline and select per-engine outputs.

    Scores every row, clips predictions to be non-negative, and keeps the
    last-cycle prediction per engine (the ``engine`` prediction scope). This is
    the pure compute shared by the in-process Ridge path and the MLflow Ridge
    pyfunc wrapper.

    Args:
        pipeline: Fitted sklearn-compatible pipeline exposing ``predict``.
        frame: Validated canonical rows (``engine_id``, ``cycle``, features),
            already sorted by engine and cycle.

    Returns:
        Tuple of the per-engine metadata rows (``engine_id``, ``cycle``, sorted
        by ``engine_id``) and the aligned non-negative predictions.

    Raises:
        ValueError: If the pipeline returns a mismatched number of predictions.
    """
    raw_predictions = pipeline.predict(frame)  # type: ignore[attr-defined]
    predictions = _clip_predictions(raw_predictions)
    if len(predictions) != len(frame):
        raise ValueError("Ridge pipeline returned an unexpected prediction count.")
    scored = frame.copy()
    scored["_prediction"] = predictions
    last_cycle_idx = scored.groupby("engine_id")["cycle"].idxmax()
    last_rows = scored.loc[last_cycle_idx].sort_values("engine_id")
    metadata = last_rows[["engine_id", "cycle"]].reset_index(drop=True)
    return metadata, last_rows["_prediction"].to_numpy(dtype=np.float64)


def sequence_final_window_predictions(
    payload: Mapping[str, object],
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    """Run the final-window inference path for any sequence architecture.

    Reads the architecture from the payload's ``sequence_config``, rebuilds the
    matching recurrent module via the registry, applies the fitted feature
    pipeline when present (or the legacy normalizer-only payload otherwise),
    builds one final window per engine, runs the forward pass, rescales by
    ``max_rul``, and clips to be non-negative (the ``final_window`` prediction
    scope). The path is identical across RNNs, so this single function serves
    both the in-process sequence path and the MLflow sequence pyfunc wrapper.

    Args:
        payload: Sequence checkpoint payload with ``model_state_dict``,
            ``feature_cols``, ``sequence_config`` (``architecture``,
            ``window_size``, ``hidden_size``, ``num_layers``, ``dropout``),
            optional ``feature_pipeline``, legacy ``normalizer_payload``, and
            ``max_rul``.
        frame: Validated canonical rows (``engine_id``, ``cycle``, features),
            each engine at least ``window_size`` cycles long.

    Returns:
        Tuple of the final-window metadata rows (``engine_id``, ``cycle``) and
        the aligned non-negative predictions.

    Raises:
        ValueError: If the checkpoint payload is missing required fields,
            carries invalid values, or names an unsupported architecture.
    """
    feature_cols = _string_sequence(payload, "feature_cols")
    sequence_config = _mapping(payload, "sequence_config")
    architecture = _string(sequence_config, "architecture")
    window_size = _positive_int(sequence_config, "window_size")
    model = build_sequence_model(
        architecture,
        input_size=len(feature_cols),
        hidden_size=_positive_int(sequence_config, "hidden_size"),
        num_layers=_positive_int(sequence_config, "num_layers"),
        dropout=_non_negative_float(sequence_config, "dropout"),
    )
    model.load_state_dict(
        cast(Mapping[str, torch.Tensor], payload["model_state_dict"])
    )
    model.to("cpu")
    model.eval()
    max_rul = _positive_int(payload, "max_rul")
    feature_pipeline = payload.get("feature_pipeline")
    if feature_pipeline is not None:
        if not isinstance(feature_pipeline, Pipeline):
            raise ValueError(
                "sequence checkpoint field 'feature_pipeline' must be a fitted "
                "sklearn Pipeline."
            )
        return _sequence_pipeline_window_inference(
            model=model,
            feature_pipeline=feature_pipeline,
            feature_cols=feature_cols,
            window_size=window_size,
            max_rul=max_rul,
            frame=frame,
        )
    normalizer = _normalizer_from_payload(payload, feature_cols)
    return _sequence_window_inference(
        model=model,
        normalizer=normalizer,
        feature_cols=feature_cols,
        window_size=window_size,
        max_rul=max_rul,
        frame=frame,
    )


def gru_final_window_predictions(
    payload: Mapping[str, object],
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    """Backward-compatible alias for :func:`sequence_final_window_predictions`.

    Retained so external callers depending on the GRU-specific name keep
    working; delegates unchanged to the generalized sequence compute, which
    reads the architecture from the payload.

    Args:
        payload: Sequence checkpoint payload (see
            :func:`sequence_final_window_predictions`).
        frame: Validated canonical rows ready for windowing.

    Returns:
        Tuple of the final-window metadata rows (``engine_id``, ``cycle``) and
        the aligned non-negative predictions.

    Raises:
        ValueError: If the checkpoint payload is missing required fields or
            carries invalid values.
    """
    return sequence_final_window_predictions(payload, frame)


def _sequence_window_inference(
    *,
    model: SequenceRULRegressor,
    normalizer: OperatingModeNormalizer,
    feature_cols: Sequence[str],
    window_size: int,
    max_rul: int,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    """Normalize, window, forward, rescale, and clip already-loaded inputs.

    Args:
        model: Loaded, evaluation-mode sequence model.
        normalizer: Fitted operating-mode normalizer.
        feature_cols: Ordered feature column names.
        window_size: Sequence window size.
        max_rul: Maximum-RUL cap used to rescale raw model output.
        frame: Validated canonical rows ready for windowing.

    Returns:
        Tuple of the final-window metadata rows (``engine_id``, ``cycle``) and
        the aligned non-negative predictions.
    """
    normalized = normalizer.transform(frame)
    windows = build_final_windows(
        normalized,
        feature_cols,
        window_size,
        target_col=None,
    )
    with torch.no_grad():
        tensor = torch.as_tensor(windows.X, dtype=torch.float32, device="cpu")
        lengths_tensor = torch.as_tensor(
            windows.lengths, dtype=torch.int64, device="cpu"
        )
        raw_predictions = model(tensor, lengths_tensor).detach().cpu().numpy()
    rescaled = np.asarray(raw_predictions, dtype=np.float64).reshape(-1) * max_rul
    predictions = _clip_predictions(rescaled)
    metadata = windows.metadata[["engine_id", "cycle"]].reset_index(drop=True)
    return metadata, predictions


def _sequence_pipeline_window_inference(
    *,
    model: SequenceRULRegressor,
    feature_pipeline: Pipeline,
    feature_cols: Sequence[str],
    window_size: int,
    max_rul: int,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    """Apply the fitted feature pipeline before sequence window inference.

    Args:
        model: Loaded, evaluation-mode sequence model.
        feature_pipeline: Fitted feature engineering pipeline from training.
        feature_cols: Ordered feature column names.
        window_size: Sequence window size.
        max_rul: Maximum-RUL cap used to rescale raw model output.
        frame: Validated canonical raw rows ready for transformation.

    Returns:
        Tuple of the final-window metadata rows (``engine_id``, ``cycle``) and
        the aligned non-negative predictions.

    Raises:
        ValueError: If the fitted pipeline does not return a DataFrame.
    """
    features = feature_pipeline.transform(frame)
    if not isinstance(features, pd.DataFrame):
        raise ValueError("sequence feature pipeline must return a DataFrame.")
    engineered = pd.concat(
        [
            frame[["engine_id", "cycle"]].reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )
    windows = build_final_windows(
        engineered,
        feature_cols,
        window_size,
        target_col=None,
    )
    with torch.no_grad():
        tensor = torch.as_tensor(windows.X, dtype=torch.float32, device="cpu")
        lengths_tensor = torch.as_tensor(
            windows.lengths, dtype=torch.int64, device="cpu"
        )
        raw_predictions = model(tensor, lengths_tensor).detach().cpu().numpy()
    rescaled = np.asarray(raw_predictions, dtype=np.float64).reshape(-1) * max_rul
    predictions = _clip_predictions(rescaled)
    metadata = windows.metadata[["engine_id", "cycle"]].reset_index(drop=True)
    return metadata, predictions


class PyfuncPredictor:
    """Adapt a loaded registry pyfunc model to the inference predictor contract.

    Wraps an MLflow pyfunc model whose ``predict`` returns the
    ``engine_id``/``cycle``/``prediction`` output frame (see
    :mod:`turbofan.registry`) and presents the ``metadata`` property and
    ``predict`` method that batch prediction and the serving API expect.

    The pyfunc boundary validates raw records internally and raises on bad
    input, so the partial-mode row-skipping warnings produced by the retired
    in-process predictors are no longer surfaced; ``warnings`` is always empty
    and ``allow_partial`` is accepted for call-compatibility but does not change
    the strict validation performed inside the wrapper.

    Args:
        model: Loaded pyfunc model returning the prediction output frame.
        model_type: Model family identifier (``"ridge"`` or ``"gru"``).
        artifact_id: Stable identifier for the resolved registry version.
    """

    def __init__(
        self,
        model: object,
        *,
        model_type: ModelType,
        artifact_id: str,
    ) -> None:
        """Store the loaded model and derived metadata.

        Args:
            model: Loaded pyfunc model returning the prediction output frame.
            model_type: Model family identifier (``"ridge"`` or ``"gru"``).
            artifact_id: Stable identifier for the resolved registry version.
        """
        self._model = model
        self._metadata = PredictionMetadata(
            model_type=model_type,
            artifact_id=artifact_id,
            prediction_scope=_MODEL_SCOPES[model_type],
            input_rows=0,
            prediction_rows=0,
            warnings=[],
        )

    @property
    def metadata(self) -> PredictionMetadata:
        """Return descriptive metadata for the loaded registry model.

        The ``input_rows``/``prediction_rows`` fields are placeholders here; the
        per-request counts are populated by :meth:`predict`.

        Returns:
            Response-level metadata describing the resolved model.
        """
        return self._metadata

    def predict(
        self,
        records: RawRecords,
        *,
        allow_partial: bool = False,
    ) -> PredictionResult:
        """Predict remaining useful life via the loaded pyfunc model.

        Args:
            records: Raw canonical inference records.
            allow_partial: Accepted for call-compatibility; the pyfunc wrapper
                validates strictly and does not skip rows.

        Returns:
            Prediction response with per-row predictions and response metadata.
        """
        del allow_partial
        input_rows = len(records)
        frame = _records_to_frame(records)
        output = self._model.predict(frame)  # type: ignore[attr-defined]
        prediction_rows = _prediction_rows_from_output(
            output,
            model_type=self._metadata.model_type,
            artifact_id=self._metadata.artifact_id,
            prediction_scope=self._metadata.prediction_scope,
        )
        return PredictionResult(
            predictions=prediction_rows,
            metadata=PredictionMetadata(
                model_type=self._metadata.model_type,
                artifact_id=self._metadata.artifact_id,
                prediction_scope=self._metadata.prediction_scope,
                input_rows=input_rows,
                prediction_rows=len(prediction_rows),
                warnings=[],
            ),
        )


def _records_to_frame(records: RawRecords) -> pd.DataFrame:
    """Coerce raw records into a canonical-column DataFrame for the pyfunc model.

    Args:
        records: Raw records as mappings or a DataFrame.

    Returns:
        DataFrame restricted to the canonical columns the pyfunc model expects.

    Raises:
        SchemaValidationError: If the records fail canonical validation.
    """
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        frame = pd.DataFrame(list(records))
    # Validate up front so missing-column / bad-cell errors map to a clear
    # SchemaValidationError before the records cross the pyfunc boundary.
    validate_raw_records(frame)
    return frame.loc[:, [c for c in CANONICAL_COLUMNS if c in frame.columns]]


def _prediction_rows_from_output(
    output: pd.DataFrame,
    *,
    model_type: ModelType,
    artifact_id: str,
    prediction_scope: PredictionScope,
) -> list[PredictionRow]:
    """Build prediction rows from the pyfunc output frame.

    Args:
        output: Pyfunc output frame with ``engine_id``, ``cycle``, and
            ``prediction`` columns.
        model_type: Model family used for prediction.
        artifact_id: Resolved registry model identifier.
        prediction_scope: Prediction scope for the model family.

    Returns:
        One :class:`PredictionRow` per output row.
    """
    predicted_at = datetime.now(UTC)
    records = cast(
        list[dict[str, object]],
        output.loc[:, ["engine_id", "cycle", "prediction"]].to_dict("records"),
    )
    return [
        PredictionRow(
            engine_id=int(cast(int, row["engine_id"])),
            cycle=int(cast(int, row["cycle"])),
            prediction=float(cast(float, row["prediction"])),
            model_type=model_type,
            artifact_id=artifact_id,
            prediction_scope=prediction_scope,
            predicted_at=predicted_at,
        )
        for row in records
    ]


def _clip_predictions(values: object) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.clip(array, a_min=0.0, a_max=None)


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"sequence checkpoint field {key!r} must be a mapping.")
    return cast(Mapping[str, object], value)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"sequence checkpoint field {key!r} must be a non-empty string."
        )
    return value


def _string_sequence(payload: Mapping[str, object], key: str) -> list[str]:
    value = payload[key]
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(
            f"sequence checkpoint field {key!r} must be a string sequence."
        )
    result = list(value)
    if not result or not all(isinstance(item, str) for item in result):
        raise ValueError(
            f"sequence checkpoint field {key!r} must be a string sequence."
        )
    return cast(list[str], result)


def _positive_int(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"sequence checkpoint field {key!r} must be a positive integer."
        )
    return value


def _non_negative_float(payload: Mapping[str, object], key: str) -> float:
    value = payload[key]
    if not isinstance(value, int | float) or isinstance(value, bool) or value < 0.0:
        raise ValueError(f"sequence checkpoint field {key!r} must be non-negative.")
    return float(value)


def _normalizer_from_payload(
    payload: Mapping[str, object],
    feature_cols: Sequence[str],  # noqa: ARG001
) -> OperatingModeNormalizer:
    """Reconstruct an ``OperatingModeNormalizer`` from a sequence checkpoint.

    Args:
        payload: Full checkpoint payload mapping.
        feature_cols: Feature column names (unused; kept for call-site
            compatibility).

    Returns:
        Fitted ``OperatingModeNormalizer`` ready to call ``transform`` on.

    Raises:
        ValueError: If ``normalizer_payload`` is not a mapping.
    """
    norm_payload = payload["normalizer_payload"]
    if not isinstance(norm_payload, Mapping):
        raise ValueError(
            "sequence checkpoint field 'normalizer_payload' must be a mapping."
        )
    return OperatingModeNormalizer.from_payload(dict(norm_payload))
