"""Runtime predictors for local turbofan inference artifacts."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from turbofan.inference.manifest import ModelMetadata, load_model_metadata
from turbofan.inference.schemas import (
    PredictionMetadata,
    PredictionResult,
    PredictionRow,
    RawRecords,
    SchemaValidationError,
    validate_raw_records,
)
from turbofan.models.gru import GRURULRegressor
from turbofan.preprocessing.normalization import OperatingModeNormalizer
from turbofan.sequences.windowing import build_final_windows


def ridge_engine_predictions(
    pipeline: object,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    """Score validated rows with a Ridge pipeline and select per-engine outputs.

    Scores every row, clips predictions to be non-negative, and keeps the
    last-cycle prediction per engine (the ``engine`` prediction scope). This is
    the pure compute shared by :class:`RidgePredictor` and the MLflow Ridge
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


def gru_final_window_predictions(
    payload: Mapping[str, object],
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    """Run the GRU final-window inference path on validated rows.

    Reconstructs the GRU model and normalizer from a checkpoint payload, then
    normalizes, builds one final window per engine, runs the forward pass,
    rescales by ``max_rul``, and clips to be non-negative (the ``final_window``
    prediction scope). This is the pure compute shared by :class:`GRUPredictor`
    and the MLflow GRU pyfunc wrapper.

    Args:
        payload: GRU checkpoint payload with ``model_state_dict``,
            ``feature_cols``, ``sequence_config`` (``window_size``,
            ``hidden_size``, ``num_layers``, ``dropout``), ``normalizer_payload``,
            and ``max_rul``.
        frame: Validated canonical rows (``engine_id``, ``cycle``, features),
            each engine at least ``window_size`` cycles long.

    Returns:
        Tuple of the final-window metadata rows (``engine_id``, ``cycle``) and
        the aligned non-negative predictions.

    Raises:
        ValueError: If the checkpoint payload is missing required fields or
            carries invalid values.
    """
    feature_cols = _string_sequence(payload, "feature_cols")
    sequence_config = _mapping(payload, "sequence_config")
    window_size = _positive_int(sequence_config, "window_size")
    model = GRURULRegressor(
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
    normalizer = _normalizer_from_payload(payload, feature_cols)
    max_rul = _positive_int(payload, "max_rul")
    return _gru_window_inference(
        model=model,
        normalizer=normalizer,
        feature_cols=feature_cols,
        window_size=window_size,
        max_rul=max_rul,
        frame=frame,
    )


def _gru_window_inference(
    *,
    model: GRURULRegressor,
    normalizer: OperatingModeNormalizer,
    feature_cols: Sequence[str],
    window_size: int,
    max_rul: int,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    """Normalize, window, forward, rescale, and clip already-loaded GRU inputs.

    Args:
        model: Loaded, evaluation-mode GRU model.
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


class Predictor(Protocol):
    """Common runtime interface for loaded inference predictors."""

    @property
    def metadata(self) -> ModelMetadata:
        """Return metadata for the loaded artifact.

        Returns:
            Model metadata loaded from the artifact manifest.
        """
        ...

    def predict(
        self,
        records: RawRecords,
        *,
        allow_partial: bool = False,
    ) -> PredictionResult:
        """Predict remaining useful life for validated records.

        Args:
            records: Raw canonical inference records.
            allow_partial: Whether invalid or short inputs may be skipped.

        Returns:
            Prediction response containing rows and metadata.
        """
        ...


class RidgePredictor:
    """Predict per-engine RUL using a fitted sklearn-compatible pipeline.

    Scores all input rows and returns the last-cycle prediction per engine.

    Args:
        metadata: Model metadata pointing at a joblib artifact.

    Raises:
        ValueError: If the metadata is not for a Ridge engine predictor.
    """

    def __init__(self, metadata: ModelMetadata) -> None:
        """Load the fitted Ridge pipeline.

        Args:
            metadata: Model metadata pointing at a joblib artifact.

        Raises:
            ValueError: If the metadata is not for a Ridge engine predictor.
        """
        if metadata.model_type != "ridge" or metadata.prediction_scope != "engine":
            raise ValueError("RidgePredictor requires ridge engine metadata.")
        self._metadata = metadata
        self._pipeline = joblib.load(metadata.model_path)

    @property
    def metadata(self) -> ModelMetadata:
        """Return metadata for the loaded artifact.

        Returns:
            Model metadata loaded from the artifact manifest.
        """
        return self._metadata

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
        rows, predictions = ridge_engine_predictions(
            self._pipeline, validation.records
        )
        prediction_rows = _prediction_rows(
            metadata=self._metadata,
            rows=rows,
            predictions=predictions,
        )
        return _prediction_result(
            metadata=self._metadata,
            predictions=prediction_rows,
            input_rows=input_rows,
            warnings=validation.warnings,
        )


class GRUPredictor:
    """Predict final-window RUL values using a serialized GRU checkpoint.

    Args:
        metadata: Model metadata pointing at a torch checkpoint.

    Raises:
        ValueError: If the metadata or checkpoint payload is invalid.
    """

    def __init__(self, metadata: ModelMetadata) -> None:
        """Load the GRU model and sequence preprocessing metadata.

        Args:
            metadata: Model metadata pointing at a torch checkpoint.

        Raises:
            ValueError: If the metadata or checkpoint payload is invalid.
        """
        if metadata.model_type != "gru" or metadata.prediction_scope != "final_window":
            raise ValueError("GRUPredictor requires gru final_window metadata.")
        self._metadata = metadata
        payload = _load_torch_payload(metadata.model_path)
        self._feature_cols = _string_sequence(payload, "feature_cols")
        sequence_config = _mapping(payload, "sequence_config")
        self._window_size = _positive_int(sequence_config, "window_size")
        self._model = GRURULRegressor(
            input_size=len(self._feature_cols),
            hidden_size=_positive_int(sequence_config, "hidden_size"),
            num_layers=_positive_int(sequence_config, "num_layers"),
            dropout=_non_negative_float(sequence_config, "dropout"),
        )
        self._model.load_state_dict(
            cast(Mapping[str, torch.Tensor], payload["model_state_dict"])
        )
        self._model.to("cpu")
        self._model.eval()
        self._normalizer = _normalizer_from_payload(payload, self._feature_cols)
        self._max_rul = _positive_int(payload, "max_rul")

    @property
    def metadata(self) -> ModelMetadata:
        """Return metadata for the loaded artifact.

        Returns:
            Model metadata loaded from the artifact manifest.
        """
        return self._metadata

    def predict(
        self,
        records: RawRecords,
        *,
        allow_partial: bool = False,
    ) -> PredictionResult:
        """Predict one non-negative RUL value per eligible engine final window.

        Args:
            records: Raw canonical inference records.
            allow_partial: Whether short engines and row validation errors may be
                skipped with warnings.

        Returns:
            Final-window prediction response.

        Raises:
            SchemaValidationError: If strict mode receives a short engine or no
                eligible final windows remain.
        """
        input_rows = _input_row_count(records)
        validation = validate_raw_records(records, partial=allow_partial)
        warning_messages = list(validation.warnings)
        frame = self._filter_short_engines(
            validation.records,
            allow_partial=allow_partial,
            warnings=warning_messages,
        )
        rows, predictions = _gru_window_inference(
            model=self._model,
            normalizer=self._normalizer,
            feature_cols=self._feature_cols,
            window_size=self._window_size,
            max_rul=self._max_rul,
            frame=frame,
        )
        prediction_rows = _prediction_rows(
            metadata=self._metadata,
            rows=rows,
            predictions=predictions,
        )
        return _prediction_result(
            metadata=self._metadata,
            predictions=prediction_rows,
            input_rows=input_rows,
            warnings=warning_messages,
        )

    def _filter_short_engines(
        self,
        frame: pd.DataFrame,
        *,
        allow_partial: bool,
        warnings: list[str],
    ) -> pd.DataFrame:
        short_engines: list[int] = []
        for engine_id, group in frame.groupby("engine_id", sort=True):
            if len(group) < self._window_size:
                short_engines.append(int(cast(int, engine_id)))
        if not short_engines:
            return frame
        if not allow_partial:
            joined = ", ".join(str(engine_id) for engine_id in short_engines)
            raise SchemaValidationError(
                "Engine(s) shorter than window_size "
                f"{self._window_size}: {joined}."
            )
        for engine_id in short_engines:
            warnings.append(
                f"Engine {engine_id}: shorter than window_size "
                f"{self._window_size}; left-zero-padded."
            )
        # Return the frame unchanged — downstream windowing will pad.
        return frame


def load_predictor(artifact_path: Path) -> Predictor:
    """Load a predictor from a manifest file or compatibility artifact directory.

    Args:
        artifact_path: Path accepted by ``load_model_metadata``.

    Returns:
        Loaded predictor for the artifact model type.

    Raises:
        ValueError: If the manifest model type is unsupported.
    """
    metadata = load_model_metadata(artifact_path)
    if metadata.model_type == "ridge":
        return RidgePredictor(metadata)
    if metadata.model_type == "gru":
        return GRUPredictor(metadata)
    raise ValueError(f"Unsupported model type: {metadata.model_type}")


def _input_row_count(records: RawRecords) -> int:
    if isinstance(records, pd.DataFrame):
        return len(records)
    return len(records)


def _clip_predictions(values: object) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.clip(array, a_min=0.0, a_max=None)


def _prediction_rows(
    *,
    metadata: ModelMetadata,
    rows: pd.DataFrame,
    predictions: npt.NDArray[np.float64],
) -> list[PredictionRow]:
    if len(predictions) != len(rows):
        raise ValueError("Prediction count does not match metadata row count.")
    predicted_at = datetime.now(UTC)
    result: list[PredictionRow] = []
    row_records = cast(
        list[dict[str, int]],
        rows.loc[:, ["engine_id", "cycle"]].to_dict("records"),
    )
    for position, row in enumerate(row_records):
        result.append(
            PredictionRow(
                engine_id=int(row["engine_id"]),
                cycle=int(row["cycle"]),
                prediction=float(predictions[position]),
                model_type=metadata.model_type,
                artifact_id=metadata.artifact_id,
                prediction_scope=metadata.prediction_scope,
                predicted_at=predicted_at,
            )
        )
    return result


def _prediction_result(
    *,
    metadata: ModelMetadata,
    predictions: list[PredictionRow],
    input_rows: int,
    warnings: list[str],
) -> PredictionResult:
    return PredictionResult(
        predictions=predictions,
        metadata=PredictionMetadata(
            model_type=metadata.model_type,
            artifact_id=metadata.artifact_id,
            prediction_scope=metadata.prediction_scope,
            input_rows=input_rows,
            prediction_rows=len(predictions),
            warnings=warnings,
        ),
    )


def _load_torch_payload(path: Path) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("GRU checkpoint payload must be a mapping.")
    if "normalizer_means" in payload or "normalizer_stds" in payload:
        raise ValueError(
            "GRU checkpoint uses a legacy flat-stat normalizer format. "
            "Retrain the model with an operating-mode normalizer payload."
        )
    for key in [
        "model_state_dict",
        "sequence_config",
        "feature_cols",
        "normalizer_type",
        "normalizer_payload",
        "max_rul",
    ]:
        if key not in payload:
            raise ValueError(f"GRU checkpoint payload missing {key!r}.")
    return cast(Mapping[str, object], payload)


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"GRU checkpoint field {key!r} must be a mapping.")
    return cast(Mapping[str, object], value)


def _string_sequence(payload: Mapping[str, object], key: str) -> list[str]:
    value = payload[key]
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"GRU checkpoint field {key!r} must be a string sequence.")
    result = list(value)
    if not result or not all(isinstance(item, str) for item in result):
        raise ValueError(f"GRU checkpoint field {key!r} must be a string sequence.")
    return cast(list[str], result)


def _positive_int(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"GRU checkpoint field {key!r} must be a positive integer.")
    return value


def _non_negative_float(payload: Mapping[str, object], key: str) -> float:
    value = payload[key]
    if not isinstance(value, int | float) or isinstance(value, bool) or value < 0.0:
        raise ValueError(f"GRU checkpoint field {key!r} must be non-negative.")
    return float(value)


def _normalizer_from_payload(
    payload: Mapping[str, object],
    feature_cols: Sequence[str],  # noqa: ARG001
) -> OperatingModeNormalizer:
    """Reconstruct an ``OperatingModeNormalizer`` from a GRU checkpoint payload.

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
            "GRU checkpoint field 'normalizer_payload' must be a mapping."
        )
    return OperatingModeNormalizer.from_payload(dict(norm_payload))
