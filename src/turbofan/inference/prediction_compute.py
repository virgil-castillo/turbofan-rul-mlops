"""Pure inference compute for Ridge and sequence registry models."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from turbofan.models.sequence_models import SequenceRULRegressor, build_sequence_model
from turbofan.preprocessing.normalization import OperatingModeNormalizer
from turbofan.sequences.windowing import build_final_windows

DEFAULT_MAX_RUL: int = 125
"""Default maximum-RUL cap for Ridge predictions."""


def ridge_engine_predictions(
    pipeline: object,
    frame: pd.DataFrame,
    *,
    max_rul: int = DEFAULT_MAX_RUL,
) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    """Score validated rows with a Ridge pipeline and select per-engine outputs.

    Scores every row, clips predictions to ``[0, max_rul]``, and keeps the
    last-cycle prediction per engine.

    Args:
        pipeline: Fitted sklearn-compatible pipeline exposing ``predict``.
        frame: Validated canonical rows already sorted by engine and cycle.
        max_rul: Maximum-RUL ceiling applied to predictions.

    Returns:
        Tuple of per-engine metadata rows and aligned clipped predictions.

    Raises:
        ValueError: If the pipeline returns a mismatched prediction count.
    """
    raw_predictions = pipeline.predict(frame)  # type: ignore[attr-defined]
    predictions = _clip_predictions(raw_predictions, max_value=float(max_rul))
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
    """Run final-window inference for any supported sequence architecture.

    Reads the architecture from the payload's ``sequence_config``, rebuilds the
    matching recurrent module, applies the fitted feature pipeline when present
    or the legacy normalizer-only payload otherwise, builds one final window per
    engine, runs the forward pass, rescales by ``max_rul``, and clips to be
    non-negative.

    Args:
        payload: Sequence checkpoint payload.
        frame: Validated canonical rows, each engine at least ``window_size``
            cycles long.

    Returns:
        Tuple of final-window metadata rows and aligned non-negative
        predictions.

    Raises:
        ValueError: If the payload carries invalid values or names an
            unsupported architecture.
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
    """Backward-compatible alias for sequence final-window predictions.

    Args:
        payload: Sequence checkpoint payload.
        frame: Validated canonical rows ready for windowing.

    Returns:
        Tuple of final-window metadata rows and aligned non-negative
        predictions.

    Raises:
        ValueError: If the checkpoint payload is invalid.
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
    """Normalize, window, forward, rescale, and clip already-loaded inputs."""
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
    """Apply the fitted feature pipeline before sequence window inference."""
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


def _clip_predictions(
    values: object,
    *,
    max_value: float | None = None,
) -> npt.NDArray[np.float64]:
    """Clip predictions to be non-negative, with an optional upper cap."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.clip(array, a_min=0.0, a_max=max_value)


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
    """Reconstruct an operating-mode normalizer from a sequence checkpoint."""
    norm_payload = payload["normalizer_payload"]
    if not isinstance(norm_payload, Mapping):
        raise ValueError(
            "sequence checkpoint field 'normalizer_payload' must be a mapping."
        )
    return OperatingModeNormalizer.from_payload(dict(norm_payload))
