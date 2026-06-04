"""Configuration schema for the turbofan package."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, cast

import yaml
from pydantic import BaseModel, Field

PositiveWindow = Annotated[int, Field(gt=0)]


class DataConfig(BaseModel):
    """Configuration for the data layer.

    Args:
        raw_dir: Path to raw data directory.
        processed_dir: Path to processed data directory.
        interim_dir: Path to interim data directory.
        fd_subset: C-MAPSS fault dataset subset identifier.
        max_rul: Maximum RUL cap for piecewise-linear labels.
        test_size: Fraction of training engines held out for validation.
        random_seed: Seed for all random operations.
    """

    raw_dir: Path
    processed_dir: Path
    interim_dir: Path
    fd_subset: Literal["FD001", "FD002", "FD003", "FD004"] = "FD001"
    max_rul: int = Field(default=125, gt=0)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_seed: int = 42


FeatureSetName = Literal[
    "raw",
    "rolling_mean",
    "rolling_stats",
    "raw_plus_rolling_mean",
    "raw_plus_rolling_stats",
    "lag",
    "raw_plus_lag",
    "rolling_std",
    "rolling_slope",
    "rolling_delta",
]


class ModelFeatureConfig(BaseModel):
    """Per-model feature-engineering overrides.

    Any field left as ``None`` inherits the shared ``FeatureConfig`` value when
    resolved via :meth:`FeatureConfig.for_model`.

    Args:
        feature_set: Feature family for this model, or ``None`` to inherit.
        windows: Rolling window sizes for this model, or ``None`` to inherit.
        lag_steps: Lag offsets for this model, or ``None`` to inherit.
    """

    feature_set: FeatureSetName | None = None
    windows: list[PositiveWindow] | None = None
    lag_steps: list[PositiveWindow] | None = None


class ResolvedFeatureConfig(BaseModel):
    """Fully resolved feature settings for a single model.

    Args:
        feature_set: Selected feature family.
        windows: Rolling window sizes.
        lag_steps: Lag offsets.
    """

    feature_set: FeatureSetName
    windows: list[PositiveWindow]
    lag_steps: list[PositiveWindow]


class FeatureConfig(BaseModel):
    """Configuration for feature engineering.

    The top-level ``feature_set`` / ``windows`` / ``lag_steps`` are the shared
    defaults. Optional ``ridge`` and ``gru`` blocks override them per model;
    resolve the effective settings for a model with :meth:`for_model`.

    Args:
        sensor_cols_to_drop: Sensor column names to remove before modeling.
            Determined from EDA; applied without recomputation at fit time.
        n_modes: Number of operating-mode clusters for normalization.
        feature_set: Shared engineered feature family (per-model fallback).
        windows: Shared rolling window sizes. Used by rolling feature sets.
        lag_steps: Shared lag offsets. Used by the lag feature set.
        ridge: Optional Ridge-specific feature overrides.
        gru: Optional GRU-specific feature overrides.
        lstm: Optional LSTM-specific feature overrides.
    """

    sensor_cols_to_drop: list[str] = Field(default_factory=list)
    n_modes: int = Field(default=1, gt=0)
    feature_set: FeatureSetName = "raw"
    windows: list[PositiveWindow] = Field(default_factory=lambda: [10])
    lag_steps: list[PositiveWindow] = Field(default_factory=lambda: [1])
    ridge: ModelFeatureConfig | None = None
    gru: ModelFeatureConfig | None = None
    lstm: ModelFeatureConfig | None = None

    def for_model(
        self, model: Literal["ridge", "gru", "lstm"]
    ) -> ResolvedFeatureConfig:
        """Resolve effective feature settings for a model.

        Applies the model-specific override block when present, falling back to
        the shared values for any field the override leaves unset.

        Args:
            model: Model whose feature settings to resolve.

        Returns:
            Fully resolved feature settings for the model.
        """
        overrides: dict[str, ModelFeatureConfig | None] = {
            "ridge": self.ridge,
            "gru": self.gru,
            "lstm": self.lstm,
        }
        override = overrides[model]
        feature_set = self.feature_set
        windows = self.windows
        lag_steps = self.lag_steps
        if override is not None:
            if override.feature_set is not None:
                feature_set = override.feature_set
            if override.windows is not None:
                windows = override.windows
            if override.lag_steps is not None:
                lag_steps = override.lag_steps
        return ResolvedFeatureConfig(
            feature_set=feature_set, windows=windows, lag_steps=lag_steps
        )


class ModelConfig(BaseModel):
    """Configuration for baseline model training.

    Args:
        name: Baseline model identifier.
        alpha: Ridge regularization strength.
        artifact_dir: Directory for local run artifacts.
    """

    name: Literal["ridge"] = "ridge"
    alpha: float = Field(default=100.0, gt=0.0)
    artifact_dir: Path = Path("artifacts/models")


class SequenceConfig(BaseModel):
    """Configuration for sequence model training (GRU or LSTM).

    GRU and LSTM share the same hyperparameter surface; the recurrent layer is
    selected by ``architecture`` and every other field applies unchanged to
    both.

    Args:
        architecture: Sequence model architecture identifier (``gru`` or
            ``lstm``).
        window_size: Number of cycles per sequence window.
        batch_size: Training batch size.
        hidden_size: Recurrent hidden state width.
        num_layers: Number of stacked recurrent layers.
        dropout: Dropout probability between recurrent layers.
        learning_rate: Adam optimizer learning rate.
        weight_decay: Adam L2 weight-decay strength. Defaults to ``0.0`` (no
            regularization), preserving prior behaviour; raise it to penalize
            large weights and curb overfitting. Shared by GRU and LSTM.
        epochs: Maximum training epochs.
        patience: Early-stopping patience in epochs.
        device: Requested torch device.
        artifact_dir: Directory for local sequence run artifacts.
    """

    architecture: Literal["gru", "lstm"] = "gru"
    window_size: int = Field(default=45, gt=0)
    batch_size: int = Field(default=64, gt=0)
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=1, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    epochs: int = Field(default=50, gt=0)
    patience: int = Field(default=8, gt=0)
    device: Literal["cpu", "cuda"] = "cpu"
    artifact_dir: Path = Path("artifacts/models")


class InferenceConfig(BaseModel):
    """Configuration for local inference serving.

    Args:
        artifact_path: Deprecated. The serving API now resolves the production
            model by name from the MLflow registry (``TURBOFAN_MODEL_NAME`` or
            the ``--model`` flag), so this field is retained only for backward
            config compatibility and is not consulted by ``create_app``.
        host: Host interface for the API server.
        port: Port for the API server.
        allow_partial: Whether serving clients may skip invalid records.
    """

    artifact_path: Path | None = None
    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, lt=65536)
    allow_partial: bool = False


class ProjectConfig(BaseModel):
    """Top-level project configuration.

    Args:
        project_name: Human-readable project name.
        data: Data layer configuration.
        features: Feature engineering configuration.
        model: Baseline model training configuration.
        sequence: Sequence model (GRU/LSTM) training configuration.
        inference: Local inference serving configuration.
    """

    project_name: str
    data: DataConfig
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sequence: SequenceConfig = Field(default_factory=SequenceConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


def _deep_merge(
    base: dict[str, object], override: dict[str, object]
) -> dict[str, object]:
    """Recursively merge override into base, returning a new dict.

    Nested dicts are merged key-by-key; all other values are replaced.

    Args:
        base: Base dictionary.
        override: Dictionary whose values take precedence.

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(
                cast(dict[str, object], result[key]),
                cast(dict[str, object], value),
            )
        else:
            result[key] = value
    return result


def _load_raw_config(path: Path) -> dict[str, object]:
    """Load a YAML config and recursively resolve its ``_base_`` chain.

    Each ``_base_`` reference is resolved relative to the file that declares it,
    so a chain like ``fd001_lstm.yaml`` → ``fd001.yaml`` → ``default.yaml`` is
    fully composed: every base in the chain is loaded and the more specific file
    is deep-merged on top. The ``_base_`` key is consumed at each level and does
    not appear in the returned mapping.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        The fully merged raw config mapping.

    Raises:
        FileNotFoundError: If a config file in the chain does not exist.
        yaml.YAMLError: If a file in the chain is not valid YAML.
    """
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Failed to parse config file {path}: {exc}") from exc

    if "_base_" in raw:
        base_path = (path.parent / raw.pop("_base_")).resolve()
        base = _load_raw_config(base_path)
        raw = _deep_merge(base, raw)
    return cast(dict[str, object], raw)


def load_config(path: Path) -> ProjectConfig:
    """Load and validate project configuration from a YAML file.

    If the file contains a ``_base_`` key, the referenced file is loaded first
    and the current file is deep-merged on top, allowing subset configs to
    override only the fields that differ from the base. ``_base_`` references are
    resolved recursively, so a config may extend another config that itself
    extends a shared default.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated ProjectConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If the config structure is invalid.
    """
    raw = _load_raw_config(path)
    return ProjectConfig.model_validate(raw)
