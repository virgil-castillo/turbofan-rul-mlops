"""Configuration schema for the turbofan package."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


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


class FeatureConfig(BaseModel):
    """Configuration for feature engineering.

    Args:
        sensor_std_threshold: Maximum training standard deviation at which
            sensor columns are dropped.
        sensor_keep: Sensor columns to force-keep even when low-variance.
    """

    sensor_std_threshold: float = Field(default=0.0, ge=0.0)
    sensor_keep: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """Configuration for baseline model training.

    Args:
        name: Baseline model identifier.
        alpha: Ridge regularization strength.
        artifact_dir: Directory for local run artifacts.
    """

    name: Literal["ridge"] = "ridge"
    alpha: float = Field(default=1.0, gt=0.0)
    artifact_dir: Path = Path("artifacts/models")


class SequenceConfig(BaseModel):
    """Configuration for GRU sequence model training.

    Args:
        architecture: Sequence model architecture identifier.
        window_size: Number of cycles per sequence window.
        batch_size: Training batch size.
        hidden_size: GRU hidden state width.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability between GRU layers.
        learning_rate: Adam optimizer learning rate.
        epochs: Maximum training epochs.
        patience: Early-stopping patience in epochs.
        device: Requested torch device.
        artifact_dir: Directory for local sequence run artifacts.
    """

    architecture: Literal["gru"] = "gru"
    window_size: int = Field(default=30, gt=0)
    batch_size: int = Field(default=64, gt=0)
    hidden_size: int = Field(default=64, gt=0)
    num_layers: int = Field(default=1, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    epochs: int = Field(default=50, gt=0)
    patience: int = Field(default=8, gt=0)
    device: Literal["cpu", "cuda"] = "cpu"
    artifact_dir: Path = Path("artifacts/models")


class InferenceConfig(BaseModel):
    """Configuration for local inference serving.

    Args:
        artifact_path: Optional model artifact manifest or run directory.
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
        sequence: GRU sequence model training configuration.
        inference: Local inference serving configuration.
    """

    project_name: str
    data: DataConfig
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sequence: SequenceConfig = Field(default_factory=SequenceConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


def load_config(path: Path) -> ProjectConfig:
    """Load and validate project configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated ProjectConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If the config structure is invalid.
    """
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Failed to parse config file {path}: {exc}") from exc
    return ProjectConfig.model_validate(raw)
