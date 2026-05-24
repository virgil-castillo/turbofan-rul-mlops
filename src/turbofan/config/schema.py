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


class ProjectConfig(BaseModel):
    """Top-level project configuration.

    Args:
        project_name: Human-readable project name.
        data: Data layer configuration.
        model: Baseline model training configuration.
    """

    project_name: str
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)


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
