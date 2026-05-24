"""Tests for turbofan.config.schema."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from turbofan.config.schema import DataConfig, load_config


def _write_config(tmp_path: Path, data: dict[str, object]) -> Path:
    """Write a config dict to a YAML file and return the path."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data))
    return path


def test_load_config_valid(tmp_path: Path) -> None:
    """Valid YAML round-trips through load_config without error."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
        },
    )
    cfg = load_config(cfg_file)
    assert cfg.project_name == "test-project"
    assert cfg.data.fd_subset == "FD001"
    assert cfg.data.max_rul == 125
    assert cfg.data.random_seed == 42


def test_path_fields_are_path_objects(tmp_path: Path) -> None:
    """DataConfig path fields are Path objects, not strings."""
    cfg = DataConfig(
        raw_dir="data/raw",  # type: ignore[arg-type]
        processed_dir="data/processed",  # type: ignore[arg-type]
        interim_dir="data/interim",  # type: ignore[arg-type]
    )
    assert isinstance(cfg.raw_dir, Path)
    assert isinstance(cfg.processed_dir, Path)
    assert isinstance(cfg.interim_dir, Path)


def test_invalid_fd_subset_raises(tmp_path: Path) -> None:
    """Invalid fd_subset value raises ValidationError."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
                "fd_subset": "FD999",
            },
        },
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_missing_project_name_raises(tmp_path: Path) -> None:
    """Missing required project_name raises ValidationError."""
    cfg_file = _write_config(
        tmp_path,
        {
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
        },
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_missing_data_section_raises(tmp_path: Path) -> None:
    """Missing required data section raises ValidationError."""
    cfg_file = _write_config(tmp_path, {"project_name": "test"})
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_invalid_max_rul_raises(tmp_path: Path) -> None:
    """max_rul <= 0 raises ValidationError."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
                "max_rul": -1,
            },
        },
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_invalid_test_size_raises(tmp_path: Path) -> None:
    """test_size outside (0, 1) raises ValidationError."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
                "test_size": 1.5,
            },
        },
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    """Syntactically invalid YAML raises yaml.YAMLError."""
    import yaml as _yaml

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [unclosed bracket")
    with pytest.raises(_yaml.YAMLError):
        load_config(bad_yaml)


def test_load_default_config() -> None:
    """The committed default.yaml loads and validates without error."""
    project_root = Path(__file__).parent.parent.parent
    cfg_path = project_root / "configs" / "default.yaml"
    cfg = load_config(cfg_path)
    assert cfg.project_name == "turbofan-rul-mlops"
    assert cfg.data.fd_subset == "FD001"
    assert cfg.data.max_rul == 125


def test_model_config_defaults_when_section_omitted(tmp_path: Path) -> None:
    """Model config has stable defaults when omitted from YAML."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
        },
    )
    cfg = load_config(cfg_file)
    assert cfg.model.name == "ridge"
    assert cfg.model.alpha == 1.0
    assert cfg.model.artifact_dir == Path("artifacts/models")


def test_model_config_loads_custom_values(tmp_path: Path) -> None:
    """Model config accepts configured baseline values."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "model": {
                "name": "ridge",
                "alpha": 2.5,
                "artifact_dir": "artifacts/custom",
            },
        },
    )
    cfg = load_config(cfg_file)
    assert cfg.model.name == "ridge"
    assert cfg.model.alpha == 2.5
    assert cfg.model.artifact_dir == Path("artifacts/custom")


def test_invalid_model_name_raises(tmp_path: Path) -> None:
    """Unsupported model name raises ValidationError."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "model": {"name": "neural-net"},
        },
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_invalid_model_alpha_raises(tmp_path: Path) -> None:
    """Ridge alpha must be positive."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "model": {"alpha": 0.0},
        },
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)
