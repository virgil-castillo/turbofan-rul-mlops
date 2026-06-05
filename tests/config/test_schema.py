"""Tests for turbofan.config.schema."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from turbofan.config.schema import DataConfig, FeatureConfig, load_config


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
    assert cfg.inference.artifact_path is None
    assert cfg.inference.host == "0.0.0.0"
    assert cfg.inference.port == 8000
    assert cfg.inference.allow_partial is False
    assert cfg.features.sensor_cols_to_drop == []
    assert cfg.model.name == "ridge"
    assert cfg.model.alpha == 100.0


def test_feature_config_defaults_when_section_omitted(tmp_path: Path) -> None:
    """Feature config has stable defaults when omitted from YAML."""
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
    assert cfg.features.sensor_cols_to_drop == []


def test_feature_config_loads_custom_values(tmp_path: Path) -> None:
    """Feature config accepts an explicit sensor drop list."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "features": {
                "sensor_cols_to_drop": ["s_1", "s_5", "s_16"],
            },
        },
    )
    cfg = load_config(cfg_file)
    assert cfg.features.sensor_cols_to_drop == ["s_1", "s_5", "s_16"]


def test_invalid_feature_config_raises() -> None:
    """Non-positive n_modes is rejected."""
    with pytest.raises(ValidationError):
        FeatureConfig(n_modes=0)


def test_inference_config_loads_custom_values(tmp_path: Path) -> None:
    """Inference config accepts local serving values."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "inference": {
                "artifact_path": "artifacts/models/baseline/run",
                "host": "127.0.0.1",
                "port": 9000,
                "allow_partial": True,
            },
        },
    )
    cfg = load_config(cfg_file)
    assert cfg.inference.artifact_path == Path("artifacts/models/baseline/run")
    assert cfg.inference.host == "127.0.0.1"
    assert cfg.inference.port == 9000
    assert cfg.inference.allow_partial is True


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
    assert cfg.model.alpha == 100.0
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


def test_invalid_model_feature_set_raises(tmp_path: Path) -> None:
    """Unsupported feature sets are rejected."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "features": {"feature_set": "ops"},
        },
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_invalid_model_window_raises(tmp_path: Path) -> None:
    """Rolling windows must be positive."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "features": {"windows": [0]},
        },
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_sequence_config_defaults_when_section_omitted(tmp_path: Path) -> None:
    """Sequence config has stable defaults when omitted from YAML."""
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
    assert cfg.sequence.architecture == "gru"
    assert cfg.sequence.window_size == 45
    assert cfg.sequence.batch_size == 64
    assert cfg.sequence.hidden_size == 64
    assert cfg.sequence.num_layers == 1
    assert cfg.sequence.dropout == 0.0
    assert cfg.sequence.learning_rate == 0.001
    assert cfg.sequence.epochs == 50
    assert cfg.sequence.patience == 8
    assert cfg.sequence.device == "cpu"
    assert cfg.sequence.artifact_dir == Path("artifacts/models")


def test_sequence_config_loads_custom_values(tmp_path: Path) -> None:
    """Sequence config accepts configured GRU training values."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "sequence": {
                "architecture": "gru",
                "window_size": 12,
                "batch_size": 8,
                "hidden_size": 16,
                "num_layers": 2,
                "dropout": 0.1,
                "learning_rate": 0.01,
                "epochs": 5,
                "patience": 2,
                "device": "cpu",
                "artifact_dir": "artifacts/custom-sequence",
            },
        },
    )
    cfg = load_config(cfg_file)
    assert cfg.sequence.window_size == 12
    assert cfg.sequence.batch_size == 8
    assert cfg.sequence.hidden_size == 16
    assert cfg.sequence.num_layers == 2
    assert cfg.sequence.dropout == 0.1
    assert cfg.sequence.learning_rate == 0.01
    assert cfg.sequence.epochs == 5
    assert cfg.sequence.patience == 2
    assert cfg.sequence.device == "cpu"
    assert cfg.sequence.artifact_dir == Path("artifacts/custom-sequence")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("architecture", "transformer"),
        ("window_size", 0),
        ("batch_size", 0),
        ("hidden_size", 0),
        ("num_layers", 0),
        ("dropout", 1.0),
        ("learning_rate", 0.0),
        ("epochs", 0),
        ("patience", 0),
        ("device", "mps"),
    ],
)
def test_invalid_sequence_config_raises(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Invalid sequence config values raise ValidationError."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "sequence": {field: value},
        },
    )
    with pytest.raises(ValidationError):
        load_config(cfg_file)


# ── _base_ config composition ─────────────────────────────────────────────────

_BASE_DATA = {
    "project_name": "base-project",
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "interim_dir": "data/interim",
        "fd_subset": "FD001",
        "max_rul": 125,
    },
    "features": {"sensor_cols_to_drop": [], "n_modes": 1},
}


def test_base_key_loads_base_config(tmp_path: Path) -> None:
    """A config with _base_ inherits all fields from the referenced file."""
    base = tmp_path / "default.yaml"
    base.write_text(yaml.dump(_BASE_DATA))

    override = tmp_path / "override.yaml"
    override.write_text(yaml.dump({"_base_": "default.yaml"}))

    cfg = load_config(override)
    assert cfg.project_name == "base-project"
    assert cfg.data.fd_subset == "FD001"


def test_base_key_override_replaces_scalar(tmp_path: Path) -> None:
    """Scalar values in the override replace those from the base."""
    base = tmp_path / "default.yaml"
    base.write_text(yaml.dump(_BASE_DATA))

    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.dump({"_base_": "default.yaml", "project_name": "override-project"})
    )

    cfg = load_config(override)
    assert cfg.project_name == "override-project"


def test_base_key_override_merges_nested(tmp_path: Path) -> None:
    """Nested dicts are merged: override keys replace, unmentioned keys are kept."""
    base = tmp_path / "default.yaml"
    base.write_text(yaml.dump(_BASE_DATA))

    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.dump({"_base_": "default.yaml", "data": {"fd_subset": "FD002"}})
    )

    cfg = load_config(override)
    assert cfg.data.fd_subset == "FD002"
    assert cfg.data.max_rul == 125


def test_base_key_override_replaces_list(tmp_path: Path) -> None:
    """List values in the override replace the base list entirely."""
    base = tmp_path / "default.yaml"
    base.write_text(yaml.dump(_BASE_DATA))

    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.dump(
            {
                "_base_": "default.yaml",
                "features": {"sensor_cols_to_drop": ["s_1", "s_5"], "n_modes": 6},
            }
        )
    )

    cfg = load_config(override)
    assert cfg.features.sensor_cols_to_drop == ["s_1", "s_5"]
    assert cfg.features.n_modes == 6


def test_base_key_not_present_in_validated_config(tmp_path: Path) -> None:
    """The _base_ key is consumed and does not leak into the validated model."""
    base = tmp_path / "default.yaml"
    base.write_text(yaml.dump(_BASE_DATA))

    override = tmp_path / "override.yaml"
    override.write_text(yaml.dump({"_base_": "default.yaml"}))

    cfg = load_config(override)
    assert not hasattr(cfg, "_base_")


# ── new rolling-family FeatureSetName values ───────────────────────────────────


@pytest.mark.parametrize(
    "feature_set",
    ["rolling_std", "rolling_slope", "rolling_delta", "rolling_min", "rolling_max"],
)
def test_new_feature_set_names_accepted_in_feature_config(
    tmp_path: Path, feature_set: str
) -> None:
    """New modular rolling feature sets are valid FeatureSetName values."""
    cfg_file = _write_config(
        tmp_path,
        {
            "project_name": "test-project",
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "interim_dir": "data/interim",
            },
            "features": {"feature_set": feature_set},
        },
    )
    cfg = load_config(cfg_file)
    assert cfg.features.feature_set == feature_set
