"""Tests that the committed explicit-GRU subset configs load and resolve."""
from __future__ import annotations

from pathlib import Path

import pytest

from turbofan.config.schema import load_config

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


@pytest.mark.parametrize("subset", _SUBSETS)
def test_gru_subset_config_selects_gru_and_inherits_hyperparameters(
    subset: str,
) -> None:
    """Each fd00X_gru.yaml selects GRU while inheriting the subset base config.

    The explicit GRU variant overrides only the architecture: the data subset
    and the shared sequence hyperparameters (window size, hidden size, learning
    rate) come straight from the matching subset base config.
    """
    gru_path = _PROJECT_ROOT / "configs" / "subsets" / f"{subset.lower()}_gru.yaml"
    base_path = _PROJECT_ROOT / "configs" / "subsets" / f"{subset.lower()}.yaml"

    gru_cfg = load_config(gru_path)
    base_cfg = load_config(base_path)

    assert gru_cfg.sequence.architecture == "gru"
    assert gru_cfg.data.fd_subset == subset
    assert gru_cfg.sequence.window_size == base_cfg.sequence.window_size
    assert gru_cfg.sequence.hidden_size == base_cfg.sequence.hidden_size
    assert gru_cfg.sequence.learning_rate == base_cfg.sequence.learning_rate


@pytest.mark.parametrize("subset", _SUBSETS)
def test_gru_subset_config_resolves_gru_features(subset: str) -> None:
    """The explicit GRU config resolves the subset's GRU feature settings."""
    gru_path = _PROJECT_ROOT / "configs" / "subsets" / f"{subset.lower()}_gru.yaml"
    base_path = _PROJECT_ROOT / "configs" / "subsets" / f"{subset.lower()}.yaml"

    gru_cfg = load_config(gru_path)
    base_cfg = load_config(base_path)

    resolved = gru_cfg.features.for_model("gru")
    base_resolved = base_cfg.features.for_model("gru")

    assert resolved.feature_families == base_resolved.feature_families
    assert resolved.windows == base_resolved.windows
