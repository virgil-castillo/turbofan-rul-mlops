"""Baseline sklearn pipeline factory."""
from __future__ import annotations

from typing import Literal

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from turbofan.features.pipeline import build_feature_pipeline


def build_baseline_pipeline(
    model_name: Literal["ridge"] = "ridge",
    alpha: float = 1.0,
    windows: list[int] | None = None,
    op_cols: list[str] | None = None,
) -> Pipeline:
    """Build an unfitted feature-plus-regressor sklearn Pipeline.

    Args:
        model_name: Baseline model identifier. Only ``"ridge"`` is supported.
        alpha: Ridge regularization strength.
        windows: Rolling window sizes for the feature pipeline.
        op_cols: Operational setting columns for normalization.

    Returns:
        Unfitted sklearn Pipeline with ``features`` and ``model`` steps.

    Raises:
        ValueError: If ``model_name`` is unsupported.
    """
    if model_name != "ridge":
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(
        [
            ("features", build_feature_pipeline(windows=windows, op_cols=op_cols)),
            ("model", Ridge(alpha=alpha)),
        ]
    )
