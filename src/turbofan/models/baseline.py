"""Baseline sklearn pipeline factory."""
from __future__ import annotations

from typing import Literal

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from turbofan.features.engineering import FeatureFamily
from turbofan.features.pipeline import build_feature_pipeline


def build_baseline_pipeline(
    model_name: Literal["ridge"] = "ridge",
    alpha: float = 100.0,
    op_cols: list[str] | None = None,
    sensor_drop: list[str] | None = None,
    n_modes: int = 1,
    random_state: int = 42,
    feature_families: list[FeatureFamily] | None = None,
    windows: list[int] | None = None,
    lag_steps: list[int] | None = None,
) -> Pipeline:
    """Build an unfitted feature-plus-regressor sklearn Pipeline.

    The pipeline has two steps: ``features`` (the shared feature pipeline)
    and ``model`` (a Ridge regressor). All feature engineering is handled
    by ``build_feature_pipeline``.

    Args:
        model_name: Baseline model identifier. Only ``"ridge"`` is supported.
        alpha: Ridge regularization strength.
        op_cols: Operating-condition columns for KMeans clustering.
        sensor_drop: Sensor column names to remove before feature engineering.
        n_modes: Operating-mode count for ``OperatingModeNormalizer``.
        random_state: KMeans random seed for the normalizer.
        feature_families: Ordered feature families to expose to Ridge.
        windows: Rolling window sizes. Forwarded to ``FeatureEngineer``.
        lag_steps: Lag offsets. Forwarded to ``FeatureEngineer``.

    Returns:
        Unfitted sklearn Pipeline with feature engineering and Ridge.

    Raises:
        ValueError: If ``model_name`` is unsupported.
    """
    if model_name != "ridge":
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(
        [
            (
                "features",
                build_feature_pipeline(
                    op_cols=op_cols,
                    sensor_drop=sensor_drop,
                    n_modes=n_modes,
                    random_state=random_state,
                    feature_families=feature_families,
                    windows=windows,
                    lag_steps=lag_steps,
                ),
            ),
            ("model", Ridge(alpha=alpha)),
        ]
    )
