"""Baseline sklearn pipeline factory."""
from __future__ import annotations

from typing import Literal

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from turbofan.config.schema import ProjectConfig
from turbofan.features import pipeline
from turbofan.features.engineering import FeatureFamily


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
                pipeline.build_feature_pipeline(
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


def build_ridge_estimator(cfg: ProjectConfig, *, seed: int) -> Pipeline:
    """Build the unfitted Ridge feature-plus-model pipeline for a config.

    Args:
        cfg: Loaded project config (model + feature settings).
        seed: KMeans-normalizer ``random_state`` for the pipeline.

    Returns:
        The unfitted Ridge sklearn pipeline.
    """
    rf = cfg.features.for_model("ridge")
    return build_baseline_pipeline(
        model_name=cfg.model.name,
        alpha=cfg.model.alpha,
        feature_families=rf.feature_families,
        windows=rf.windows,
        lag_steps=rf.lag_steps,
        sensor_drop=cfg.features.sensor_cols_to_drop or None,
        n_modes=cfg.features.n_modes,
        random_state=seed,
    )
