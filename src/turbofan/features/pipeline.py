"""Feature engineering pipeline factory."""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from turbofan.features.normalizer import OperationalNormalizer
from turbofan.features.rolling import RollingFeatureExtractor
from turbofan.features.sensor_dropper import SensorDropper


def build_feature_pipeline(
    windows: list[int] | None = None,
    op_cols: list[str] | None = None,
    sensor_std_threshold: float = 0.0,
    sensor_keep: list[str] | None = None,
) -> Pipeline:
    """Build an unfitted feature engineering pipeline.

    Returns an sklearn Pipeline with three named steps:
    ``sensor_dropper``, ``rolling_features``, ``normalizer``.

    Args:
        windows: Rolling window sizes. Default ``[5, 10, 20]``.
        op_cols: Operational setting columns.
            Default ``["op_1", "op_2", "op_3"]``.
        sensor_std_threshold: Maximum training standard deviation at
            which sensor columns are dropped.
        sensor_keep: Sensor columns to force-keep even when low-variance.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline(
        [
            (
                "sensor_dropper",
                SensorDropper(
                    std_threshold=sensor_std_threshold,
                    keep=sensor_keep,
                ),
            ),
            (
                "rolling_features",
                RollingFeatureExtractor(windows=windows),
            ),
            (
                "normalizer",
                OperationalNormalizer(op_cols=op_cols),
            ),
        ]
    )
