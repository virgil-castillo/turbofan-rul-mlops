"""Feature engineering pipeline factory."""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from turbofan.features.rolling import RollingFeatureExtractor
from turbofan.features.sensor_dropper import SensorDropper
from turbofan.preprocessing.normalization import OperatingModeNormalizer


def build_feature_pipeline(
    windows: list[int] | None = None,
    op_cols: list[str] | None = None,
    sensor_drop: list[str] | None = None,
    n_modes: int = 1,
    random_state: int = 42,
) -> Pipeline:
    """Build an unfitted feature engineering pipeline.

    Returns an sklearn Pipeline with three named steps:
    ``sensor_dropper``, ``rolling_features``, ``normalizer``.

    Args:
        windows: Rolling window sizes. Default ``[5, 10, 20]``.
        op_cols: Operational setting columns.
            Default ``["op_1", "op_2", "op_3"]``.
        sensor_drop: Sensor column names to drop before rolling extraction.
            Determined from EDA; passed directly to ``SensorDropper``.
        n_modes: Number of operating modes for the normalizer.
        random_state: KMeans random seed for the normalizer.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline(
        [
            (
                "sensor_dropper",
                SensorDropper(drop=sensor_drop),
            ),
            (
                "rolling_features",
                RollingFeatureExtractor(windows=windows),
            ),
            (
                "normalizer",
                OperatingModeNormalizer(
                    op_cols=op_cols,
                    n_modes=n_modes,
                    random_state=random_state,
                ),
            ),
        ]
    )
