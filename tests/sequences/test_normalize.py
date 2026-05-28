"""Tests for turbofan.sequences.normalize."""
from __future__ import annotations

import pandas as pd


def test_operating_mode_normalizer_works_with_sequence_feature_cols() -> None:
    """OperatingModeNormalizer normalizes operating-setting and sensor columns."""
    from turbofan.preprocessing.normalization import OperatingModeNormalizer

    feature_cols = ["op_1", "op_2", "op_3"] + [f"s_{i}" for i in range(1, 22)]
    train = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "rul": [10, 9, 8, 7],
            **{col: [float(i) for i in range(4)] for col in feature_cols},
        }
    )
    normalizer = OperatingModeNormalizer(feature_cols=feature_cols)
    normalizer.fit(train)
    val = pd.DataFrame(
        {
            "engine_id": [3],
            "cycle": [1],
            "rul": [5],
            **{col: [1.0] for col in feature_cols},
        }
    )
    result = normalizer.transform(val)

    assert list(result["engine_id"]) == [3]
    assert list(result["cycle"]) == [1]
    assert list(result["rul"]) == [5]
    assert all(col in result.columns for col in feature_cols)
    for col in feature_cols:
        assert pd.api.types.is_float_dtype(result[col])
