"""Tests for turbofan.sequences.normalize."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from turbofan.sequences.normalize import SequenceNormalizer, default_feature_cols


def test_default_feature_cols_returns_ops_and_sensors() -> None:
    """Default feature columns include operating settings and sensors."""
    expected = ["op_1", "op_2", "op_3"] + [
        f"s_{idx}" for idx in range(1, 22)
    ]
    result = default_feature_cols()
    assert result == expected
    assert len(result) == 24


def test_fit_transform_preserves_metadata_and_normalizes_features() -> None:
    """Fit-transform keeps metadata unchanged and z-scores features."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 2],
            "cycle": [1, 2, 1],
            "rul": [30, 29, 45],
            "op_1": [2.0, 4.0, 6.0],
            "op_2": [2.0, 2.0, 2.0],
            "s_1": [14.0, 16.0, 18.0],
        }
    )
    normalizer = SequenceNormalizer(feature_cols=["op_1", "op_2", "s_1"])

    result = normalizer.fit_transform(df)

    pd.testing.assert_series_equal(result["engine_id"], df["engine_id"])
    pd.testing.assert_series_equal(result["cycle"], df["cycle"])
    pd.testing.assert_series_equal(result["rul"], df["rul"])
    assert normalizer.means_["op_1"] == 4.0
    assert normalizer.means_["op_2"] == 2.0
    assert normalizer.means_["s_1"] == 16.0
    assert normalizer.stds_["op_2"] == 1.0
    assert result["op_2"].tolist() == [0.0, 0.0, 0.0]
    assert abs(result["op_1"].mean()) < 1e-12


def test_transform_uses_training_statistics_for_validation_rows() -> None:
    """Transform applies stats learned only from training rows."""
    train = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [1, 2],
            "rul": [10, 9],
            "op_1": [1.0, 3.0],
        }
    )
    validation = pd.DataFrame(
        {
            "engine_id": [2, 2],
            "cycle": [1, 2],
            "rul": [8, 7],
            "op_1": [5.0, 7.0],
        }
    )
    normalizer = SequenceNormalizer(feature_cols=["op_1"])

    normalizer.fit(train)
    result = normalizer.transform(validation)

    assert result["op_1"].tolist() == [3.0, 5.0]


def test_transform_with_integer_features_returns_floats_without_warning() -> None:
    """Transform safely normalizes integer feature columns to floats."""
    train = pd.DataFrame({"op_1": [1, 4], "s_1": [10, 15]})
    validation = pd.DataFrame({"op_1": [5, 7], "s_1": [18, 22]})
    normalizer = SequenceNormalizer(feature_cols=["op_1", "s_1"]).fit(train)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = normalizer.transform(validation)

    future_warnings = [
        warning
        for warning in caught_warnings
        if issubclass(warning.category, FutureWarning)
    ]
    assert future_warnings == []
    assert result["op_1"].tolist() == pytest.approx([5.0 / 3.0, 3.0])
    assert result["s_1"].tolist() == pytest.approx([2.2, 3.8])
    assert pd.api.types.is_float_dtype(result["op_1"])
    assert pd.api.types.is_float_dtype(result["s_1"])


def test_fit_missing_feature_column_raises_key_error() -> None:
    """Fit raises KeyError when configured feature columns are missing."""
    df = pd.DataFrame({"op_1": [1.0, 2.0]})
    normalizer = SequenceNormalizer(feature_cols=["op_1", "s_1"])

    with pytest.raises(KeyError, match="s_1"):
        normalizer.fit(df)


def test_transform_before_fit_raises_runtime_error() -> None:
    """Transform raises RuntimeError when called before fit."""
    normalizer = SequenceNormalizer(feature_cols=["op_1"])
    df = pd.DataFrame({"op_1": [1.0]})

    with pytest.raises(RuntimeError, match="fit"):
        normalizer.transform(df)


def test_constructor_accepts_sequence_and_stores_list() -> None:
    """Constructor accepts any sequence of strings and stores a list."""
    normalizer = SequenceNormalizer(feature_cols=("op_1", "op_2"))

    assert normalizer.feature_cols == ["op_1", "op_2"]
