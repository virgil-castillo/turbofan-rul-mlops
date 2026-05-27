"""Tests for turbofan.preprocessing.normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.preprocessing.normalization import (
    CMAPSS_SUBSET_MODE_COUNTS,
    OperatingModeNormalizer,
    mode_count_for_subset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df_single_mode() -> pd.DataFrame:
    """Return a small single-mode DataFrame for testing."""
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "op_1": [0.0, 0.0, 0.0, 0.0],
            "op_2": [0.0, 0.0, 0.0, 0.0],
            "op_3": [0.0, 0.0, 0.0, 0.0],
            "s_1": [100.0, 102.0, 104.0, 106.0],
            "s_2": [200.0, 204.0, 208.0, 212.0],
            "rul": [10, 9, 8, 7],
        }
    )


def _make_df_multi_mode() -> pd.DataFrame:
    """Return a small multi-mode DataFrame for testing."""
    return pd.DataFrame(
        {
            "engine_id": list(range(1, 9)),
            "cycle": [1] * 8,
            "op_1": [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0],
            "op_2": [0.0] * 8,
            "op_3": [0.0] * 8,
            "s_1": [100.0, 101.0, 99.0, 102.0, 200.0, 201.0, 199.0, 202.0],
        }
    )


# ---------------------------------------------------------------------------
# Existing tests (mode count)
# ---------------------------------------------------------------------------


def test_all_subset_mode_counts_are_positive_integers() -> None:
    """Every entry in CMAPSS_SUBSET_MODE_COUNTS is a positive integer."""
    for subset, count in CMAPSS_SUBSET_MODE_COUNTS.items():
        assert isinstance(count, int), f"{subset}: expected int, got {type(count)}"
        assert count > 0, f"{subset}: mode count must be positive"


def test_single_condition_subsets_have_mode_count_one() -> None:
    """FD001 and FD003 are treated as single-condition subsets."""
    assert mode_count_for_subset("FD001") == 1
    assert mode_count_for_subset("FD003") == 1


def test_multi_condition_subsets_have_mode_count_six() -> None:
    """FD002 and FD004 are treated as six-condition subsets."""
    assert mode_count_for_subset("FD002") == 6
    assert mode_count_for_subset("FD004") == 6


def test_unsupported_subset_raises_value_error() -> None:
    """mode_count_for_subset raises ValueError for unknown subset names."""
    with pytest.raises(ValueError, match="FD999"):
        mode_count_for_subset("FD999")


# ---------------------------------------------------------------------------
# Group A: Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_raises_for_n_modes_zero() -> None:
    """OperatingModeNormalizer raises ValueError when n_modes=0."""
    with pytest.raises(ValueError, match="n_modes"):
        OperatingModeNormalizer(n_modes=0)


def test_constructor_raises_for_negative_std_floor() -> None:
    """OperatingModeNormalizer raises ValueError for negative std_floor."""
    with pytest.raises(ValueError, match="std_floor"):
        OperatingModeNormalizer(std_floor=-0.001)


# ---------------------------------------------------------------------------
# Group B: Metadata preservation (single-mode)
# ---------------------------------------------------------------------------


def test_engine_id_cycle_rul_preserved() -> None:
    """engine_id, cycle, and rul columns are passed through unchanged."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1", "s_2"], n_modes=1)
    result = norm.fit_transform(df)
    pd.testing.assert_series_equal(result["engine_id"], df["engine_id"])
    pd.testing.assert_series_equal(result["cycle"], df["cycle"])
    pd.testing.assert_series_equal(result["rul"], df["rul"])


def test_op_cols_not_in_feature_cols_preserved() -> None:
    """op_cols not listed in feature_cols are copied through unchanged."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=1)
    result = norm.fit_transform(df)
    pd.testing.assert_series_equal(result["op_1"], df["op_1"])


# ---------------------------------------------------------------------------
# Group C: Single-mode normalization
# ---------------------------------------------------------------------------


def test_single_mode_explicit_feature_cols_normalizes_sensors() -> None:
    """After fit_transform with explicit feature_cols, sensor column mean is ~0."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1", "s_2"], n_modes=1)
    result = norm.fit_transform(df)
    assert abs(result["s_1"].mean()) < 1e-9
    assert abs(result["s_2"].mean()) < 1e-9


def test_single_mode_inferred_feature_cols_normalizes_sensors() -> None:
    """feature_cols=None infers numeric columns excluding engine_id, cycle, rul."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(n_modes=1)
    result = norm.fit_transform(df)
    assert abs(result["s_1"].mean()) < 1e-9
    assert abs(result["s_2"].mean()) < 1e-9
    assert set(norm.feature_cols_).isdisjoint({"engine_id", "cycle", "rul"})


def test_op_cols_in_feature_cols_normalized_globally() -> None:
    """op_1 (constant 0.0) hits std floor, so normalized value stays 0.0."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["op_1", "s_1"], n_modes=1)
    result = norm.fit_transform(df)
    np.testing.assert_array_almost_equal(result["op_1"].values, [0.0, 0.0, 0.0, 0.0])


def test_near_zero_std_replaced_with_one_in_global_stats() -> None:
    """Single-row DataFrame produces NaN std; floor logic replaces with 1.0."""
    df = pd.DataFrame(
        {
            "op_1": [0.0],
            "op_2": [0.0],
            "op_3": [0.0],
            "s_1": [100.0],
        }
    )
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=1)
    norm.fit(df)
    assert norm.global_stds_["s_1"] == 1.0


def test_transform_before_fit_raises_runtime_error() -> None:
    """transform() before fit() raises RuntimeError."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(n_modes=1)
    with pytest.raises(RuntimeError):
        norm.transform(df)


def test_fit_raises_key_error_for_missing_op_cols() -> None:
    """fit() raises KeyError when op_cols are absent from X."""
    df = pd.DataFrame({"s_1": [1.0, 2.0]})
    norm = OperatingModeNormalizer(n_modes=1)
    with pytest.raises(KeyError):
        norm.fit(df)


def test_transform_returns_copy() -> None:
    """transform() does not mutate the original DataFrame."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1", "s_2"], n_modes=1)
    norm.fit(df)
    original_s1 = df["s_1"].copy()
    norm.transform(df)
    pd.testing.assert_series_equal(df["s_1"], original_s1)


# ---------------------------------------------------------------------------
# Group D: Multi-mode
# ---------------------------------------------------------------------------


def test_multi_mode_fits_mode_centers() -> None:
    """mode_centers_ is set and has exactly n_modes entries after fit."""
    df = _make_df_multi_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=2)
    norm.fit(df)
    assert norm.mode_centers_ is not None
    assert len(norm.mode_centers_) == 2


def test_multi_mode_normalizes_sensors_per_mode() -> None:
    """Within each KMeans cluster, the mean of s_1 is approximately 0."""
    df = _make_df_multi_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=2)
    result = norm.fit_transform(df)
    modes = norm._assign_modes(df)
    for mode_idx in np.unique(modes):
        mask = modes == mode_idx
        assert abs(result.loc[mask, "s_1"].mean()) < 1e-6


def test_multi_mode_row_between_clusters_is_non_nan() -> None:
    """Any row produces a non-NaN, non-Inf result regardless of op position.

    A row whose op_1 lies between the two training cluster centres is assigned
    to the nearest centroid (nearest-centroid assignment always produces a
    seen mode), so the output is finite.  This confirms the transform pipeline
    handles arbitrary op positions without producing degenerate values.
    """
    df = _make_df_multi_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=2)
    norm.fit(df)

    df_new = pd.DataFrame(
        {
            "engine_id": [99],
            "cycle": [1],
            "op_1": [5.5],
            "op_2": [0.0],
            "op_3": [0.0],
            "s_1": [150.0],
        }
    )
    result = norm.transform(df_new)
    assert not result["s_1"].isna().any()


def test_n_modes_exceeds_rows_raises_value_error() -> None:
    """fit() raises ValueError when n_modes > number of rows in X."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=10)
    with pytest.raises(ValueError, match="n_modes"):
        norm.fit(df)


# ---------------------------------------------------------------------------
# Group E: Payload serialization
# ---------------------------------------------------------------------------


def test_payload_has_schema_version_one_and_type() -> None:
    """to_payload() returns schema_version=1 and normalizer_type='operating_mode'."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=1)
    norm.fit(df)
    payload = norm.to_payload()
    assert payload["schema_version"] == 1
    assert payload["normalizer_type"] == "operating_mode"


def test_single_mode_round_trip_produces_identical_transforms() -> None:
    """from_payload(to_payload()) produces the same transform output."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1", "s_2"], n_modes=1)
    norm.fit(df)
    payload = norm.to_payload()
    restored = OperatingModeNormalizer.from_payload(payload)
    pd.testing.assert_frame_equal(norm.transform(df), restored.transform(df))


def test_multi_mode_round_trip_produces_identical_transforms() -> None:
    """from_payload(to_payload()) is identical for multi-mode normalizer."""
    df = _make_df_multi_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=2)
    norm.fit(df)
    payload = norm.to_payload()
    restored = OperatingModeNormalizer.from_payload(payload)
    pd.testing.assert_frame_equal(norm.transform(df), restored.transform(df))


def test_from_payload_unsupported_schema_version_raises() -> None:
    """from_payload raises ValueError for schema_version != 1."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=1)
    norm.fit(df)
    payload = norm.to_payload()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        OperatingModeNormalizer.from_payload(payload)


def test_from_payload_missing_required_key_raises() -> None:
    """from_payload raises ValueError when a required key is absent."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=1)
    norm.fit(df)
    payload = norm.to_payload()
    del payload["global_means"]
    with pytest.raises(ValueError, match="global_means"):
        OperatingModeNormalizer.from_payload(payload)


def test_from_payload_non_numeric_stat_raises() -> None:
    """from_payload raises ValueError when a stat value is non-numeric."""
    df = _make_df_single_mode()
    norm = OperatingModeNormalizer(feature_cols=["s_1"], n_modes=1)
    norm.fit(df)
    payload = norm.to_payload()
    payload["global_means"]["s_1"] = "not_a_number"
    with pytest.raises(ValueError, match="numeric"):
        OperatingModeNormalizer.from_payload(payload)
