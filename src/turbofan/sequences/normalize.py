"""Feature normalization utilities for sequence model inputs."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import pandas as pd


def default_feature_cols() -> list[str]:
    """Return default turbofan sequence feature columns.

    Returns:
        Operating setting columns followed by sensor columns.
    """
    return ["op_1", "op_2", "op_3"] + [f"s_{idx}" for idx in range(1, 22)]


class SequenceNormalizer:
    """Z-score normalize sequence feature columns using training statistics.

    Args:
        feature_cols: Feature columns to normalize. Defaults to operating
            setting columns and all 21 sensor columns.
    """

    def __init__(self, feature_cols: Sequence[str] | None = None) -> None:
        self.feature_cols = (
            list(feature_cols)
            if feature_cols is not None
            else default_feature_cols()
        )

    def fit(self, df: pd.DataFrame) -> Self:
        """Compute feature means and population standard deviations.

        Args:
            df: Training rows used to compute normalization statistics.

        Returns:
            Fitted normalizer.

        Raises:
            KeyError: If any configured feature column is missing.
        """
        self._validate_columns(df)
        self.means_ = df[self.feature_cols].mean()
        stds = df[self.feature_cols].std(ddof=0)
        self.stds_ = stds.fillna(1.0).replace(0.0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted z-score normalization to feature columns.

        Args:
            df: Rows to transform using fitted training statistics.

        Returns:
            Copy of the input DataFrame with feature columns normalized and
            non-feature columns preserved.

        Raises:
            RuntimeError: If called before fit.
            KeyError: If any configured feature column is missing.
        """
        if not hasattr(self, "means_") or not hasattr(self, "stds_"):
            raise RuntimeError("SequenceNormalizer must be fit before transform.")
        self._validate_columns(df)
        result = df.copy()
        result[self.feature_cols] = result[self.feature_cols].astype("float64")
        result.loc[:, self.feature_cols] = (
            result.loc[:, self.feature_cols] - self.means_
        ) / self.stds_
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training rows, then transform those rows.

        Args:
            df: Training rows to fit and transform.

        Returns:
            Copy of the input DataFrame with feature columns normalized.

        Raises:
            KeyError: If any configured feature column is missing.
        """
        return self.fit(df).transform(df)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing_cols = [
            col for col in self.feature_cols if col not in df.columns
        ]
        if missing_cols:
            missing = ", ".join(missing_cols)
            raise KeyError(f"Missing feature columns: {missing}")
