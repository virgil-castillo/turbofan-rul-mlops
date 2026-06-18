"""Typed runtime-boundary helpers for sklearn-style objects."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self

import pandas as pd

if TYPE_CHECKING:

    class BaseEstimator:
        """Typed stand-in for sklearn's untyped ``BaseEstimator`` base."""

        def get_params(self, deep: bool = True) -> dict[str, object]:
            """Return estimator parameters.

            Args:
                deep: Whether to include nested estimator parameters.

            Returns:
                Mapping of parameter names to values.
            """
            ...

        def set_params(self, **params: object) -> Self:
            """Set estimator parameters.

            Args:
                **params: Parameter values keyed by name.

            Returns:
                This estimator instance.
            """
            ...

    class TransformerMixin:
        """Typed stand-in for sklearn's untyped ``TransformerMixin`` base."""

        def fit_transform(
            self,
            X: pd.DataFrame,
            y: object = None,
            **fit_params: object,
        ) -> object:
            """Fit to data, then transform it.

            Args:
                X: Input DataFrame.
                y: Optional target values.
                **fit_params: Additional fit parameters.

            Returns:
                Transformed data.
            """
            ...

else:
    from sklearn.base import BaseEstimator as BaseEstimator
    from sklearn.base import TransformerMixin as TransformerMixin


__all__ = [
    "BaseEstimator",
    "DataFramePredictor",
    "DataFrameTransformer",
    "TransformerMixin",
]


class DataFramePredictor(Protocol):
    """Protocol for fitted predictors that consume a pandas DataFrame."""

    def predict(self, frame: pd.DataFrame) -> object:
        """Predict from input rows.

        Args:
            frame: Feature or raw input rows.

        Returns:
            Predictor-specific output.
        """
        ...


class DataFrameTransformer(Protocol):
    """Protocol for fitted transformers that consume a pandas DataFrame."""

    def transform(self, frame: pd.DataFrame) -> object:
        """Transform input rows.

        Args:
            frame: Input rows.

        Returns:
            Transformer-specific output.
        """
        ...
