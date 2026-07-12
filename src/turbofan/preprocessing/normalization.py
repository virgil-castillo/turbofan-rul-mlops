"""Operating-mode-aware normalization for C-MAPSS turbofan data."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import KMeans

from turbofan.sklearn_types import BaseEstimator, TransformerMixin

_EXCLUDE_COLS: frozenset[str] = frozenset({"engine_id", "cycle", "rul"})
_DEFAULT_OP_COLS: list[str] = ["op_1", "op_2", "op_3"]
_REQUIRED_PAYLOAD_KEYS: tuple[str, ...] = (
    "schema_version",
    "normalizer_type",
    "feature_cols",
    "op_cols",
    "sensor_feature_cols",
    "n_modes",
    "std_floor",
    "random_state",
    "mode_centers",
    "global_means",
    "global_stds",
    "mode_means",
    "mode_stds",
)


def _require_key(payload: dict[str, object], key: str) -> object:
    """Return payload[key], raising ValueError if the key is absent.

    Args:
        payload: Deserialized payload dictionary.
        key: Required key name.

    Returns:
        The value stored at ``key``.

    Raises:
        ValueError: If ``key`` is not present in ``payload``.
    """
    if key not in payload:
        raise ValueError(f"Missing required key in payload: {key!r}")
    return payload[key]


def _series_from_mapping(
    mapping: dict[str, Any], name: str
) -> pd.Series[Any]:
    """Convert a ``{str: float}`` mapping to a pandas Series, validating numeric values.

    Args:
        mapping: Dictionary of string keys to numeric values.
        name: Human-readable label used in error messages (e.g. ``"global_means"``).

    Returns:
        A ``pd.Series`` with the mapping's keys as index and float64 values.

    Raises:
        ValueError: If any value in ``mapping`` is non-numeric.
    """
    for k, v in mapping.items():
        try:
            float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Non-numeric value in {name!r}: key={k!r}, value={v!r}"
            ) from exc
    return pd.Series({k: float(v) for k, v in mapping.items()}, dtype=np.float64)


def _apply_floor(s: pd.Series[Any], std_floor: float) -> pd.Series[Any]:
    """Replace NaN and values at or below std_floor with 1.0.

    Args:
        s: Series of standard-deviation values.
        std_floor: Minimum absolute standard deviation; values at or below this
            threshold are replaced with 1.0.

    Returns:
        A new Series with floored values.
    """
    result = s.copy()
    result[result.isna()] = 1.0
    result[result.abs() <= std_floor] = 1.0
    return result


class OperatingModeNormalizer(BaseEstimator, TransformerMixin):
    """Normalize turbofan sensor readings per operating mode.

    For single-mode data (``n_modes=1``) all rows share a single set of
    mean/std statistics.  For multi-mode data (``n_modes > 1``) KMeans is
    fitted on the operating-condition columns (``op_cols``) and per-cluster
    statistics are computed for the sensor columns.

    The operating-condition columns themselves (``op_cols``) are normalized
    globally when they appear in ``feature_cols``; pass ``feature_cols``
    without the ``op_cols`` to keep them unnormalized.

    Args:
        feature_cols: Columns to normalize.  ``None`` infers all numeric
            columns from the training DataFrame, excluding ``engine_id``,
            ``cycle``, and ``rul``.
        op_cols: Operating-condition columns used for KMeans clustering.
            Defaults to ``["op_1", "op_2", "op_3"]``.
        n_modes: Number of operating modes / KMeans clusters.  Must be
            positive.
        std_floor: Threshold for the minimum absolute standard deviation.
            Values at or below this threshold are replaced with ``1.0``
            (not with ``std_floor`` itself) to avoid division-by-near-zero.
        random_state: Random seed forwarded to ``sklearn.cluster.KMeans``.

    Raises:
        ValueError: If ``n_modes <= 0`` or ``std_floor < 0``.
    """

    def __init__(
        self,
        feature_cols: Sequence[str] | None = None,
        op_cols: Sequence[str] | None = None,
        n_modes: int = 1,
        std_floor: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        if n_modes <= 0:
            raise ValueError(f"n_modes must be positive, got {n_modes}")
        if std_floor < 0:
            raise ValueError(f"std_floor must be non-negative, got {std_floor}")
        self.feature_cols = feature_cols
        self.op_cols = op_cols
        self.n_modes = n_modes
        self.std_floor = std_floor
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Fitted attributes (set by fit)
    # ------------------------------------------------------------------

    # op_cols_: list[str]
    # feature_cols_: list[str]
    # sensor_feature_cols_: list[str]
    # mode_centers_: list[list[float]] | None
    # global_means_: pd.Series[Any]
    # global_stds_: pd.Series[Any]
    # mode_means_: dict[int, pd.Series[Any]]
    # mode_stds_: dict[int, pd.Series[Any]]

    def fit(self, X: pd.DataFrame, y: pd.Series[Any] | None = None) -> Self:  # noqa: ARG002 - sklearn transformer API requires this signature
        """Compute per-mode statistics from ``X``.

        Args:
            X: Training DataFrame with operating-condition and sensor columns.
            y: Ignored; present for sklearn API compatibility.

        Returns:
            This fitted normalizer (``self``).

        Raises:
            KeyError: If any column in ``op_cols`` or explicit ``feature_cols``
                is absent from ``X``.
            ValueError: If ``n_modes`` exceeds the number of rows in ``X``.
        """
        op_cols: list[str] = (
            list(self.op_cols) if self.op_cols is not None else list(_DEFAULT_OP_COLS)
        )
        missing_op = [c for c in op_cols if c not in X.columns]
        if missing_op:
            raise KeyError(f"op_cols not found in X: {missing_op}")

        if self.feature_cols is not None:
            feature_cols: list[str] = list(self.feature_cols)
            missing_feat = [c for c in feature_cols if c not in X.columns]
            if missing_feat:
                raise KeyError(f"feature_cols not found in X: {missing_feat}")
        else:
            feature_cols = [
                c
                for c in X.select_dtypes(include=[np.number]).columns
                if c not in _EXCLUDE_COLS
            ]

        if self.n_modes > len(X):
            raise ValueError(
                f"n_modes ({self.n_modes}) exceeds the number of rows in X ({len(X)})"
            )

        self.op_cols_: list[str] = op_cols
        self.feature_cols_: list[str] = feature_cols
        op_set = set(op_cols)
        self.sensor_feature_cols_: list[str] = [
            c for c in feature_cols if c not in op_set
        ]

        # Global stats (over all feature_cols_)
        Xf = X[feature_cols].astype(np.float64)
        self.global_means_: pd.Series[Any] = Xf.mean(axis=0)
        raw_stds: pd.Series[Any] = Xf.std(axis=0, ddof=0)
        self.global_stds_: pd.Series[Any] = _apply_floor(raw_stds, self.std_floor)

        # Mode assignment and per-mode stats
        if self.n_modes == 1:
            self.mode_centers_: list[list[float]] | None = None
            labels = np.zeros(len(X), dtype=np.int64)
        else:
            kmeans: KMeans = KMeans(
                n_clusters=self.n_modes,
                n_init=10,
                random_state=self.random_state,
            )
            kmeans.fit(X[op_cols].astype(np.float64))
            self.mode_centers_ = kmeans.cluster_centers_.tolist()
            labels = kmeans.labels_.astype(np.int64)

        self.mode_means_: dict[int, pd.Series[Any]] = {}
        self.mode_stds_: dict[int, pd.Series[Any]] = {}

        if self.sensor_feature_cols_:
            Xs = X[self.sensor_feature_cols_].astype(np.float64)
            for mode_idx in range(self.n_modes):
                mask = labels == mode_idx
                if mask.sum() == 0:
                    continue
                subset = Xs.loc[mask]
                self.mode_means_[mode_idx] = subset.mean(axis=0)
                raw_mode_stds: pd.Series[Any] = subset.std(axis=0, ddof=0)
                self.mode_stds_[mode_idx] = _apply_floor(raw_mode_stds, self.std_floor)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize ``X`` using the fitted per-mode statistics.

        Args:
            X: DataFrame to normalize.  Must contain the columns in
                ``op_cols_``; any column in ``feature_cols_`` that is absent
                from ``X`` is silently skipped.

        Returns:
            A copy of ``X`` with feature columns normalized.

        Raises:
            RuntimeError: If the normalizer has not been fitted yet.
            KeyError: If any column in ``op_cols_`` is absent from ``X``.
        """
        if not hasattr(self, "feature_cols_"):
            raise RuntimeError(
                "This OperatingModeNormalizer instance is not fitted yet. "
                "Call fit() before transform()."
            )
        missing_op = [c for c in self.op_cols_ if c not in X.columns]
        if missing_op:
            raise KeyError(f"op_cols not found in X: {missing_op}")

        result = X.copy()

        # Cast feature cols present in X to float64
        cols_in_x = [c for c in self.feature_cols_ if c in X.columns]
        for col in cols_in_x:
            result[col] = result[col].astype(np.float64)

        modes = self._assign_modes(X)
        op_set = set(self.op_cols_)

        # Normalize sensor_feature_cols_ per mode
        for col in self.sensor_feature_cols_:
            if col not in X.columns:
                continue
            col_result = result[col].copy().astype(np.float64)
            for mode_idx in np.unique(modes):
                mask = modes == mode_idx
                if mode_idx in self.mode_means_:
                    mean_val = float(self.mode_means_[mode_idx][col])
                    std_val = float(self.mode_stds_[mode_idx][col])
                else:
                    mean_val = float(self.global_means_[col])
                    std_val = float(self.global_stds_[col])
                col_result.iloc[np.where(mask)[0]] = (
                    col_result.iloc[np.where(mask)[0]] - mean_val
                ) / std_val
            result[col] = col_result

        # Normalize op_cols that are in feature_cols_ globally
        for col in self.feature_cols_:
            if col in op_set and col in X.columns:
                mean_val = float(self.global_means_[col])
                std_val = float(self.global_stds_[col])
                result[col] = (result[col].astype(np.float64) - mean_val) / std_val

        return result

    def _assign_modes(self, X: pd.DataFrame) -> npt.NDArray[np.int64]:
        """Assign each row in ``X`` to a mode index.

        For single-mode normalizers every row is assigned to mode 0.  For
        multi-mode normalizers the nearest cluster centre (Euclidean distance
        on ``op_cols_``) is used.

        Args:
            X: DataFrame with at least the ``op_cols_`` columns present.

        Returns:
            Integer array of shape ``(len(X),)`` with mode indices.
        """
        if self.n_modes == 1:
            return np.zeros(len(X), dtype=np.int64)
        centers = np.array(self.mode_centers_, dtype=np.float64)
        op_vals = X[self.op_cols_].astype(np.float64).values
        diffs = op_vals[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diffs**2).sum(axis=2))
        result_indices: npt.NDArray[np.int64] = dists.argmin(axis=1).astype(np.int64)
        return result_indices

    def to_payload(self) -> dict[str, object]:
        """Serialise the fitted normalizer to a JSON-compatible dictionary.

        Returns:
            Dictionary with all fitted statistics and hyper-parameters needed
            to reconstruct this normalizer via :meth:`from_payload`.

        Raises:
            RuntimeError: If the normalizer has not been fitted yet.
        """
        if not hasattr(self, "feature_cols_"):
            raise RuntimeError(
                "This OperatingModeNormalizer instance is not fitted yet. "
                "Call fit() before to_payload()."
            )
        return {
            "schema_version": 1,
            "normalizer_type": "operating_mode",
            "feature_cols": list(self.feature_cols_),
            "op_cols": list(self.op_cols_),
            "sensor_feature_cols": list(self.sensor_feature_cols_),
            "n_modes": self.n_modes,
            "std_floor": self.std_floor,
            "random_state": self.random_state,
            "mode_centers": self.mode_centers_,
            "global_means": {str(k): float(v) for k, v in self.global_means_.items()},
            "global_stds": {str(k): float(v) for k, v in self.global_stds_.items()},
            "mode_means": {
                str(mode_idx): {str(k): float(v) for k, v in s.items()}
                for mode_idx, s in self.mode_means_.items()
            },
            "mode_stds": {
                str(mode_idx): {str(k): float(v) for k, v in s.items()}
                for mode_idx, s in self.mode_stds_.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Self:
        """Reconstruct a fitted normalizer from a serialised payload.

        Args:
            payload: Dictionary produced by :meth:`to_payload`.

        Returns:
            A fully-fitted :class:`OperatingModeNormalizer` ready to call
            :meth:`transform` on.

        Note:
            The ``std_floor`` stored in the payload is a *threshold*: any
            standard deviation at or below it was replaced with ``1.0``
            (not with ``std_floor``) during fitting.

        Raises:
            ValueError: For unsupported ``schema_version``, missing required
                keys, invalid hyper-parameter types or values, non-numeric
                stat values, corrupt ``mode_centers``, or mismatched
                feature/stat keys.
        """
        for key in _REQUIRED_PAYLOAD_KEYS:
            _require_key(payload, key)

        schema_version = payload["schema_version"]
        if schema_version != 1:
            raise ValueError(
                f"Unsupported schema_version: {schema_version!r}. "
                "Only version 1 is supported."
            )

        n_modes = payload["n_modes"]
        if not isinstance(n_modes, int) or n_modes <= 0:
            raise ValueError(f"n_modes must be a positive integer, got {n_modes!r}")

        std_floor = payload["std_floor"]
        try:
            std_floor_f = float(std_floor)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"std_floor must be a non-negative number, got {std_floor!r}"
            ) from exc
        if std_floor_f < 0:
            raise ValueError(f"std_floor must be non-negative, got {std_floor_f}")

        random_state = payload["random_state"]

        raw_centers = payload["mode_centers"]
        if raw_centers is not None:
            if not isinstance(raw_centers, (list, tuple)):
                raise ValueError(
                    "payload 'mode_centers' must be None or a list of lists."
                )
            for i, center in enumerate(raw_centers):
                if not isinstance(center, (list, tuple)) or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in center
                ):
                    raise ValueError(
                        f"payload 'mode_centers'[{i}] must be a list of numbers."
                    )

        obj = cls.__new__(cls)
        # Bypass __init__ validation; set params manually
        obj.feature_cols = list(payload["feature_cols"])
        obj.op_cols = list(payload["op_cols"])
        obj.n_modes = n_modes
        obj.std_floor = std_floor_f
        obj.random_state = random_state

        obj.feature_cols_ = list(payload["feature_cols"])
        obj.op_cols_ = list(payload["op_cols"])
        obj.sensor_feature_cols_ = list(payload["sensor_feature_cols"])
        obj.mode_centers_ = payload["mode_centers"]

        obj.global_means_ = _series_from_mapping(
            payload["global_means"], "global_means"
        )
        obj.global_stds_ = _series_from_mapping(
            payload["global_stds"], "global_stds"
        )

        obj.mode_means_ = {
            int(mode_idx): _series_from_mapping(s, f"mode_means[{mode_idx}]")
            for mode_idx, s in payload["mode_means"].items()
        }
        obj.mode_stds_ = {
            int(mode_idx): _series_from_mapping(s, f"mode_stds[{mode_idx}]")
            for mode_idx, s in payload["mode_stds"].items()
        }

        return obj
