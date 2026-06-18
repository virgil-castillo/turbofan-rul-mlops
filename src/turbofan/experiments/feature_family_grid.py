"""Grid definitions for the sequence feature-family screen."""
from __future__ import annotations

from dataclasses import dataclass

from turbofan.config.schema import FeatureFamilyName


@dataclass(frozen=True)
class ScreenCell:
    """One cell in the feature-family sweep grid.

    A cell fully specifies a single training run: architecture, C-MAPSS subset,
    feature configuration, rolling window or lag step when applicable, sequence
    window, and random seed.

    Args:
        architecture: Sequence model architecture, ``"gru"`` or ``"lstm"``.
        subset: C-MAPSS fault dataset subset, e.g. ``"FD001"``.
        feature_config: Stable feature-configuration label.
        feature_families: Ordered feature families passed to the pipeline.
        rolling_window: Rolling window size when applicable, else ``None``.
        lag_step: Lag step when applicable, else ``None``.
        sequence_window: Number of cycles per sequence window.
        seed: Random seed for model initialization and training.
    """

    architecture: str
    subset: str
    feature_config: str
    feature_families: list[FeatureFamilyName]
    rolling_window: int | None
    lag_step: int | None
    sequence_window: int
    seed: int

    def __post_init__(self) -> None:
        """Validate that feature_families is stored as a list."""
        object.__setattr__(self, "feature_families", list(self.feature_families))


_FEATURE_CONFIGS: list[tuple[str, list[FeatureFamilyName], str]] = [
    ("raw", ["raw"], "none"),
    ("raw+rolling_mean", ["raw", "rolling_mean"], "rolling_window"),
    ("raw+rolling_std", ["raw", "rolling_std"], "rolling_window"),
    ("raw+rolling_min", ["raw", "rolling_min"], "rolling_window"),
    ("raw+rolling_max", ["raw", "rolling_max"], "rolling_window"),
    ("raw+rolling_slope", ["raw", "rolling_slope"], "rolling_window"),
    ("raw+rolling_delta", ["raw", "rolling_delta"], "rolling_window"),
    ("raw+lag", ["raw", "lag"], "lag_step"),
]


def enumerate_cells(
    architectures: list[str],
    subsets: list[str],
    sequence_windows: list[int],
    rolling_windows: list[int],
    lag_steps: list[int],
    seeds: list[int],
) -> list[ScreenCell]:
    """Enumerate every cell of the feature-family sweep grid.

    For each architecture, subset, sequence window, and seed, emits one raw
    cell, one cell per rolling-window value for every rolling feature family,
    and one cell per lag-step value for the lag feature family.

    Args:
        architectures: Architecture names (``"gru"`` or ``"lstm"``).
        subsets: C-MAPSS subset identifiers.
        sequence_windows: Sequence window sizes.
        rolling_windows: Rolling window sizes for rolling feature configs.
        lag_steps: Lag steps for the lag feature config.
        seeds: Model-init/training random seeds.

    Returns:
        List of screen cells, one per grid point.
    """
    cells: list[ScreenCell] = []
    for arch in architectures:
        for subset in subsets:
            for sw in sequence_windows:
                for seed in seeds:
                    for label, families, swept in _FEATURE_CONFIGS:
                        if swept == "none":
                            cells.append(
                                ScreenCell(
                                    architecture=arch,
                                    subset=subset,
                                    feature_config=label,
                                    feature_families=families,
                                    rolling_window=None,
                                    lag_step=None,
                                    sequence_window=sw,
                                    seed=seed,
                                )
                            )
                        elif swept == "rolling_window":
                            for rw in rolling_windows:
                                cells.append(
                                    ScreenCell(
                                        architecture=arch,
                                        subset=subset,
                                        feature_config=label,
                                        feature_families=families,
                                        rolling_window=rw,
                                        lag_step=None,
                                        sequence_window=sw,
                                        seed=seed,
                                    )
                                )
                        else:
                            for ls in lag_steps:
                                cells.append(
                                    ScreenCell(
                                        architecture=arch,
                                        subset=subset,
                                        feature_config=label,
                                        feature_families=families,
                                        rolling_window=None,
                                        lag_step=ls,
                                        sequence_window=sw,
                                        seed=seed,
                                    )
                                )
    return cells


def cell_key(cell: ScreenCell) -> tuple[str, str, str, str, str]:
    """Return the five-string resume identity key for a cell.

    Inapplicable ``None`` factors are rendered as empty strings so keys compare
    equal to values parsed back from CSV text.

    Args:
        cell: The screen cell to compute the key for.

    Returns:
        Tuple of feature config, rolling window, lag step, sequence window, and
        seed, all rendered as strings.
    """
    rw = "" if cell.rolling_window is None else str(cell.rolling_window)
    ls = "" if cell.lag_step is None else str(cell.lag_step)
    return (
        cell.feature_config,
        rw,
        ls,
        str(cell.sequence_window),
        str(cell.seed),
    )
