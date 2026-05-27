"""Operating-mode-aware normalization for C-MAPSS turbofan data."""
from __future__ import annotations

CMAPSS_SUBSET_MODE_COUNTS: dict[str, int] = {
    "FD001": 1,
    "FD002": 6,
    "FD003": 1,
    "FD004": 6,
}


def mode_count_for_subset(fd_subset: str) -> int:
    """Return the EDA-confirmed operating-mode count for a C-MAPSS subset.

    Args:
        fd_subset: C-MAPSS subset name, e.g. ``"FD001"``.

    Returns:
        Number of operating modes for the subset.

    Raises:
        ValueError: If ``fd_subset`` is not a supported C-MAPSS subset.
    """
    if fd_subset not in CMAPSS_SUBSET_MODE_COUNTS:
        supported = sorted(CMAPSS_SUBSET_MODE_COUNTS)
        raise ValueError(
            f"Unsupported C-MAPSS subset: {fd_subset!r}. "
            f"Supported subsets: {supported}."
        )
    return CMAPSS_SUBSET_MODE_COUNTS[fd_subset]
