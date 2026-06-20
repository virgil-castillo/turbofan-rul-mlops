"""Shared official test set evaluation for sequence models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbofan.models import evaluate


def align_labels_to_eligible_engines(
    metadata: pd.DataFrame,
    rul_labels: pd.Series,
) -> pd.Series:
    """Align official RUL labels to eligible sequence test engines.

    C-MAPSS official RUL labels are ordered by engine ID, while final
    sequence windows can skip engines shorter than the window size. This
    selects labels for the eligible engine IDs before applying the
    standard count check.

    Args:
        metadata: Final-window metadata containing eligible ``engine_id``
            rows.
        rul_labels: Official RUL labels in full test engine order.

    Returns:
        Float RUL Series aligned to ``metadata``.

    Raises:
        ValueError: If an eligible engine ID cannot be mapped to a label
            row.
    """
    engine_ids = metadata["engine_id"].to_numpy(dtype=np.int64)
    label_positions = engine_ids - 1
    if np.any(label_positions < 0) or np.any(label_positions >= len(rul_labels)):
        raise ValueError(
            "Official RUL labels must include a row for every eligible test engine."
        )

    eligible_labels = rul_labels.iloc[label_positions].reset_index(drop=True)
    return evaluate.align_official_test_labels(
        metadata.reset_index(drop=True), eligible_labels
    )
