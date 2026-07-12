"""Tests for turbofan.evaluation.sequence_official."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.evaluation.sequence_official import align_labels_to_eligible_engines


class TestAlignLabelsToEligibleEngines:
    """Tests for align_labels_to_eligible_engines."""

    def test_selects_labels_for_eligible_engines(self) -> None:
        """Labels for eligible engine IDs are selected and aligned."""
        metadata = pd.DataFrame({"engine_id": [1, 3], "cycle": [10, 20]})
        rul_labels = pd.Series([100, 200, 300], name="rul")

        result = align_labels_to_eligible_engines(metadata, rul_labels)

        assert list(result) == [100.0, 300.0]
        assert result.name == "rul"
        assert result.dtype == np.float64

    def test_raises_on_out_of_range_engine_id(self) -> None:
        """Engine IDs beyond available labels raise ValueError."""
        metadata = pd.DataFrame({"engine_id": [1, 5], "cycle": [10, 20]})
        rul_labels = pd.Series([100, 200, 300], name="rul")

        with pytest.raises(ValueError, match="eligible test engine"):
            align_labels_to_eligible_engines(metadata, rul_labels)
