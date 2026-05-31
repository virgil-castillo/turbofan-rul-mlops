"""Tests for PyTorch sequence dataset helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from turbofan.sequences.dataset import SequenceDataset, build_sequence_loader
from turbofan.sequences.windowing import WindowedSequences


def _windows() -> WindowedSequences:
    """Build tiny sequence windows.

    Returns:
        Sequence windows with four samples, two timesteps, and two features.
    """
    return WindowedSequences(
        X=np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ],
            dtype=np.float64,
        ),
        y=np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        metadata=pd.DataFrame(
            {
                "engine_id": [1, 1, 2, 2],
                "cycle": [2, 3, 2, 3],
            }
        ),
        lengths=np.full(4, 2, dtype=np.int64),
    )


def test_sequence_dataset_returns_float32_tensors() -> None:
    """Dataset returns feature and scalar target tensors as float32."""
    dataset = SequenceDataset(_windows())

    features, target = dataset[0]

    assert features.shape == (2, 2)
    assert target.shape == torch.Size([])
    assert features.dtype == torch.float32
    assert target.dtype == torch.float32
    torch.testing.assert_close(
        features,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    )
    assert target.item() == 10.0


def test_sequence_dataset_len_returns_number_of_windows() -> None:
    """Dataset length equals the number of sequence windows."""
    dataset = SequenceDataset(_windows())

    assert len(dataset) == 4


def test_build_sequence_loader_preserves_order_when_shuffle_is_false() -> None:
    """Evaluation loader keeps windows in their original order."""
    loader = build_sequence_loader(_windows(), batch_size=1, shuffle=False)

    targets = [float(target.item()) for _, target in loader]

    assert targets == [10.0, 20.0, 30.0, 40.0]


def test_build_sequence_loader_respects_batch_size_when_shuffle_is_true() -> None:
    """Training loader uses the configured batch size."""
    loader = build_sequence_loader(_windows(), batch_size=3, shuffle=True)

    batch_features, batch_targets = next(iter(loader))

    assert batch_features.shape == (3, 2, 2)
    assert batch_targets.shape == (3,)
