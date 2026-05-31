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

    features, target, _length = dataset[0]

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

    targets = [float(target.item()) for _, target, _length in loader]

    assert targets == [10.0, 20.0, 30.0, 40.0]


def test_build_sequence_loader_respects_batch_size_when_shuffle_is_true() -> None:
    """Training loader uses the configured batch size."""
    loader = build_sequence_loader(_windows(), batch_size=3, shuffle=True)

    batch_features, batch_targets, batch_lengths = next(iter(loader))

    assert batch_features.shape == (3, 2, 2)
    assert batch_targets.shape == (3,)


def _windows_with_lengths() -> WindowedSequences:
    X = np.zeros((3, 4, 2), dtype=np.float32)
    X[0, -2:, :] = 1.0  # length 2
    X[1] = 1.0          # length 4
    X[2, -3:, :] = 1.0  # length 3
    y = np.asarray([5.0, 4.0, 6.0], dtype=np.float32)
    lengths = np.asarray([2, 4, 3], dtype=np.int64)
    metadata = pd.DataFrame(
        {"engine_id": [1, 2, 3], "cycle": [10, 20, 30], "padded": [True, False, True]}
    )
    return WindowedSequences(X=X, y=y, metadata=metadata, lengths=lengths)


def test_sequence_dataset_returns_three_tuple_with_lengths() -> None:
    dataset = SequenceDataset(_windows_with_lengths())
    item = dataset[0]
    assert len(item) == 3
    features, target, length = item
    assert features.shape == (4, 2)
    assert float(target) == 5.0
    assert int(length) == 2


def test_build_sequence_loader_yields_three_tuple_batches() -> None:
    loader = build_sequence_loader(
        _windows_with_lengths(), batch_size=2, shuffle=False
    )
    batches = list(loader)
    assert len(batches) == 2
    first = batches[0]
    assert len(first) == 3
    features, targets, lengths = first
    assert features.shape == (2, 4, 2)
    assert targets.shape == (2,)
    assert lengths.dtype == torch.int64
    assert lengths.tolist() == [2, 4]
