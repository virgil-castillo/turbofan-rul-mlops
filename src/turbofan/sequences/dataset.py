"""PyTorch dataset and loader for windowed turbofan sequences."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from turbofan.sequences.windowing import WindowedSequences

SequenceItem = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
SequenceBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class SequenceDataset(Dataset[SequenceItem]):
    """PyTorch dataset wrapping windowed turbofan sequences.

    Each item is a ``(features, target, length)`` 3-tuple. ``length`` is the
    number of real (un-padded) timesteps in the window, suitable for
    ``torch.nn.utils.rnn.pack_padded_sequence``.

    Args:
        windows: Windowed sequence features, labels, lengths, and metadata.
    """

    def __init__(self, windows: WindowedSequences) -> None:
        """Initialize tensors from windowed sequence arrays.

        Args:
            windows: Windowed sequence features, labels, and lengths to expose
                through the dataset interface.
        """
        self._features = torch.as_tensor(windows.X, dtype=torch.float32)
        self._targets = torch.as_tensor(windows.y, dtype=torch.float32)
        self._lengths = torch.as_tensor(windows.lengths, dtype=torch.int64)

    def __len__(self) -> int:
        """Return the number of sequence windows.

        Returns:
            Number of windows in the dataset.
        """
        return int(self._features.shape[0])

    def __getitem__(self, index: int) -> SequenceItem:
        """Return one feature window, target, and real sequence length.

        Args:
            index: Dataset row index.

        Returns:
            3-tuple of feature tensor ``(window_size, n_features)``, scalar
            target tensor, and scalar int64 length tensor.
        """
        return self._features[index], self._targets[index], self._lengths[index]


def build_sequence_loader(
    windows: WindowedSequences,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[SequenceItem]:
    """Build a DataLoader that yields ``(features, targets, lengths)`` batches.

    Args:
        windows: Windowed sequences to wrap.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle indices each epoch.

    Returns:
        DataLoader yielding 3-tuple batches.
    """
    dataset = SequenceDataset(windows)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
