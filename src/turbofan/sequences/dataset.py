"""PyTorch Dataset helpers for sequence windows."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from turbofan.sequences.windowing import WindowedSequences


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset wrapping windowed turbofan sequences.

    Args:
        windows: Windowed sequence features, labels, and metadata.
    """

    def __init__(self, windows: WindowedSequences) -> None:
        """Initialize tensors from windowed sequence arrays.

        Args:
            windows: Windowed sequence features and labels to expose through
                the dataset interface.
        """
        self._features = torch.as_tensor(windows.X, dtype=torch.float32)
        self._targets = torch.as_tensor(windows.y, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of sequence windows.

        Returns:
            Number of windows in the dataset.
        """
        return int(self._features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one feature window and target.

        Args:
            index: Dataset row index.

        Returns:
            Feature tensor with shape ``(window_size, n_features)`` and scalar
            target tensor.
        """
        return self._features[index], self._targets[index]


def build_sequence_loader(
    windows: WindowedSequences,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a PyTorch DataLoader for sequence windows.

    Args:
        windows: Windowed sequence features and labels.
        batch_size: Number of sequence windows per batch.
        shuffle: Whether to shuffle windows between epochs.

    Returns:
        DataLoader yielding batches of feature and target tensors.
    """
    return DataLoader(
        SequenceDataset(windows),
        batch_size=batch_size,
        shuffle=shuffle,
    )
