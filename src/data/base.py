from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
import torch
from torch.utils.data import Dataset


class VideoDatasetBase(ABC, Dataset):
    """
    Base class for all video datasets.
    Enforces canonical output format for trainer compatibility.

    All implementations must return batches in the format:
    {
        "latents": FloatTensor[T, C, H, W],  # VAE-encoded latents
        "mask": BoolTensor[T],                # Frame validity mask
        "metadata": Dict[str, Any]            # Optional dataset-specific fields
    }
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return single sample in canonical format.

        Returns:
            Dictionary with keys:
                - latents: [T, C, H, W] tensor of VAE-encoded frames
                - mask: [T] boolean tensor indicating valid frames (vs padding)
                - metadata: Dict with dataset-specific information
        """
        pass

    @abstractmethod
    def get_collate_fn(self) -> Callable:
        """
        Return custom collate function for batching.

        The collate function should stack samples into batches:
        {
            "latents": [B, T, C, H, W],
            "mask": [B, T],
            "metadata": List[Dict]  # Length B
        }
        """
        pass
