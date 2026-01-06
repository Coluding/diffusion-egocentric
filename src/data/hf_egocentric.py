from datasets import load_dataset
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from .base import VideoDatasetBase
from .latent_cache import LatentCache
from .transforms import VideoTransforms


class EgocentricDataset(VideoDatasetBase):
    """
    Adapter for builddotai/Egocentric-100K from Hugging Face.

    Transforms HF dataset samples into canonical format with cached latents.
    """

    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[Path] = None,
        max_frames: int = 64,
        fps: int = 8,
        resolution: tuple = (256, 456),
        streaming: bool = True,
        **hf_kwargs
    ):
        """
        Args:
            split: Dataset split ("train" or "validation")
            cache_dir: Path to latent cache (required for training)
            max_frames: Maximum frames per clip
            fps: Target FPS for temporal resampling
            resolution: Target (height, width)
            streaming: Whether to stream from HF
            **hf_kwargs: Additional arguments for load_dataset
        """
        self.split = split
        self.max_frames = max_frames
        self.target_fps = fps
        self.resolution = resolution

        # Load HF dataset
        self.hf_dataset = load_dataset(
            "builddotai/Egocentric-100K",
            split=split,
            streaming=streaming,
            **hf_kwargs
        )

        # Convert streaming dataset to list for indexing (if not streaming)
        if not streaming:
            self.hf_dataset = list(self.hf_dataset)

        # Initialize latent cache
        self.cache = LatentCache(cache_dir) if cache_dir else None
        if self.cache is None and split == "train":
            raise ValueError("cache_dir is required for training split")

        # Get dataset length
        if streaming:
            # For streaming, we need to iterate once to count (expensive)
            # Or rely on dataset info if available
            try:
                self._length = self.hf_dataset.info.splits[split].num_examples
            except:
                # Fallback: assume large dataset
                self._length = 100000  # Egocentric-100K approximate size
        else:
            self._length = len(self.hf_dataset)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return sample in canonical format.

        Returns:
            {
                "latents": [T, C, H, W],
                "mask": [T],
                "metadata": {video_id, worker_id, factory_id, fps, duration}
            }
        """
        # Get HF sample
        sample = self.hf_dataset[idx]

        # Extract video ID (used as cache key)
        video_id = sample.get("video_id", f"video_{idx}")

        # Check cache
        if self.cache and self.cache.has(video_id):
            latents = self.cache.get(video_id)
        else:
            # Handled by collate fn
            return None

        # Temporal processing: pad/crop to max_frames
        latents, mask = VideoTransforms.pad_or_crop_temporal(
            latents,
            self.max_frames,
            mode="random" if self.split == "train" else "center"
        )

        # Build metadata
        metadata = {
            "video_id": video_id,
            "worker_id": sample.get("worker_id", "unknown"),
            "factory_id": sample.get("factory_id", "unknown"),
            "fps": self.target_fps,
            "duration": len(latents) / self.target_fps,
        }

        return {
            "latents": latents,
            "mask": mask,
            "metadata": metadata
        }

    def get_collate_fn(self):
        def collate(batch):
            batch = [b for b in batch if b is not None]

            if len(batch) == 0:
                return None  

            return {
                "latents": torch.stack([b["latents"] for b in batch]),
                "mask": torch.stack([b["mask"] for b in batch]),
                "metadata": [b["metadata"] for b in batch]
            }
        return collate

