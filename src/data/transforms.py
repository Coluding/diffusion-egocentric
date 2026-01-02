import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from typing import Tuple, Optional
import numpy as np
from pathlib import Path


class VideoTransforms:
    """Video preprocessing utilities for Cosmos training."""

    @staticmethod
    def load_video(video_path: str, max_frames: Optional[int] = None) -> torch.Tensor:
        """
        Load video from file.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to load (None = all)

        Returns:
            Video tensor [T, C, H, W] in range [0, 1]
        """
        try:
            from torchvision.io import read_video
            video, audio, info = read_video(video_path, pts_unit='sec')
            # video is [T, H, W, C], convert to [T, C, H, W]
            video = video.permute(0, 3, 1, 2).float() / 255.0

            if max_frames is not None and video.shape[0] > max_frames:
                video = video[:max_frames]

            return video
        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_path}: {e}")

    @staticmethod
    def spatial_resize(video: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """
        Resize video spatially to (H, W).

        Args:
            video: [T, C, H, W]
            size: Target (height, width)

        Returns:
            Resized video [T, C, H', W']
        """
        T, C, H, W = video.shape
        target_h, target_w = size

        # Resize each frame
        resized_frames = []
        for t in range(T):
            frame = video[t]  # [C, H, W]
            resized = F.resize(frame, [target_h, target_w], antialias=True)
            resized_frames.append(resized)

        return torch.stack(resized_frames, dim=0)

    @staticmethod
    def temporal_resample(video: torch.Tensor, target_fps: int, original_fps: int) -> torch.Tensor:
        """
        Resample video to target FPS.

        Args:
            video: [T, C, H, W]
            target_fps: Target frames per second
            original_fps: Original video FPS

        Returns:
            Resampled video [T', C, H, W]
        """
        if original_fps == target_fps:
            return video

        T = video.shape[0]
        duration = T / original_fps
        target_frames = int(duration * target_fps)

        # Linear interpolation indices
        indices = torch.linspace(0, T - 1, target_frames)
        indices_floor = indices.floor().long()
        indices_ceil = indices.ceil().long()
        weights = indices - indices_floor.float()

        # Interpolate
        resampled = []
        for i, (idx_floor, idx_ceil, weight) in enumerate(zip(indices_floor, indices_ceil, weights)):
            if idx_floor == idx_ceil:
                resampled.append(video[idx_floor])
            else:
                frame = (1 - weight) * video[idx_floor] + weight * video[idx_ceil]
                resampled.append(frame)

        return torch.stack(resampled, dim=0)

    @staticmethod
    def normalize_for_vae(video: torch.Tensor) -> torch.Tensor:
        """
        Normalize video for VAE input.

        Cosmos VAE expects input in range [-1, 1].

        Args:
            video: [T, C, H, W] in range [0, 1]

        Returns:
            Normalized video [-1, 1]
        """
        return video * 2.0 - 1.0

    @staticmethod
    def denormalize_from_vae(video: torch.Tensor) -> torch.Tensor:
        """
        Denormalize VAE output to [0, 1].

        Args:
            video: [T, C, H, W] in range [-1, 1]

        Returns:
            Video in range [0, 1]
        """
        return (video + 1.0) / 2.0

    @staticmethod
    def pad_or_crop_temporal(
        video: torch.Tensor,
        target_frames: int,
        mode: str = "random"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad or crop video to target number of frames.

        Args:
            video: [T, C, H, W]
            target_frames: Target number of frames
            mode: "random" for random crop, "center" for center crop

        Returns:
            Tuple of (video [target_frames, C, H, W], mask [target_frames])
        """
        T = video.shape[0]

        if T == target_frames:
            return video, torch.ones(T, dtype=torch.bool)

        elif T > target_frames:
            # Crop
            if mode == "random":
                start = torch.randint(0, T - target_frames + 1, (1,)).item()
            else:  # center
                start = (T - target_frames) // 2

            cropped = video[start:start + target_frames]
            mask = torch.ones(target_frames, dtype=torch.bool)
            return cropped, mask

        else:
            # Pad with zeros
            pad_frames = target_frames - T
            padding = torch.zeros(pad_frames, *video.shape[1:], dtype=video.dtype)
            padded = torch.cat([video, padding], dim=0)

            mask = torch.cat([
                torch.ones(T, dtype=torch.bool),
                torch.zeros(pad_frames, dtype=torch.bool)
            ])

            return padded, mask
