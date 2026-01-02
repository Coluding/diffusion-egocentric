import torch
import torchvision
from pathlib import Path
from typing import Union, List
import numpy as np


class VideoUtils:
    """
    Utilities for saving and visualizing videos.
    """

    @staticmethod
    def save_video_mp4(
        video: torch.Tensor,
        save_path: Union[str, Path],
        fps: int = 8,
        quality: int = 5
    ):
        """
        Save video tensor as MP4 file.

        Args:
            video: Video tensor [T, C, H, W] in range [0, 1]
            save_path: Output path
            fps: Frames per second
            quality: Video quality (1-10, higher is better)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to uint8 [0, 255]
        video_uint8 = (video * 255).clamp(0, 255).to(torch.uint8)

        # torchvision expects [T, H, W, C]
        video_uint8 = video_uint8.permute(0, 2, 3, 1).cpu()

        # Save video
        torchvision.io.write_video(
            str(save_path),
            video_uint8,
            fps=fps,
            options={"crf": str(51 - quality * 5)}  # Lower CRF = higher quality
        )

    @staticmethod
    def save_video_grid(
        videos: torch.Tensor,
        save_path: Union[str, Path],
        nrow: int = 4,
        fps: int = 8
    ):
        """
        Save grid of videos as single MP4.

        Args:
            videos: Video batch [B, T, C, H, W] in range [0, 1]
            save_path: Output path
            nrow: Number of videos per row in grid
            fps: Frames per second
        """
        B, T, C, H, W = videos.shape

        # Create grid for each frame
        grid_frames = []
        for t in range(T):
            frame_batch = videos[:, t]  # [B, C, H, W]
            grid = torchvision.utils.make_grid(frame_batch, nrow=nrow, padding=2)
            grid_frames.append(grid)

        # Stack frames: [T, C, H_grid, W_grid]
        grid_video = torch.stack(grid_frames, dim=0)

        # Save
        VideoUtils.save_video_mp4(grid_video, save_path, fps=fps)

    @staticmethod
    def tensor_to_numpy(video: torch.Tensor) -> np.ndarray:
        """
        Convert video tensor to numpy array.

        Args:
            video: [T, C, H, W] in range [0, 1]

        Returns:
            Numpy array [T, H, W, C] in range [0, 255] uint8
        """
        video_np = (video * 255).clamp(0, 255).to(torch.uint8)
        video_np = video_np.permute(0, 2, 3, 1).cpu().numpy()
        return video_np

    @staticmethod
    def save_frames(
        video: torch.Tensor,
        save_dir: Union[str, Path],
        prefix: str = "frame"
    ):
        """
        Save individual frames as images.

        Args:
            video: [T, C, H, W] in range [0, 1]
            save_dir: Output directory
            prefix: Filename prefix
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for t, frame in enumerate(video):
            frame_path = save_dir / f"{prefix}_{t:04d}.png"
            torchvision.utils.save_image(frame, frame_path)

    @staticmethod
    def create_video_comparison(
        videos_dict: dict,
        save_path: Union[str, Path],
        fps: int = 8
    ):
        """
        Create side-by-side comparison of multiple videos.

        Args:
            videos_dict: Dict of {label: video_tensor [T, C, H, W]}
            save_path: Output path
            fps: Frames per second
        """
        # Stack videos horizontally
        videos = list(videos_dict.values())
        labels = list(videos_dict.keys())

        # Ensure all videos have same length
        min_t = min(v.shape[0] for v in videos)
        videos = [v[:min_t] for v in videos]

        # Concatenate horizontally
        combined = torch.cat(videos, dim=3)  # Concat along width

        VideoUtils.save_video_mp4(combined, save_path, fps=fps)

    @staticmethod
    def normalize_for_display(video: torch.Tensor) -> torch.Tensor:
        """
        Normalize video tensor to [0, 1] for display.

        Handles both [-1, 1] and [0, 1] input ranges.

        Args:
            video: Video tensor

        Returns:
            Normalized video [0, 1]
        """
        vmin = video.min()
        vmax = video.max()

        if vmin < 0:
            # Assume [-1, 1] range
            return (video + 1.0) / 2.0
        else:
            # Assume [0, 1] range
            return video.clamp(0, 1)
