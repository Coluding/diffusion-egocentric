from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict, Any
import torch


class TensorBoardLogger:
    """
    TensorBoard logging wrapper.
    """

    def __init__(
        self,
        log_dir: str,
        enabled: bool = True
    ):
        """
        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether to enable logging
        """
        self.enabled = enabled

        if self.enabled:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            print(f"Initialized TensorBoard: {log_dir}")
        else:
            self.writer = None
            print("TensorBoard logging disabled")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to TensorBoard.

        Args:
            metrics: Dict of metric_name -> value
            step: Global step number
        """
        if not self.enabled:
            return

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

    def log_video(
        self,
        video: torch.Tensor,
        name: str = "video",
        fps: int = 8,
        step: Optional[int] = None
    ):
        """
        Log video to TensorBoard.

        Args:
            video: Video tensor [T, C, H, W] in range [0, 1]
            name: Video tag
            fps: Frames per second
            step: Global step
        """
        if not self.enabled:
            return

        # Add batch dimension: [1, T, C, H, W]
        video = video.unsqueeze(0)

        # TensorBoard expects [N, T, C, H, W]
        self.writer.add_video(name, video, step, fps=fps)

    def log_videos(
        self,
        videos: torch.Tensor,
        name: str = "videos",
        fps: int = 8,
        step: Optional[int] = None,
        max_videos: int = 4
    ):
        """
        Log multiple videos to TensorBoard.

        Args:
            videos: Video batch [B, T, C, H, W]
            name: Base tag for videos
            fps: Frames per second
            step: Global step
            max_videos: Maximum videos to log
        """
        if not self.enabled:
            return

        num_videos = min(len(videos), max_videos)
        videos = videos[:num_videos]

        # Log as single batch or individual videos
        self.writer.add_video(name, videos, step, fps=fps)

    def close(self):
        """Close TensorBoard writer."""
        if self.enabled and self.writer:
            self.writer.close()
