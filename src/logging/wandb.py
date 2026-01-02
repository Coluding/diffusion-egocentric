import wandb
from typing import Optional, Dict, Any
import torch


class WandBLogger:
    """
    Weights & Biases logging wrapper.
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        enabled: bool = True
    ):
        """
        Args:
            project: W&B project name
            entity: W&B entity (team/user)
            run_name: Run name
            config: Configuration dict to log
            enabled: Whether to enable logging
        """
        self.enabled = enabled

        if self.enabled:
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config
            )
            print(f"Initialized W&B: {project}/{run_name}")
        else:
            print("W&B logging disabled")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to W&B.

        Args:
            metrics: Dict of metric_name -> value
            step: Optional step number
        """
        if not self.enabled:
            return

        wandb.log(metrics, step=step)

    def log_video(
        self,
        video: torch.Tensor,
        name: str = "video",
        fps: int = 8,
        step: Optional[int] = None
    ):
        """
        Log video to W&B.

        Args:
            video: Video tensor [T, C, H, W] in range [0, 1]
            name: Video name/key
            fps: Frames per second
            step: Optional step number
        """
        if not self.enabled:
            return

        # Convert to numpy [T, H, W, C] in range [0, 255]
        video_np = (video * 255).clamp(0, 255).to(torch.uint8)
        video_np = video_np.permute(0, 2, 3, 1).cpu().numpy()

        # Log to W&B
        wandb.log({
            name: wandb.Video(video_np, fps=fps, format="mp4")
        }, step=step)

    def log_videos(
        self,
        videos: torch.Tensor,
        name: str = "videos",
        fps: int = 8,
        step: Optional[int] = None,
        max_videos: int = 4
    ):
        """
        Log multiple videos to W&B.

        Args:
            videos: Video batch [B, T, C, H, W]
            name: Base name for videos
            fps: Frames per second
            step: Optional step number
            max_videos: Maximum number of videos to log
        """
        if not self.enabled:
            return

        num_videos = min(len(videos), max_videos)

        for i in range(num_videos):
            self.log_video(
                videos[i],
                name=f"{name}_{i}",
                fps=fps,
                step=step
            )

    def finish(self):
        """Finish W&B run."""
        if self.enabled:
            wandb.finish()
