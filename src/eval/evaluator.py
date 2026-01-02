import torch
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from ..diffusion.sampling import VideoSampler
from ..diffusion.scheduler import SchedulerFactory
from .video_utils import VideoUtils


class CosmosEvaluator:
    """
    Evaluation and video generation for trained Cosmos models.
    """

    def __init__(
        self,
        unet,
        vae,
        config: dict,
        device: str = "cuda"
    ):
        """
        Args:
            unet: Trained UNet model
            vae: VAE model
            config: Evaluation configuration
            device: Device for inference
        """
        self.unet = unet
        self.vae = vae
        self.config = config
        self.device = device

        # Create inference scheduler
        self.scheduler = SchedulerFactory.create_inference_scheduler(
            num_inference_steps=config.get("num_inference_steps", 50)
        )

        # Create sampler
        self.sampler = VideoSampler(
            unet=unet,
            vae=vae,
            scheduler=self.scheduler,
            device=device
        )

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int = 100,
        batch_size: int = 4,
        save_dir: Optional[Path] = None
    ) -> list:
        """
        Generate video samples.

        Args:
            num_samples: Total number of videos to generate
            batch_size: Batch size for generation
            save_dir: Directory to save videos (optional)

        Returns:
            List of generated video tensors
        """
        print(f"Generating {num_samples} video samples...")

        all_videos = []
        seeds = self.config.get("eval_seeds", [42])

        # Generate in batches
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Generating videos"):
            current_batch_size = min(batch_size, num_samples - len(all_videos))

            # Use different seed for each batch
            seed = seeds[batch_idx % len(seeds)] + batch_idx

            # Generate videos
            videos = self.sampler.sample(
                batch_size=current_batch_size,
                num_frames=self.config.get("num_frames", 64),
                height=self.config.get("height", 256),
                width=self.config.get("width", 456),
                guidance_scale=self.config.get("guidance_scale", 7.5),
                num_inference_steps=self.config.get("num_inference_steps", 50),
                seed=seed,
                return_latents=False
            )

            all_videos.append(videos)

            # Save individual videos if save_dir provided
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                for i, video in enumerate(videos):
                    video_idx = batch_idx * batch_size + i
                    video_path = save_dir / f"sample_{video_idx:04d}.mp4"
                    VideoUtils.save_video_mp4(
                        video,
                        video_path,
                        fps=self.config.get("fps", 8)
                    )

        # Concatenate all batches
        all_videos = torch.cat(all_videos, dim=0)[:num_samples]

        print(f"Generated {len(all_videos)} videos")

        return all_videos

    @torch.no_grad()
    def evaluate_checkpoint(
        self,
        output_dir: str,
        checkpoint_step: int = None
    ):
        """
        Full evaluation of loaded model checkpoint.

        Note: Checkpoint should be loaded into self.unet before calling this method.

        Args:
            output_dir: Output directory for results
            checkpoint_step: Optional step number for logging
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Evaluating model")
        print(f"Output directory: {output_dir}")

        # Generate samples
        num_samples = self.config.get("num_samples", 100)
        videos = self.generate_samples(
            num_samples=num_samples,
            batch_size=self.config.get("batch_size", 4),
            save_dir=output_dir / "videos"
        )

        # Save video grid
        grid_path = output_dir / "video_grid.mp4"
        VideoUtils.save_video_grid(
            videos[:min(16, len(videos))],  # First 16 videos
            grid_path,
            nrow=4,
            fps=self.config.get("fps", 8)
        )
        print(f"Saved video grid to {grid_path}")

        # Save raw tensors if requested
        if self.config.get("save_raw_tensors", False):
            torch.save(videos, output_dir / "generated_videos.pt")
            print(f"Saved raw tensors to {output_dir / 'generated_videos.pt'}")

        print("Evaluation complete!")

        result = {
            "num_samples": len(videos),
            "output_dir": str(output_dir)
        }
        if checkpoint_step is not None:
            result["checkpoint_step"] = checkpoint_step

        return result
