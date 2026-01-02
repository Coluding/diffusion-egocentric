import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import time

from .losses import DiffusionLoss
from .fsdp import FSDPConfig


class CosmosTrainer:
    """
    Main training loop for Cosmos-2B finetuning.

    Handles:
    - Training epoch iteration
    - Checkpoint saving/loading
    - Evaluation triggering
    - Distributed training coordination
    """

    def __init__(
        self,
        unet,
        vae,
        scheduler,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer,
        lr_scheduler,
        config: dict,
        logger=None,
        device: str = "cuda"
    ):
        """
        Args:
            unet: (FSDP-wrapped) UNet model
            vae: (FSDP-wrapped) VAE model
            scheduler: Noise scheduler
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
            config: Training configuration dict
            logger: Logger instance (wandb/tensorboard)
            device: Device for training
        """
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.logger = logger
        self.device = device

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Distributed training
        self.is_distributed = dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def train(self):
        """
        Main training loop.
        """
        num_epochs = self.config.get("num_epochs", 10)
        max_steps = self.config.get("max_steps", None)

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            self.train_epoch()

            if max_steps and self.global_step >= max_steps:
                print(f"Reached max_steps ({max_steps}), stopping training")
                break

    def train_epoch(self):
        """Single training epoch."""
        self.unet.train()
        self.vae.eval()  # VAE always in eval (frozen)

        # Progress bar (only on rank 0)
        if self._is_main_process():
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {self.epoch}",
                total=len(self.train_dataloader)
            )
        else:
            pbar = self.train_dataloader

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(pbar):
            loss = self.training_step(batch)

            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            if self._is_main_process():
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
                    "step": self.global_step
                })

            # Logging
            if self.global_step % self.config.get("log_interval", 10) == 0:
                self._log_metrics({
                    "train/loss": loss.item(),
                    "train/lr": self.lr_scheduler.get_last_lr()[0],
                    "train/epoch": self.epoch,
                    "train/step": self.global_step
                })

            # Checkpointing
            if self.global_step % self.config.get("save_interval", 1000) == 0:
                self.save_checkpoint()

            # Evaluation
            if self.global_step % self.config.get("eval_interval", 2500) == 0:
                self.evaluate()

            # Check max steps
            max_steps = self.config.get("max_steps")
            if max_steps and self.global_step >= max_steps:
                break

        # Log epoch metrics
        avg_loss = epoch_loss / max(num_batches, 1)
        if self._is_main_process():
            print(f"Epoch {self.epoch} - Avg Loss: {avg_loss:.4f}")

    def training_step(self, batch) -> torch.Tensor:
        """
        Single training step.

        Args:
            batch: Batch dict with "latents" and "mask"

        Returns:
            Loss tensor
        """
        # Move batch to device
        latents = batch["latents"].to(self.device)
        mask = batch["mask"].to(self.device)

        batch_size = latents.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long
        )

        # Add noise to latents (forward diffusion)
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Predict noise with UNet
        model_output = self.unet(
            noisy_latents,
            timesteps,
            # Add text conditioning if available
            encoder_hidden_states=batch.get("text_embeddings")
        ).sample

        # Compute loss
        loss = DiffusionLoss.simple_mse_loss(model_output, noise, mask)

        # Gradient accumulation
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)
        loss = loss / grad_accum_steps

        # Backward pass
        loss.backward()

        self.global_step += 1

        # Optimizer step (only after gradient accumulation)
        if self.global_step % grad_accum_steps == 0:
            # Gradient clipping
            max_grad_norm = self.config.get("max_grad_norm")
            grad_norm = None
            if max_grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.unet.parameters(),
                    max_grad_norm
                )
                # Log gradient norm
                if self.global_step % self.config.get("log_interval", 10) == 0:
                    if self._is_main_process():
                        grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                        self._log_metrics({
                            "train/grad_norm": grad_norm_value,
                            "train/step": self.global_step
                        })

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return loss * grad_accum_steps  # Return unscaled loss for logging

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluation with validation set.
        """
        if self.val_dataloader is None:
            return

        self.unet.eval()

        val_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            latents = batch["latents"].to(self.device)
            mask = batch["mask"].to(self.device)
            batch_size = latents.shape[0]

            # Sample timesteps
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps,
                (batch_size,), device=self.device
            )

            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            # Predict
            model_output = self.unet(noisy_latents, timesteps).sample

            # Compute loss
            loss = DiffusionLoss.simple_mse_loss(model_output, noise, mask)
            val_loss += loss.item()
            num_batches += 1

        avg_val_loss = val_loss / max(num_batches, 1)

        # Log validation metrics
        metrics = {
            "val/loss": avg_val_loss,
            "val/step": self.global_step
        }

        # Generate sample videos if configured
        video_log_interval = self.config.get("video_log_interval")
        if video_log_interval and self.global_step % video_log_interval == 0:
            if self._is_main_process() and self.logger:
                print("Generating sample videos...")
                videos = self._generate_sample_videos(num_videos=4, seed=42)
                if videos is not None and hasattr(self.logger, 'log_videos'):
                    self.logger.log_videos(
                        videos,
                        name="val/generated_videos",
                        fps=self.config.get("fps", 8),
                        step=self.global_step,
                        max_videos=4
                    )
                    print(f"Logged {len(videos)} sample videos")

        self._log_metrics(metrics)

        if self._is_main_process():
            print(f"Validation Loss: {avg_val_loss:.4f}")

        self.unet.train()

    def _generate_sample_videos(self, num_videos: int = 4, seed: int = 42) -> Optional[torch.Tensor]:
        """
        Generate sample videos for visual evaluation.

        Args:
            num_videos: Number of videos to generate
            seed: Random seed for reproducibility

        Returns:
            Generated video tensor [B, T, C, H, W] or None if generation fails
        """
        try:
            from ..diffusion.sampling import VideoSampler
            from ..diffusion.scheduler import SchedulerFactory

            # Create inference scheduler
            inference_scheduler = SchedulerFactory.create_inference_scheduler(
                num_inference_steps=self.config.get("num_inference_steps", 50)
            )

            # Create sampler
            sampler = VideoSampler(
                unet=self.unet,
                vae=self.vae,
                scheduler=inference_scheduler,
                device=self.device
            )

            # Generate videos
            videos = sampler.sample(
                batch_size=num_videos,
                num_frames=self.config.get("max_frames", 64),
                height=self.config.get("resolution", [256, 456])[0],
                width=self.config.get("resolution", [256, 456])[1],
                guidance_scale=self.config.get("guidance_scale", 7.5),
                num_inference_steps=self.config.get("num_inference_steps", 50),
                seed=seed,
                return_latents=False
            )

            return videos

        except Exception as e:
            print(f"Warning: Failed to generate sample videos: {e}")
            return None

    def save_checkpoint(self):
        """Save training checkpoint."""
        if not self._is_main_process():
            return

        checkpoint_dir = Path(self.config["output_dir"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get FSDP state dict (only on rank 0)
        if self.is_distributed:
            unet_state_dict = FSDPConfig.get_fsdp_state_dict(self.unet, full_state=True)
        else:
            unet_state_dict = self.unet.state_dict()

        # Save checkpoint
        checkpoint = {
            "unet": unet_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config,
        }

        checkpoint_path = checkpoint_dir / f"checkpoint-{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)

        print(f"Saved checkpoint to {checkpoint_path}")

        # Keep only last N checkpoints
        self._cleanup_checkpoints(checkpoint_dir)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint and resume training state.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load UNet state dict
        if self.is_distributed:
            # For FSDP, use load_state_dict directly
            self.unet.load_state_dict(checkpoint["unet"])
        else:
            self.unet.load_state_dict(checkpoint["unet"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load scheduler state
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # Restore training state
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]

        print(f"Resumed from step {self.global_step}, epoch {self.epoch}")

        return checkpoint

    def _cleanup_checkpoints(self, checkpoint_dir: Path, keep_last: int = 5):
        """Remove old checkpoints, keep only last N."""
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*.pt"))
        if len(checkpoints) > keep_last:
            for ckpt in checkpoints[:-keep_last]:
                ckpt.unlink()
                print(f"Removed old checkpoint: {ckpt}")

    def _log_metrics(self, metrics: dict):
        """Log metrics to logger."""
        if self.logger and self._is_main_process():
            self.logger.log(metrics, step=self.global_step)

    def _is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0
