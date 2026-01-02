import torch
from typing import Optional, Union
from diffusers import DDIMScheduler


class VideoSampler:
    """
    Generate video samples from diffusion model.

    Supports:
    - Classifier-free guidance
    - Text conditioning
    - Custom noise initialization
    """

    def __init__(
        self,
        unet,
        vae,
        scheduler: DDIMScheduler,
        device: str = "cuda"
    ):
        """
        Args:
            unet: Trained diffusion UNet
            vae: VAE for decoding latents to pixels
            scheduler: DDIM scheduler for inference
            device: Device for inference
        """
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.device = device

        # Move models to device
        self.unet.to(device)
        self.vae.to(device)

        # Set to eval mode
        self.unet.eval()
        self.vae.eval()

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        num_frames: int = 64,
        height: int = 256,
        width: int = 456,
        text_embeddings: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        return_latents: bool = False
    ) -> torch.Tensor:
        """
        Generate video samples.

        Args:
            batch_size: Number of videos to generate
            num_frames: Number of frames per video
            height: Video height in pixels
            width: Video width in pixels
            text_embeddings: Optional text conditioning [B, seq_len, dim]
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            return_latents: If True, return latents instead of decoded video

        Returns:
            Generated video tensor:
            - If return_latents=False: [B, T, C, H, W] in pixel space [0, 1]
            - If return_latents=True: [B, T_latent, C_latent, H_latent, W_latent]
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Calculate latent dimensions (8x8x8 compression for Cosmos)
        latent_t = num_frames // 8
        latent_h = height // 8
        latent_w = width // 8

        # Get latent channels from VAE config
        try:
            latent_c = self.vae.config.latent_channels
        except:
            latent_c = 4  # Default for most VAEs

        # Initialize random noise
        latents = torch.randn(
            batch_size, latent_t, latent_c, latent_h, latent_w,
            device=self.device,
            dtype=self.unet.dtype
        )

        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Prepare text embeddings for classifier-free guidance
        if text_embeddings is not None and guidance_scale > 1.0:
            # Duplicate for unconditional + conditional
            # Unconditional: zeros or special <unk> token
            uncond_embeddings = torch.zeros_like(text_embeddings)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Denoising loop
        for t in self.scheduler.timesteps:
            # Expand latents for CFG
            if text_embeddings is not None and guidance_scale > 1.0:
                latent_input = torch.cat([latents] * 2)
            else:
                latent_input = latents

            # Predict noise
            noise_pred = self.unet(
                latent_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample

            # Classifier-free guidance
            if text_embeddings is not None and guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Scheduler step
            latents = self.scheduler.step(
                noise_pred, t, latents
            ).prev_sample

        # Return latents if requested
        if return_latents:
            return latents

        # Decode latents to pixels
        videos = self.vae.decode(latents).sample

        # Denormalize from [-1, 1] to [0, 1]
        videos = (videos + 1.0) / 2.0
        videos = torch.clamp(videos, 0.0, 1.0)

        return videos

    @torch.no_grad()
    def sample_with_prompts(
        self,
        prompts: list,
        text_encoder,
        tokenizer,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate videos from text prompts.

        Args:
            prompts: List of text prompts
            text_encoder: Text encoder model
            tokenizer: Text tokenizer
            **kwargs: Additional arguments for sample()

        Returns:
            Generated videos [B, T, C, H, W]
        """
        # Tokenize prompts
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Encode text
        text_embeddings = text_encoder(
            text_inputs.input_ids.to(self.device)
        )[0]

        # Generate videos
        return self.sample(
            batch_size=len(prompts),
            text_embeddings=text_embeddings,
            **kwargs
        )
