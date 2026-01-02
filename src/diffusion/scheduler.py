from diffusers import DDPMScheduler, DDIMScheduler


class SchedulerFactory:
    """
    Factory for creating noise schedulers.

    Design decision:
    - Training: DDPM (standard, stable, Markovian)
    - Inference: DDIM (faster, deterministic, non-Markovian)
    """

    @staticmethod
    def create_train_scheduler(
        num_train_timesteps: int = 1000,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon"
    ) -> DDPMScheduler:
        """
        Create DDPM scheduler for training.

        Args:
            num_train_timesteps: Number of diffusion steps (1000 is standard)
            beta_schedule: Noise schedule type
                - "scaled_linear": Best for latent diffusion
                - "linear": Best for pixel-space diffusion
            prediction_type: What the model predicts
                - "epsilon": Predict noise (standard)
                - "v_prediction": Predict velocity (alternative)

        Returns:
            DDPM scheduler
        """
        return DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=False,  # No clamping in latent space
            variance_type="fixed_small",  # Standard for latent models
        )

    @staticmethod
    def create_inference_scheduler(
        num_inference_steps: int = 50,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon"
    ) -> DDIMScheduler:
        """
        Create DDIM scheduler for fast inference.

        DDIM allows much fewer steps (20-50) compared to DDPM (1000)
        while maintaining quality.

        Args:
            num_inference_steps: Number of denoising steps (50 is good quality/speed)
            beta_schedule: Should match training scheduler
            prediction_type: Should match training scheduler

        Returns:
            DDIM scheduler
        """
        return DDIMScheduler(
            num_train_timesteps=1000,  # Must match training
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=False,
            set_alpha_to_one=False,  # Better for video
        )
