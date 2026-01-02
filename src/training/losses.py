import torch
import torch.nn.functional as F
from typing import Optional


class DiffusionLoss:
    """
    Loss functions for diffusion model training.
    """

    @staticmethod
    def simple_mse_loss(
        model_output: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard MSE loss for diffusion training.

        Args:
            model_output: Predicted noise [B, T, C, H, W]
            target: Ground truth noise [B, T, C, H, W]
            mask: Frame validity mask [B, T]

        Returns:
            Scalar loss
        """
        # Compute MSE
        loss = F.mse_loss(model_output, target, reduction='none')

        # Apply temporal mask if provided
        if mask is not None:
            # Expand mask: [B, T] -> [B, T, 1, 1, 1]
            mask_expanded = mask.view(mask.shape[0], mask.shape[1], 1, 1, 1).float()
            loss = loss * mask_expanded

            # Normalize by valid frames
            return loss.sum() / mask_expanded.sum().clamp(min=1.0)
        else:
            return loss.mean()

    @staticmethod
    def snr_weighted_loss(
        model_output: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler,
        mask: Optional[torch.Tensor] = None,
        min_snr_gamma: float = 5.0
    ) -> torch.Tensor:
        """
        SNR-weighted loss using Min-SNR strategy.

        Helps balance loss across different timesteps by weighting
        based on signal-to-noise ratio.

        Reference: "Efficient Diffusion Training via Min-SNR Weighting Strategy"
        https://arxiv.org/abs/2303.09556

        Args:
            model_output: Predicted noise [B, T, C, H, W]
            target: Ground truth noise [B, T, C, H, W]
            timesteps: Timestep indices [B]
            scheduler: Noise scheduler (for extracting SNR)
            mask: Frame validity mask [B, T]
            min_snr_gamma: Maximum SNR weight (clamping)

        Returns:
            Scalar loss
        """
        # Get SNR from scheduler
        alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        # SNR = (alpha_t / (1 - alpha_t))
        snr = (sqrt_alpha_prod / sqrt_one_minus_alpha_prod) ** 2

        # Min-SNR clamping
        snr_weight = torch.clamp(snr, max=min_snr_gamma)

        # Compute base MSE loss
        base_loss = F.mse_loss(model_output, target, reduction='none')

        # Apply SNR weighting: [B] -> [B, 1, 1, 1, 1]
        snr_weight = snr_weight.view(-1, 1, 1, 1, 1)
        weighted_loss = base_loss * snr_weight

        # Apply mask
        if mask is not None:
            mask_expanded = mask.view(mask.shape[0], mask.shape[1], 1, 1, 1).float()
            weighted_loss = weighted_loss * mask_expanded
            return weighted_loss.sum() / mask_expanded.sum().clamp(min=1.0)
        else:
            return weighted_loss.mean()

    @staticmethod
    def huber_loss(
        model_output: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        delta: float = 1.0
    ) -> torch.Tensor:
        """
        Huber loss for robust training.

        Less sensitive to outliers than MSE.

        Args:
            model_output: Predicted noise [B, T, C, H, W]
            target: Ground truth noise [B, T, C, H, W]
            mask: Frame validity mask [B, T]
            delta: Huber delta parameter

        Returns:
            Scalar loss
        """
        loss = F.huber_loss(model_output, target, reduction='none', delta=delta)

        if mask is not None:
            mask_expanded = mask.view(mask.shape[0], mask.shape[1], 1, 1, 1).float()
            loss = loss * mask_expanded
            return loss.sum() / mask_expanded.sum().clamp(min=1.0)
        else:
            return loss.mean()
