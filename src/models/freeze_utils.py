import torch.nn as nn
from typing import List


class ModelFreezer:
    """
    Utilities for freezing model components.

    Implements parameter-efficient finetuning by freezing spatial components
    while keeping temporal components trainable.
    """

    @staticmethod
    def freeze_vae(vae: nn.Module):
        """
        Freeze entire VAE (100% frozen).

        VAE is only used for encoding/decoding and should never be trained.
        """
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()

        frozen_params = sum(p.numel() for p in vae.parameters())
        print(f"Froze VAE: {frozen_params:,} parameters")

    @staticmethod
    def freeze_spatial_blocks(unet: nn.Module, keep_output_proj_trainable: bool = True):
        """
        Freeze spatial self-attention and conv blocks in UNet.
        Keep temporal attention and adapters trainable.

        This function implements heuristics to identify spatial vs temporal components.
        Adjust module name patterns based on actual Cosmos architecture.

        Naming conventions (adjust after inspecting model):
        - Spatial: "attn1", "conv_in", "conv1", "conv2", spatial norms
        - Temporal: "attn_temporal", "temp_conv", "temporal"
        - Cross-attention: "attn2" (will have LoRA)
        - Output: "conv_out" (optionally trainable)

        Args:
            unet: UNet model
            keep_output_proj_trainable: Keep output projection trainable
        """
        frozen_count = 0
        trainable_count = 0

        # Keywords for spatial components (freeze these)
        spatial_keywords = [
            "attn1",  # Spatial self-attention
            "conv_in",  # Input convolution
            "conv1", "conv2",  # Spatial convolutions
            "norm1",  # Spatial norms
            "time_embed",  # Time embedding (shared across frames)
        ]

        # Keywords for temporal components (keep trainable)
        temporal_keywords = [
            "attn_temporal",
            "temp_conv",
            "temporal",
            "attn2",  # Cross-attention (for LoRA)
        ]

        # Keywords for output (conditionally trainable)
        output_keywords = ["conv_out", "out_layers"]

        for name, param in unet.named_parameters():
            should_freeze = False

            # Check if this is an output projection
            if any(kw in name for kw in output_keywords):
                if keep_output_proj_trainable:
                    param.requires_grad = True
                    trainable_count += param.numel()
                else:
                    param.requires_grad = False
                    frozen_count += param.numel()
                continue

            # Check if temporal (keep trainable)
            if any(kw in name for kw in temporal_keywords):
                param.requires_grad = True
                trainable_count += param.numel()
                continue

            # Check if spatial (freeze)
            if any(kw in name for kw in spatial_keywords):
                param.requires_grad = False
                frozen_count += param.numel()
                continue

            # Default: freeze unknown components for safety
            # (adjust after inspecting architecture)
            param.requires_grad = False
            frozen_count += param.numel()

        print(f"\nParameter Freezing Summary:")
        print(f"  Frozen spatial blocks: {frozen_count:,} parameters")
        print(f"  Trainable temporal blocks: {trainable_count:,} parameters")
        print(f"  Trainable ratio: {100 * trainable_count / (frozen_count + trainable_count):.2f}%")

    @staticmethod
    def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
        """
        Get only trainable parameters for optimizer.

        Args:
            model: Model to extract trainable params from

        Returns:
            List of trainable parameters
        """
        return [p for p in model.parameters() if p.requires_grad]

    @staticmethod
    def print_trainable_summary(model: nn.Module):
        """
        Print detailed summary of trainable vs frozen parameters.

        Args:
            model: Model to summarize
        """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        print(f"\n{'=' * 60}")
        print(f"Model Parameter Summary:")
        print(f"  Total params: {total:,}")
        print(f"  Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
        print(f"  Frozen params: {total - trainable:,} ({100 * (1 - trainable / total):.2f}%)")
        print(f"{'=' * 60}\n")

    @staticmethod
    def print_trainable_modules(model: nn.Module, max_lines: int = 50):
        """
        Print which modules are trainable.

        Args:
            model: Model to inspect
            max_lines: Maximum lines to print
        """
        print("\nTrainable modules:")
        print("-" * 60)

        trainable_modules = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_modules.append((name, param.numel()))

        for i, (name, num_params) in enumerate(trainable_modules[:max_lines]):
            print(f"  {name:50s} {num_params:>10,}")

        if len(trainable_modules) > max_lines:
            print(f"  ... and {len(trainable_modules) - max_lines} more modules")

        print("-" * 60 + "\n")
