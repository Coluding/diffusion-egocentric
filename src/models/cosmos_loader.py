import torch
import os
from diffusers import DiffusionPipeline, AutoencoderKL, UNet3DConditionModel
from typing import Tuple, Optional


class CosmosModelLoader:
    """
    Load and prepare Cosmos-2B video diffusion model components.

    NVIDIA Cosmos models use:
    - VAE: Cosmos-1.0-Tokenizer-CV8x8x8 (8x8x8 spatiotemporal compression)
    - UNet: 3D diffusion transformer with spatial + temporal blocks
    """

    @staticmethod
    def load_full_pipeline(
        model_id: str = "nvidia/Cosmos-Predict2.5-2B",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ) -> DiffusionPipeline:
        """
        Load complete diffusion pipeline.

        Args:
            model_id: HuggingFace model identifier
            device: Target device
            dtype: Model dtype

        Returns:
            Complete pipeline with VAE, UNet, scheduler
        """
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None
        )
        pipeline.to(device)
        return pipeline

    @staticmethod
    def load_components_separately(
        model_id: str = "nvidia/Cosmos-Predict2.5-2B",
        vae_id: str = "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8",
        dtype: torch.dtype = torch.float16
    ) -> Tuple[AutoencoderKL, UNet3DConditionModel]:
        """
        Load VAE and UNet separately for fine-grained control.

        This is preferred for finetuning as it allows:
        - Separate device placement
        - Independent freezing
        - Custom wrapping (e.g., FSDP)

        Args:
            model_id: HuggingFace model identifier for full model
            vae_id: VAE model identifier
            dtype: Model dtype

        Returns:
            Tuple of (vae, unet)
        """
        # Get HuggingFace token from environment
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print("Using HF_TOKEN for authentication")

        # Load VAE (will be frozen during training)
        print(f"Loading VAE from {vae_id}...")
        vae = AutoencoderKL.from_pretrained(
            vae_id,
            torch_dtype=dtype,
            token=hf_token
        )
        print(f"VAE loaded: {sum(p.numel() for p in vae.parameters()):,} parameters")

        # Load UNet
        print(f"Loading UNet from {model_id}...")
        try:
            unet = UNet3DConditionModel.from_pretrained(
                model_id,
                subfolder="unet",
                torch_dtype=dtype,
                token=hf_token
            )
        except Exception as e:
            print(f"Failed to load from subfolder, trying direct load: {e}")
            # Fallback: load full pipeline and extract UNet
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                token=hf_token
            )
            unet = pipeline.unet

        print(f"UNet loaded: {sum(p.numel() for p in unet.parameters()):,} parameters")

        return vae, unet

    @staticmethod
    def get_model_config(model_id: str) -> dict:
        """
        Retrieve model configuration for architecture inspection.

        Args:
            model_id: HuggingFace model identifier

        Returns:
            Model configuration dict
        """
        try:
            config = UNet3DConditionModel.load_config(
                model_id,
                subfolder="unet"
            )
        except:
            # Fallback: load from main config
            from diffusers import DiffusionPipeline
            pipeline = DiffusionPipeline.from_pretrained(model_id)
            config = pipeline.unet.config

        return config

    @staticmethod
    def inspect_architecture(model: torch.nn.Module):
        """
        Print model architecture for debugging freezing/LoRA strategies.

        Args:
            model: Model to inspect
        """
        print("\n" + "=" * 60)
        print("Model Architecture")
        print("=" * 60)

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    print(f"{name:60s} {type(module).__name__:30s} {num_params:>12,}")

        print("=" * 60)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print("=" * 60 + "\n")
