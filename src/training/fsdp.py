import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from functools import partial
from typing import Optional


class FSDPConfig:
    """
    FSDP2 configuration for Cosmos-2B training.

    Key strategy:
    1. VAE: NO_SHARD (frozen, shared across ranks)
    2. UNet: FULL_SHARD (trainable, large model)
    3. Mixed precision: BF16 for compute, FP32 for parameters
    """

    @staticmethod
    def get_mixed_precision_policy(dtype: str = "bf16") -> MixedPrecision:
        """
        Create mixed precision policy.

        BF16 is preferred on H100 GPUs for better numerical stability
        and performance compared to FP16.

        Args:
            dtype: "bf16", "fp16", or "fp32"

        Returns:
            MixedPrecision policy
        """
        if dtype == "bf16":
            return MixedPrecision(
                param_dtype=torch.float32,  # Master weights in FP32
                reduce_dtype=torch.bfloat16,  # Gradient reduction in BF16
                buffer_dtype=torch.bfloat16,  # Buffers in BF16
            )
        elif dtype == "fp16":
            return MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:  # fp32
            return None

    @staticmethod
    def get_wrap_policy(model_type: str = "unet"):
        """
        Module-level wrapping policy.

        For UNet: Wrap at transformer block level (not individual layers)
        This balances communication overhead with memory efficiency.

        Args:
            model_type: "unet" or "vae"

        Returns:
            Wrap policy or None
        """
        if model_type == "unet":
            from diffusers.models.attention import BasicTransformerBlock
            return ModuleWrapPolicy({BasicTransformerBlock})
        else:
            # No wrapping for VAE (will use NO_SHARD)
            return None

    @staticmethod
    def wrap_vae(
        vae: nn.Module,
        device_id: Optional[int] = None
    ) -> FSDP:
        """
        Wrap VAE with NO_SHARD strategy.

        VAE is frozen, so we don't shard it. This reduces
        communication overhead since all ranks need the full model.

        Args:
            vae: VAE model
            device_id: CUDA device ID

        Returns:
            FSDP-wrapped VAE
        """
        return FSDP(
            vae,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            device_id=device_id or torch.cuda.current_device(),
        )

    @staticmethod
    def wrap_unet(
        unet: nn.Module,
        mixed_precision: str = "bf16",
        cpu_offload: bool = False,
        device_id: Optional[int] = None,
        use_gradient_checkpointing: bool = True
    ) -> FSDP:
        """
        Wrap UNet with FULL_SHARD strategy.

        Args:
            unet: UNet model
            mixed_precision: "bf16", "fp16", or "fp32"
            cpu_offload: Whether to offload parameters to CPU
            device_id: CUDA device ID
            use_gradient_checkpointing: Enable gradient checkpointing

        Returns:
            FSDP-wrapped UNet
        """
        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()

        # Create FSDP config
        fsdp_config = {
            "sharding_strategy": ShardingStrategy.FULL_SHARD,
            "mixed_precision": FSDPConfig.get_mixed_precision_policy(mixed_precision),
            "auto_wrap_policy": FSDPConfig.get_wrap_policy("unet"),
            "device_id": device_id or torch.cuda.current_device(),
            "use_orig_params": True,  # Better for optimizer state
        }

        # Add CPU offload if requested
        if cpu_offload:
            fsdp_config["cpu_offload"] = CPUOffload(offload_params=True)

        return FSDP(unet, **fsdp_config)

    @staticmethod
    def get_fsdp_state_dict(model: FSDP, full_state: bool = True):
        """
        Extract state dict from FSDP model.

        Args:
            model: FSDP-wrapped model
            full_state: If True, gather full state dict (rank 0 only)
                       If False, get local shard

        Returns:
            State dict
        """
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig

        if full_state:
            # Gather full state dict on rank 0
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                return model.state_dict()
        else:
            # Get local shard
            return model.state_dict()
