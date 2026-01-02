import torch
import torch.nn as nn
from typing import List, Optional
import math


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer.

    Replaces: y = W @ x
    With: y = (W + α/r * B @ A) @ x

    Where:
    - W: Original frozen weight
    - A: Down-projection [in_features, rank]
    - B: Up-projection [rank, out_features]
    - α: Scaling factor (alpha)
    - r: Rank

    LoRA reduces trainable parameters from in_features * out_features
    to rank * (in_features + out_features).
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        """
        Args:
            original_layer: Original linear layer to adapt
            rank: LoRA rank (lower = fewer params, typical: 4-32)
            alpha: LoRA alpha (scaling factor, typically 2 * rank)
            dropout: Dropout rate for LoRA path
        """
        super().__init__()

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze original layer
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Scaling: α / r
        self.scaling = alpha / rank

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize LoRA matrices
        # A: Kaiming uniform (similar to original weight init)
        # B: Zero (so initial LoRA contribution is zero)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining original and LoRA paths.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Original frozen path
        original_out = self.original_layer(x)

        # LoRA trainable path
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))

        return original_out + lora_out * self.scaling

    def extra_repr(self) -> str:
        return f"rank={self.lora_A.out_features}, alpha={self.scaling * self.lora_A.out_features}"


class LoRAInjector:
    """
    Inject LoRA into specified modules of a model.

    Typically used for cross-attention layers in diffusion models.
    """

    @staticmethod
    def inject_lora(
        unet: nn.Module,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        target_modules: List[str] = None
    ) -> nn.Module:
        """
        Replace target linear layers with LoRA versions.

        Args:
            unet: UNet model
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: LoRA dropout rate
            target_modules: List of module name patterns to target
                Default: cross-attention Q, K, V, output projections

        Returns:
            Modified unet with LoRA injected
        """
        if target_modules is None:
            target_modules = [
                "attn2.to_q",
                "attn2.to_k",
                "attn2.to_v",
                "attn2.to_out"
            ]

        lora_count = 0
        replaced_params = 0
        lora_params = 0

        # Iterate through all modules
        for name, module in list(unet.named_modules()):
            # Check if this module matches any target pattern
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Get parent module and attribute name
                    *parent_path, attr = name.split('.')
                    parent = unet
                    for p in parent_path:
                        parent = getattr(parent, p)

                    # Create LoRA layer
                    lora_layer = LoRALayer(
                        module,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout
                    )

                    # Replace module
                    setattr(parent, attr, lora_layer)

                    # Statistics
                    lora_count += 1
                    replaced_params += module.weight.numel()
                    lora_params += lora_layer.lora_A.weight.numel() + lora_layer.lora_B.weight.numel()

        print(f"\nLoRA Injection Summary:")
        print(f"  Injected LoRA into {lora_count} layers")
        print(f"  Original parameters: {replaced_params:,}")
        print(f"  LoRA parameters: {lora_params:,}")
        print(f"  Parameter reduction: {100 * (1 - lora_params / replaced_params):.2f}%")

        return unet

    @staticmethod
    def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
        """
        Extract only LoRA parameters from model.

        Useful for:
        - Creating optimizer that only trains LoRA
        - Saving only LoRA weights

        Args:
            model: Model with LoRA layers

        Returns:
            List of LoRA parameters
        """
        lora_params = []
        for module in model.modules():
            if isinstance(module, LoRALayer):
                lora_params.extend([
                    module.lora_A.weight,
                    module.lora_B.weight
                ])
        return lora_params

    @staticmethod
    def save_lora_weights(model: nn.Module, path: str):
        """
        Save only LoRA weights to file.

        This creates a small checkpoint containing only the trainable LoRA parameters,
        which can be loaded on top of the base model.

        Args:
            model: Model with LoRA layers
            path: Save path
        """
        lora_state_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRALayer):
                lora_state_dict[f"{name}.lora_A.weight"] = module.lora_A.weight
                lora_state_dict[f"{name}.lora_B.weight"] = module.lora_B.weight

        torch.save(lora_state_dict, path)
        print(f"Saved LoRA weights to {path}")
        print(f"  LoRA parameters: {sum(p.numel() for p in lora_state_dict.values()):,}")

    @staticmethod
    def load_lora_weights(model: nn.Module, path: str):
        """
        Load LoRA weights from file.

        Args:
            model: Model with LoRA layers (must have same architecture)
            path: Path to LoRA weights
        """
        lora_state_dict = torch.load(path)

        for name, module in model.named_modules():
            if isinstance(module, LoRALayer):
                a_key = f"{name}.lora_A.weight"
                b_key = f"{name}.lora_B.weight"

                if a_key in lora_state_dict:
                    module.lora_A.weight.data = lora_state_dict[a_key]
                if b_key in lora_state_dict:
                    module.lora_B.weight.data = lora_state_dict[b_key]

        print(f"Loaded LoRA weights from {path}")
