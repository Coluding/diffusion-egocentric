import torch
import torch.nn as nn
from typing import Optional


class TemporalAdapter(nn.Module):
    """
    Lightweight adapter for temporal attention blocks.

    Architecture: Input -> LayerNorm -> Down -> GELU -> Up -> Residual

    Adapters add minimal parameters while allowing effective adaptation
    of pre-trained models to new tasks.
    """

    def __init__(
        self,
        dim: int,
        reduction_factor: int = 8,
        dropout: float = 0.0
    ):
        """
        Args:
            dim: Hidden dimension of the attention block
            reduction_factor: Bottleneck reduction (dim / reduction_factor)
            dropout: Dropout rate
        """
        super().__init__()

        bottleneck_dim = max(1, dim // reduction_factor)

        self.adapter = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )

        # Initialize near-zero for stable training
        # This ensures adapters start with minimal effect
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with residual connection.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Output tensor [..., dim]
        """
        return x + self.adapter(x)


class TemporalAdapterInjector:
    """
    Inject temporal adapters into UNet temporal blocks.

    Strategy:
    1. Identify temporal attention modules
    2. Wrap their forward pass to include adapter
    3. Register adapter as submodule for gradient tracking
    """

    @staticmethod
    def inject_adapters(
        unet: nn.Module,
        reduction_factor: int = 8,
        dropout: float = 0.0,
        target_modules: Optional[list] = None
    ) -> nn.Module:
        """
        Insert adapters after temporal attention layers.

        Args:
            unet: UNet model
            reduction_factor: Adapter bottleneck reduction
            dropout: Adapter dropout rate
            target_modules: List of module name patterns to target
                Default: temporal attention modules

        Returns:
            Modified unet with adapters
        """
        if target_modules is None:
            target_modules = [
                "attn_temporal",
                "temp_attn",
                "temporal_transformer"
            ]

        adapter_count = 0
        total_adapter_params = 0

        for name, module in unet.named_modules():
            # Check if this is a temporal attention module
            if any(target in name for target in target_modules):
                # Try to infer hidden dimension
                hidden_dim = None

                # Check common attribute names for output projection
                for attr_name in ["out_proj", "to_out", "proj_out", "linear"]:
                    if hasattr(module, attr_name):
                        proj = getattr(module, attr_name)
                        if isinstance(proj, nn.Linear):
                            hidden_dim = proj.out_features
                            break

                if hidden_dim is None:
                    print(f"Warning: Could not infer hidden dim for {name}, skipping")
                    continue

                # Create adapter
                adapter = TemporalAdapter(
                    dim=hidden_dim,
                    reduction_factor=reduction_factor,
                    dropout=dropout
                )

                # Register as submodule
                module.register_module("temporal_adapter", adapter)

                # Wrap forward method
                original_forward = module.forward

                def create_wrapped_forward(orig_fn, adapter_module):
                    def wrapped_forward(*args, **kwargs):
                        out = orig_fn(*args, **kwargs)
                        # Handle both tensor and tuple returns
                        if isinstance(out, tuple):
                            out = (adapter_module(out[0]),) + out[1:]
                        else:
                            out = adapter_module(out)
                        return out
                    return wrapped_forward

                module.forward = create_wrapped_forward(original_forward, adapter)

                adapter_count += 1
                total_adapter_params += sum(p.numel() for p in adapter.parameters())

        print(f"\nTemporal Adapter Injection Summary:")
        print(f"  Injected {adapter_count} adapters")
        print(f"  Total adapter parameters: {total_adapter_params:,}")

        return unet

    @staticmethod
    def get_adapter_params(model: nn.Module) -> list:
        """
        Extract only adapter parameters from model.

        Args:
            model: Model with temporal adapters

        Returns:
            List of adapter parameters
        """
        adapter_params = []
        for module in model.modules():
            if isinstance(module, TemporalAdapter):
                adapter_params.extend(list(module.parameters()))
        return adapter_params
