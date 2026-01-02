#!/usr/bin/env python3
"""
Main evaluation script for Cosmos-2B.

Usage:
    python src/main_eval.py \\
        checkpoint_path=/path/to/checkpoint.pt \\
        output_dir=/path/to/output
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path

from models.cosmos_loader import CosmosModelLoader
from models.lora import LoRAInjector
from models.temporal import TemporalAdapterInjector
from eval.evaluator import CosmosEvaluator


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    """
    Main evaluation function.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 60)
    print("Evaluation Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Check checkpoint path
    if not cfg.get("checkpoint_path"):
        raise ValueError("checkpoint_path must be provided")

    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load checkpoint first to get model config
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint or use default
    model_cfg = checkpoint.get("config", {}).get("model", {})
    model_id = model_cfg.get("name", "nvidia/Cosmos-Predict2.5-2B")
    vae_id = model_cfg.get("vae_name", "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8")

    # Load models
    print("\nLoading models...")
    vae, unet = CosmosModelLoader.load_components_separately(
        model_id=model_id,
        vae_id=vae_id,
        dtype=torch.float16
    )

    # Apply same modifications as training
    print("\nApplying model modifications...")

    # Inject LoRA if used
    if model_cfg.get("use_lora", True):
        unet = LoRAInjector.inject_lora(
            unet,
            rank=model_cfg.get("lora_rank", 16),
            alpha=model_cfg.get("lora_alpha", 32),
            target_modules=model_cfg.get("lora_target_modules", [
                "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out"
            ])
        )

    # Inject temporal adapters if used
    if model_cfg.get("use_temporal_adapters", True):
        unet = TemporalAdapterInjector.inject_adapters(
            unet,
            reduction_factor=model_cfg.get("adapter_reduction_factor", 8)
        )

    # Load checkpoint weights
    print("\nLoading checkpoint weights...")
    try:
        unet.load_state_dict(checkpoint["unet"])
        print(f"Loaded UNet from step {checkpoint.get('global_step', 'unknown')}")
    except Exception as e:
        print(f"Warning: Could not load full state dict: {e}")
        print("Attempting to load with strict=False...")
        unet.load_state_dict(checkpoint["unet"], strict=False)

    # Move to device
    unet.to(device)
    vae.to(device)

    # Set to eval mode
    unet.eval()
    vae.eval()

    # Create evaluator
    print("\nCreating evaluator...")
    evaluator = CosmosEvaluator(
        unet=unet,
        vae=vae,
        config=OmegaConf.to_container(cfg, resolve=True),
        device=device
    )

    # Run evaluation
    output_dir = Path(cfg.get("output_dir", "./eval_output"))
    print(f"\nRunning evaluation, saving to {output_dir}...")
    print("=" * 60)

    results = evaluator.evaluate_checkpoint(
        output_dir=str(output_dir),
        checkpoint_step=checkpoint.get("global_step")
    )

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
