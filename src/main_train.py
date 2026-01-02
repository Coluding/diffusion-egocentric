#!/usr/bin/env python3
"""
Main training script for Cosmos-2B finetuning.

Launch with torchrun:
    torchrun --nproc_per_node=3 src/main_train.py [hydra overrides]
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from pathlib import Path
import os

from data.hf_egocentric import EgocentricDataset
from data.dataset import ComposedVideoDataset
from models.cosmos_loader import CosmosModelLoader
from models.freeze_utils import ModelFreezer
from models.lora import LoRAInjector
from models.temporal import TemporalAdapterInjector
from diffusion.scheduler import SchedulerFactory
from training.fsdp import FSDPConfig
from training.trainer import CosmosTrainer
from logging.wandb import WandBLogger
from logging.tensorboard import TensorBoardLogger


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()

        torch.cuda.set_device(local_rank)

        print(f"Initialized distributed: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = f"cuda:{local_rank}"

    # Set random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Load models
    print("\nLoading models...")
    vae, unet = CosmosModelLoader.load_components_separately(
        model_id=cfg.model.name,
        vae_id=cfg.model.vae_name,
        dtype=torch.float16 if cfg.model.dtype == "fp16" else torch.bfloat16
    )

    # Apply model modifications
    print("\nApplying model modifications...")

    # 1. Freeze VAE
    if cfg.model.freeze_vae:
        ModelFreezer.freeze_vae(vae)

    # 2. Freeze spatial blocks
    if cfg.model.freeze_spatial_blocks:
        ModelFreezer.freeze_spatial_blocks(
            unet,
            keep_output_proj_trainable=cfg.model.keep_output_proj_trainable
        )

    # 3. Inject LoRA
    if cfg.model.use_lora:
        unet = LoRAInjector.inject_lora(
            unet,
            rank=cfg.model.lora_rank,
            alpha=cfg.model.lora_alpha,
            dropout=cfg.model.lora_dropout,
            target_modules=cfg.model.lora_target_modules
        )

    # 4. Inject temporal adapters
    if cfg.model.use_temporal_adapters:
        unet = TemporalAdapterInjector.inject_adapters(
            unet,
            reduction_factor=cfg.model.adapter_reduction_factor,
            dropout=cfg.model.adapter_dropout
        )

    # Print parameter summary
    ModelFreezer.print_trainable_summary(unet)
    ModelFreezer.print_trainable_modules(unet, max_lines=30)

    # Wrap with FSDP
    if cfg.model.use_fsdp and world_size > 1:
        print("\nWrapping models with FSDP...")
        vae = FSDPConfig.wrap_vae(vae, device_id=local_rank)
        unet = FSDPConfig.wrap_unet(
            unet,
            mixed_precision=cfg.mixed_precision,
            cpu_offload=cfg.model.cpu_offload,
            device_id=local_rank,
            use_gradient_checkpointing=cfg.model.use_gradient_checkpointing
        )
    else:
        vae.to(device)
        unet.to(device)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = EgocentricDataset(
        split=cfg.data.split,
        cache_dir=Path(cfg.data.cache_dir),
        max_frames=cfg.data.max_frames,
        fps=cfg.data.fps,
        resolution=tuple(cfg.data.resolution),
        streaming=cfg.data.streaming
    )

    val_dataset = None
    if cfg.data.val_split:
        val_dataset = EgocentricDataset(
            split=cfg.data.val_split,
            cache_dir=Path(cfg.data.cache_dir),
            max_frames=cfg.data.max_frames,
            fps=cfg.data.fps,
            resolution=tuple(cfg.data.resolution),
            streaming=cfg.data.get("val_streaming", False)
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
        collate_fn=train_dataset.get_collate_fn()
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.val_batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            collate_fn=val_dataset.get_collate_fn()
        )

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")

    # Create optimizer
    print("\nCreating optimizer...")
    trainable_params = ModelFreezer.get_trainable_params(unet)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
        eps=cfg.optim.eps
    )

    # Create LR scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=cfg.optim.warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_steps - cfg.optim.warmup_steps,
        eta_min=cfg.optim.min_lr
    )

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.optim.warmup_steps]
    )

    # Create noise scheduler
    scheduler = SchedulerFactory.create_train_scheduler()

    # Create loggers
    logger = None
    if rank == 0:
        if cfg.logging.use_wandb:
            logger = WandBLogger(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                run_name=cfg.logging.wandb_run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                enabled=True
            )
        elif cfg.logging.use_tensorboard:
            logger = TensorBoardLogger(
                log_dir=cfg.logging.tensorboard_dir,
                enabled=True
            )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = CosmosTrainer(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=OmegaConf.to_container(cfg, resolve=True),
        logger=logger,
        device=device
    )

    # Resume from checkpoint if specified
    if cfg.get("resume_from_checkpoint"):
        checkpoint_path = Path(cfg.resume_from_checkpoint)
        if checkpoint_path.exists():
            trainer.load_checkpoint(str(checkpoint_path))
        else:
            print(f"WARNING: Checkpoint not found at {checkpoint_path}, starting from scratch")

    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()

    print("\nTraining complete!")

    # Cleanup
    if logger and rank == 0:
        if hasattr(logger, 'finish'):
            logger.finish()
        elif hasattr(logger, 'close'):
            logger.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
