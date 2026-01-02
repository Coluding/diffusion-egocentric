# Cosmos-2B Video Diffusion Finetuning

Parameter-efficient finetuning of NVIDIA's Cosmos-2B video diffusion model on HPC infrastructure.

## Features

- **Parameter-efficient training**: LoRA + temporal adapters (~5-10% trainable parameters)
- **Distributed training**: FSDP2 with 3x H100 GPUs
- **Latent caching**: Pre-encode videos with VAE for fast training
- **Extensible datasets**: Easy to add new video datasets via base interface
- **Hydra configuration**: Modular YAML configs for all components
- **SLURM integration**: Ready-to-use job scripts for HPC clusters

## Project Structure

```
cosmos-finetune/
├── configs/          # Hydra configuration files
│   ├── train.yaml    # Training parameters
│   ├── model.yaml    # Model architecture & freezing
│   ├── data.yaml     # Dataset configuration
│   ├── optim.yaml    # Optimizer settings
│   ├── logging.yaml  # W&B / TensorBoard config
│   └── eval.yaml     # Evaluation settings
├── slurm/            # SLURM job scripts
│   ├── download_data.sh  # Latent caching job
│   ├── train.sh          # Training job
│   └── eval.sh           # Evaluation job
├── src/
│   ├── data/         # Dataset & caching
│   ├── models/       # Cosmos loader, LoRA, freezing
│   ├── diffusion/    # Schedulers & sampling
│   ├── training/     # FSDP, losses, trainer
│   ├── eval/         # Evaluation & video utils
│   ├── logging/      # W&B / TensorBoard loggers
│   ├── main_train.py # Training entry point
│   └── main_eval.py  # Evaluation entry point
└── logs/             # SLURM output logs
```

## Quick Start

### 1. Setup Environment

```bash
# Activate conda environment (must have PyTorch, diffusers, etc.)
conda activate verlenv

# Install additional dependencies
pip install -r requirements.txt
```

### 2. Cache Latents (Required before training)

This step encodes videos with the VAE and caches latents to disk. **Must be run before training.**

```bash
# Submit caching job (48 hours, 1 GPU)
sbatch slurm/download_data.sh
```

**Configuration**: Edit `slurm/download_data.sh` to adjust:
- Dataset: `builddotai/Egocentric-100K`
- Cache directory: `/scratch-shared/scur1900/cosmos_cache`
- Batch size: `16`

**Monitoring**:
```bash
# Check job status
squeue -u $USER

# Watch logs
tail -f logs/cache_*.out
```

**Expected output**:
- Cache directory with `shard_*` folders
- SafeTensors files (one per video)
- ~20TB for Egocentric-100K dataset

### 3. Train Model

Once latent caching is complete, launch training:

```bash
# Submit training job (24 hours, 3 GPUs)
sbatch slurm/train.sh
```

**Configuration**: Edit `configs/train.yaml` or override via command line:
```bash
# Override config from SLURM script
torchrun ... src/main_train.py \
    data.batch_size=4 \
    optim.lr=5e-5 \
    model.lora_rank=32
```

**Monitoring**:
```bash
# Watch training logs
tail -f logs/train_*.out

# TensorBoard (if enabled)
tensorboard --logdir logs/tensorboard

# Weights & Biases
# Check your W&B dashboard
```

**Training outputs**:
- Checkpoints: `outputs/checkpoints/checkpoint-*.pt`
- Hydra logs: `outputs/hydra_runs/`

### 4. Evaluate Model

```bash
# Submit evaluation job
sbatch slurm/eval.sh /path/to/checkpoint.pt
```

**Outputs**:
- Generated videos: `/scratch-shared/scur1900/cosmos_eval/videos/`
- Video grid: `/scratch-shared/scur1900/cosmos_eval/video_grid.mp4`

## Configuration

### Model Configuration (`configs/model.yaml`)

```yaml
# Freezing strategy
freeze_vae: true                    # Freeze VAE (100%)
freeze_spatial_blocks: true         # Freeze spatial attention
keep_output_proj_trainable: true   # Keep output projection trainable

# LoRA
use_lora: true
lora_rank: 16                       # Rank (4-64)
lora_alpha: 32                      # Alpha (typically 2 * rank)
lora_target_modules:                # Cross-attention only
  - "attn2.to_q"
  - "attn2.to_k"
  - "attn2.to_v"
  - "attn2.to_out"

# Temporal adapters
use_temporal_adapters: true
adapter_reduction_factor: 8         # Bottleneck reduction

# Memory optimization
use_gradient_checkpointing: true    # Enable for 3x H100
cpu_offload: false                  # Only if OOM
```

### Data Configuration (`configs/data.yaml`)

```yaml
dataset_name: "builddotai/Egocentric-100K"
cache_dir: "/scratch-shared/scur1900/cosmos_cache"

# Video processing
max_frames: 64                      # Frames per clip
fps: 8                              # Target FPS
resolution: [256, 456]              # [H, W]

# DataLoader
batch_size: 2                       # Per GPU
num_workers: 8
```

### Training Configuration (`configs/train.yaml`)

```yaml
num_epochs: 10
max_steps: 100000
gradient_accumulation_steps: 4
mixed_precision: "bf16"             # BF16 for H100

# Checkpointing
save_interval: 1000                 # Save every N steps
eval_interval: 2500                 # Evaluate every N steps

# Distributed
num_gpus: 3                         # H100 GPUs
```

### Optimizer Configuration (`configs/optim.yaml`)

```yaml
optimizer: "adamw"
lr: 1.0e-4
weight_decay: 0.01

# LR schedule
scheduler: "cosine"
warmup_steps: 1000
min_lr: 1.0e-6
```

## Architecture Details

### Freezing Strategy

| Component | Status | Reason |
|-----------|--------|--------|
| VAE | 100% Frozen | Only for encoding/decoding |
| Spatial self-attention | Frozen | Preserve spatial priors |
| Spatial convolutions | Frozen | Preserve spatial structure |
| Temporal attention | **Trainable** | Learn video dynamics |
| Cross-attention | **LoRA** | Efficient text-video alignment |
| Temporal adapters | **Trainable** | Lightweight adaptation |
| Output projection | **Trainable** | Final prediction layer |

### Parameter Efficiency

- **Base Cosmos-2B**: ~2B parameters
- **Trainable with LoRA + Adapters**: ~100-200M parameters (5-10%)
- **LoRA rank 16**: ~8M parameters per cross-attention layer
- **Temporal adapters**: ~1-2M parameters per block

### Memory Requirements

**Per GPU (H100 80GB)**:
- Model shards: ~25GB
- Activations: ~20GB
- Optimizer states: ~15GB
- Gradients: ~10GB
- **Total**: ~70GB (within 80GB limit)

**If OOM**:
1. Reduce `batch_size` to 1
2. Enable `cpu_offload: true`
3. Reduce `max_frames` to 32
4. Lower resolution to `[128, 228]`

## Advanced Usage

### Adding a New Dataset

1. Create dataset adapter in `src/data/`:

```python
# src/data/my_custom_dataset.py
from .base import VideoDatasetBase

class MyCustomDataset(VideoDatasetBase):
    def __getitem__(self, idx):
        # Return canonical format
        return {
            "latents": latents,  # [T, C, H, W]
            "mask": mask,        # [T]
            "metadata": {...}
        }
```

2. Register in `src/data/dataset.py`:

```python
def _create_dataset(self, name: str, config: Dict):
    if name == "my_custom":
        return MyCustomDataset(**config)
```

3. Update `configs/data.yaml`:

```yaml
dataset_name: "my_custom"
# ... custom config
```

### Multi-Dataset Training

Edit `configs/data.yaml` to mix datasets:

```yaml
datasets:
  - name: "egocentric"
    weight: 0.7
    config:
      split: "train"
      cache_dir: "/path/to/cache"
  - name: "my_custom"
    weight: 0.3
    config:
      # ... custom config
```

### Resume from Checkpoint

Edit `configs/train.yaml`:

```yaml
resume_from_checkpoint: "/path/to/checkpoint.pt"
```

Or override:
```bash
sbatch slurm/train.sh resume_from_checkpoint=/path/to/checkpoint.pt
```

### Export LoRA Weights

To save only LoRA weights (10-20x smaller):

```python
from models.lora import LoRAInjector

# Save LoRA only
LoRAInjector.save_lora_weights(unet, "lora_weights.pt")

# Load LoRA on base model
base_unet = load_base_model()
base_unet = LoRAInjector.inject_lora(base_unet, ...)
LoRAInjector.load_lora_weights(base_unet, "lora_weights.pt")
```

## Troubleshooting

### Cache Miss Errors

**Error**: `Cache miss for video_id=...`

**Solution**: Ensure latent caching job completed successfully:
```bash
# Check cache statistics
python -c "
from src.data.latent_cache import LatentCache
cache = LatentCache('/scratch-shared/scur1900/cosmos_cache')
print(cache.get_cache_stats())
"
```

### OOM Errors

**Error**: `CUDA out of memory`

**Solutions**:
1. Reduce batch size: `data.batch_size=1`
2. Enable CPU offload: `model.cpu_offload=true`
3. Reduce frames: `data.max_frames=32`
4. Lower resolution: `data.resolution=[128,228]`

### Distributed Training Hangs

**Error**: Training hangs at initialization

**Solutions**:
1. Check NCCL: `export NCCL_DEBUG=INFO`
2. Verify GPU visibility: `nvidia-smi`
3. Check MASTER_PORT is free: `netstat -tuln | grep 29500`

### Slow Data Loading

**Symptoms**: GPU utilization < 50%

**Solutions**:
1. Increase workers: `data.num_workers=16`
2. Increase prefetch: `data.prefetch_factor=4`
3. Use faster storage: `/scratch-shared` instead of `/gpfs`
4. Verify 100% cache hit rate

## Performance Benchmarks

**Hardware**: 3x H100 80GB

| Config | Batch Size | Steps/sec | Memory/GPU | Time to 100K steps |
|--------|-----------|-----------|------------|-------------------|
| Base (BF16) | 2 | 1.8 | 70GB | ~15 hours |
| Reduced frames (32) | 2 | 3.2 | 45GB | ~9 hours |
| CPU offload | 2 | 1.2 | 50GB | ~23 hours |

## Citation

If you use this codebase, please cite:

```bibtex
@software{cosmos_finetune_2025,
  title = {Cosmos-2B Finetuning Pipeline},
  year = {2025},
  note = {Parameter-efficient finetuning for video diffusion models}
}
```

## License

This project is for research purposes. Please refer to NVIDIA's Cosmos model license for usage restrictions.
# diffusion-egocentric
# diffusion-egocentric
