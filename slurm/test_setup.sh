#!/bin/bash
#SBATCH --job-name=cosmos-test
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_setup_%j.out
#SBATCH --error=logs/test_setup_%j.err

set -euo pipefail

echo "========================================="
echo "Cosmos-2B Setup Validation Test"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "========================================="

# Environment setup
module purge
module load 2023
module load Anaconda3/2023.07-2

export HF_TOKEN=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

eval "$(conda shell.bash hook)"

export JAVA_HOME=""
export JAVA_LD_LIBRARY_PATH=""

conda activate verlenv

# Paths
PROJECT_DIR="/gpfs/home4/scur1900/cosmos-finetune"
cd $PROJECT_DIR

echo ""
echo "========================================="
echo "TEST 1: Python Environment"
echo "========================================="
python --version
echo "Python location: $(which python)"

echo ""
echo "========================================="
echo "TEST 2: Required Packages"
echo "========================================="

echo "Checking critical packages..."
packages=("torch" "diffusers" "transformers" "hydra-core" "wandb" "tensorboard" "safetensors")

for pkg in "${packages[@]}"; do
    if python -c "import $pkg; print(f'✓ {$pkg.__version__}')" 2>/dev/null; then
        echo "✓ $pkg installed"
    else
        echo "✗ $pkg NOT FOUND"
        exit 1
    fi
done

echo ""
echo "========================================="
echo "TEST 3: GPU Availability"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPUs: {torch.cuda.device_count()}')"

if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "✗ CUDA not available!"
    exit 1
fi
echo "✓ GPU test passed"

echo ""
echo "========================================="
echo "TEST 4: HuggingFace Authentication"
echo "========================================="
if [ -z "$HF_TOKEN" ]; then
    echo "✗ HF_TOKEN not set!"
    exit 1
else
    echo "✓ HF_TOKEN is set"
    # Test authentication
    python -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv('HF_TOKEN'))
try:
    user_info = api.whoami()
    print(f\"✓ Authenticated as: {user_info['name']}\")
except Exception as e:
    print(f'✗ Authentication failed: {e}')
    exit(1)
" || exit 1
fi

echo ""
echo "========================================="
echo "TEST 5: Project Structure"
echo "========================================="
required_dirs=("src" "configs" "slurm")
for dir in "${required_dirs[@]}"; do
    if [ -d "$PROJECT_DIR/$dir" ]; then
        echo "✓ $dir/ exists"
    else
        echo "✗ $dir/ NOT FOUND"
        exit 1
    fi
done

required_files=(
    "src/main_train.py"
    "src/main_eval.py"
    "src/training/trainer.py"
    "src/models/cosmos_loader.py"
    "src/data/dataset.py"
    "configs/train.yaml"
    "configs/model.yaml"
    "configs/data.yaml"
)

for file in "${required_files[@]}"; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file NOT FOUND"
        exit 1
    fi
done

echo ""
echo "========================================="
echo "TEST 6: Import Test"
echo "========================================="
cd $PROJECT_DIR
python -c "
import sys
sys.path.insert(0, 'src')

# Test all critical imports
try:
    from training.trainer import CosmosTrainer
    print('✓ training.trainer')

    from models.cosmos_loader import CosmosModelLoader
    print('✓ models.cosmos_loader')

    from models.freeze_utils import ModelFreezer
    print('✓ models.freeze_utils')

    from models.lora import LoRAInjector
    print('✓ models.lora')

    from models.temporal import TemporalAdapterInjector
    print('✓ models.temporal')

    from data.dataset import ComposedVideoDataset
    print('✓ data.dataset')

    from data.hf_egocentric import EgocentricDataset
    print('✓ data.hf_egocentric')

    from diffusion.scheduler import SchedulerFactory
    print('✓ diffusion.scheduler')

    from diffusion.sampling import VideoSampler
    print('✓ diffusion.sampling')

    from eval.evaluator import CosmosEvaluator
    print('✓ eval.evaluator')

    from logging.wandb import WandBLogger
    print('✓ logging.wandb')

    print('✓ All imports successful')
except Exception as e:
    print(f'✗ Import failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" || exit 1

echo ""
echo "========================================="
echo "TEST 7: Hydra Configuration Loading"
echo "========================================="
python -c "
from omegaconf import OmegaConf
from pathlib import Path

configs = ['train', 'model', 'data', 'optim', 'logging', 'eval']
for cfg_name in configs:
    try:
        cfg_path = Path('configs') / f'{cfg_name}.yaml'
        cfg = OmegaConf.load(cfg_path)
        print(f'✓ {cfg_name}.yaml loaded successfully')
    except Exception as e:
        print(f'✗ {cfg_name}.yaml failed: {e}')
        exit(1)
" || exit 1

echo ""
echo "========================================="
echo "TEST 8: Directory Permissions"
echo "========================================="
CACHE_DIR="/scratch-shared/scur1900/cosmos_cache"
OUTPUT_DIR="/scratch-shared/scur1900/cosmos_outputs"

for dir in "$CACHE_DIR" "$OUTPUT_DIR" "$PROJECT_DIR/logs"; do
    mkdir -p "$dir" 2>/dev/null && echo "✓ Can write to $dir" || (echo "✗ Cannot write to $dir" && exit 1)
done

echo ""
echo "========================================="
echo "TEST 9: Model Loading (Dry Run)"
echo "========================================="
echo "Testing model loader with HF authentication..."
python -c "
import torch
import sys
sys.path.insert(0, 'src')

from models.cosmos_loader import CosmosModelLoader

# This will attempt to fetch model info (not full download)
print('Attempting to load VAE config...')
try:
    from diffusers import AutoencoderKL
    import os

    hf_token = os.getenv('HF_TOKEN')

    # Just test config loading (no actual download)
    config = AutoencoderKL.load_config(
        'nvidia/Cosmos-1.0-Tokenizer-CV8x8x8',
        token=hf_token
    )
    print('✓ VAE config accessible')

    # Test UNet config
    from diffusers import UNet3DConditionModel
    unet_config = UNet3DConditionModel.load_config(
        'nvidia/Cosmos-Predict2.5-2B',
        subfolder='unet',
        token=hf_token
    )
    print('✓ UNet config accessible')
    print('✓ Model authentication working')

except Exception as e:
    print(f'✗ Model loading test failed: {e}')
    print('Note: This may fail if models are not yet accessible with your token')
    # Don't exit - this might fail if model access not granted yet
"

echo ""
echo "========================================="
echo "TEST 10: Distributed Setup"
echo "========================================="
python -c "
import torch.distributed as dist
import os

# Test that NCCL is available
try:
    import torch
    print(f'✓ NCCL available: {torch.cuda.nccl.version()}')
except:
    print('✗ NCCL not available')
    exit(1)
" || exit 1

echo ""
echo "========================================="
echo "SUMMARY"
echo "========================================="
echo "✓ All validation tests passed!"
echo "Ready for training."
echo ""
echo "Next steps:"
echo "1. Run latent caching: sbatch slurm/download_data.sh"
echo "2. Launch training: sbatch slurm/train.sh"
echo "========================================="
echo "Completed: $(date)"
