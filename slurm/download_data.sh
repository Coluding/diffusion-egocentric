#!/bin/bash
#SBATCH --job-name=cosmos-cache
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/cache_%j.out
#SBATCH --error=logs/cache_%j.err

set -euo pipefail

# Environment setup
module purge
module load 2023
module load Anaconda3/2023.07-2

export HF_TOKEN=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

eval "$(conda shell.bash hook)"

export JAVA_HOME=""
export JAVA_LD_LIBRARY_PATH=""

conda activate verlenv

# Paths (define early so we can use them)
PROJECT_DIR="/gpfs/home4/scur1900/cosmos-finetune"
CACHE_DIR="/scratch-shared/scur1900/cosmos_cache"

# Set HuggingFace cache to scratch (not home directory)
export HF_HOME="/scratch-shared/scur1900/huggingface_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Install/upgrade required dependencies
#echo "Installing/upgrading dependencies..."
#pip install --upgrade pip
#pip install --upgrade -r $PROJECT_DIR/requirements.txt
#echo "Dependencies installed successfully"

# HuggingFace authentication (required for Cosmos model access)
# Set your token: export HF_TOKEN="hf_..." before running
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Model download may fail for gated models."
    echo "Set it with: export HF_TOKEN='hf_...' before running sbatch"
fi
mkdir -p $CACHE_DIR
mkdir -p $PROJECT_DIR/logs

echo "========================================="
echo "Starting latent caching job"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Cache dir: $CACHE_DIR"
echo "========================================="

# Run caching script
# Caching only 1/3 of the dataset (~64,300 videos out of ~192,900)
python $PROJECT_DIR/src/data/cache_latents.py \
    --dataset builddotai/Egocentric-10K \
    --split train \
    --cache-dir $CACHE_DIR \
    --batch-size 2 \
    --num-workers 4 \
    --max-samples 64300 \
    --vae-model madebyollin/sdxl-vae-fp16-fix

echo "Caching complete!"
