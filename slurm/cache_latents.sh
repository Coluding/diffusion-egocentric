#!/bin/bash
#SBATCH --job-name=cosmos-cache-latents
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/cache_latents_%j.out
#SBATCH --error=logs/cache_latents_%j.err

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

# Paths
PROJECT_DIR="/gpfs/home4/scur1900/cosmos-finetune"
CACHE_DIR="/scratch-shared/scur1900/cosmos_cache"

# Set HuggingFace cache to scratch (data already downloaded)
export HF_HOME="/scratch-shared/scur1900/huggingface_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# HuggingFace authentication
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. VAE model download may fail."
    echo "Set it with: export HF_TOKEN='hf_...' before running sbatch"
fi

# Create necessary directories
mkdir -p $CACHE_DIR
mkdir -p $PROJECT_DIR/logs

echo "========================================="
echo "Latent Caching Job (No Download)"
echo "========================================="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Dataset cache: $HF_DATASETS_CACHE"
echo "Latent cache dir: $CACHE_DIR"
echo "========================================="

# Check disk space before starting
echo ""
echo "Disk space check:"
du -sh $HF_DATASETS_CACHE
du -sh $CACHE_DIR
df -h /scratch-shared | grep scratch
echo ""

# -u flag for unbuffered output (logs appear immediately)
# Reduced batch size and workers to avoid OOM
# Note: trust_remote_code removed (deprecated in newer datasets library)
python -u $PROJECT_DIR/src/data/cache_latents.py \
    --dataset builddotai/Egocentric-10K \
    --split train \
    --cache-dir $CACHE_DIR \
    --vae-model madebyollin/sdxl-vae-fp16-fix \
    --batch-size 1 \
   # --num-workers 2 \
    --resolution 256 456 \
    --fps 8

echo ""
echo "========================================="
echo "Latent caching complete!"
echo "========================================="

# Final disk space check
echo ""
echo "Final disk space:"
du -sh $CACHE_DIR
df -h /scratch-shared | grep scratch
echo ""
