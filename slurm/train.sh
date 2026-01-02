#!/bin/bash
#SBATCH --job-name=cosmos-train
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

# Environment
module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"

export JAVA_HOME=""
export JAVA_LD_LIBRARY_PATH=""

conda activate verlenv

# Install/upgrade required dependencies
echo "Installing/upgrading dependencies..."
pip install --upgrade pip
pip install --upgrade -r /gpfs/home4/scur1900/cosmos-finetune/requirements.txt
echo "Dependencies installed successfully"

export HF_TOKEN=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

# Set environment variables
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=0

# Paths
PROJECT_DIR="/gpfs/home4/scur1900/cosmos-finetune"
OUTPUT_DIR="/scratch-shared/scur1900/cosmos_outputs"
mkdir -p $OUTPUT_DIR
mkdir -p $PROJECT_DIR/logs

echo "========================================="
echo "Training Job Info"
echo "========================================="
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Job ID: $SLURM_JOB_ID"
echo "Output: $OUTPUT_DIR"
echo "========================================="
nvidia-smi
echo "========================================="

# Distributed training setup
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_GPUS_PER_NODE

# Launch training with torchrun
cd $PROJECT_DIR

torchrun \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/main_train.py \
    output_dir=$OUTPUT_DIR \
    data.cache_dir=/scratch-shared/scur1900/cosmos_cache

echo "Training complete!"
