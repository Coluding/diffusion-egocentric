#!/bin/bash
#SBATCH --job-name=cosmos-eval
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail

# Environment
module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"

export JAVA_HOME=""
export JAVA_LD_LIBRARY_PATH=""

conda activate verlenv

# Paths
PROJECT_DIR="/gpfs/home4/scur1900/cosmos-finetune"
CHECKPOINT_PATH=$1  # Pass as argument
OUTPUT_DIR="/scratch-shared/scur1900/cosmos_eval"
mkdir -p $OUTPUT_DIR
mkdir -p $PROJECT_DIR/logs

echo "========================================="
echo "Evaluation Job"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "========================================="

cd $PROJECT_DIR

python src/main_eval.py \
    checkpoint_path=$CHECKPOINT_PATH \
    output_dir=$OUTPUT_DIR \
    num_samples=100 \
    num_inference_steps=50

echo "Evaluation complete! Results in $OUTPUT_DIR"
