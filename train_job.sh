#!/bin/bash

#SBATCH --job-name=planner-sft
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%j.log

set -e

ml purge
ml GCCcore/13.3.0
ml Python/3.12.3
source /scratch/user/$USER/venv_dl/bin/activate

export HF_HOME=/scratch/user/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONUNBUFFERED=1

mkdir -p logs

cd /scratch/user/$USER/task-graph-gar

echo "Starting SFT training..."
python train.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --device cuda \
    --gradient_checkpointing False \
    --sft_samples 1000 \
    --batch_size 4
echo "Done!"
