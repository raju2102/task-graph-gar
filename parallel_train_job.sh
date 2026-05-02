#!/bin/bash

#SBATCH --job-name=planner-parallel-sft
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%j_parallel_sft.log

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

echo "Starting parallel SFT fine-tuning from checkpoint..."
python train.py \
    --mode parallel \
    --checkpoint_path ./planner_model \
    --save_path ./planner_model_v2 \
    --parallel_samples 500 \
    --linear_samples 300 \
    --device cuda \
    --gradient_checkpointing False \
    --batch_size 4
echo "Done!"
