#!/bin/bash
#SBATCH --job-name=planner-sft
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#SBATCH --time=08:00:00
#SBATCH --qos=standard

mkdir -p logs

export HF_HOME=/scratch/user/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

source ~/.bashrc
conda activate dl

cd ~/PlannerCode
python train.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --device cuda \
    --gradient_checkpointing False \
    --sft_samples 1000
