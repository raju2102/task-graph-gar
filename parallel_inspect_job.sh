#!/bin/bash

#SBATCH --job-name=planner-parallel-inspect
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/%j_parallel_inspect.log

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

if [ -z "$1" ]; then
    echo "No problem given — randomly sampling a combined problem from the dataset..."
    python inspect_data.py --parallel --model_path ./planner_model_v3 --n 1
else
    echo "Problem: $1"
    python inspect_data.py --parallel --model_path ./planner_model_v3 --problem "$1"
fi
echo "Done!"
