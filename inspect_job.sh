#!/bin/bash

#SBATCH --job-name=planner-inspect
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/%j_inspect.log

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

echo "Running inspection..."
python inspect_data.py --model_path ./planner_model --n 5
echo "Done!"
