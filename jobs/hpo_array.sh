#!/usr/bin/env bash
#SBATCH --account=r250123
#SBATCH --job-name=proj140-hpo
#SBATCH --output=logs/hpo_%A_%a.out
#SBATCH --error=logs/hpo_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --constraint=armgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --chdir=/project/r250123/CHPS1010/CHPS1010

set -euo pipefail

PROJECT_DIR=/project/r250123/CHPS1010/CHPS1010
VENV_DIR="$PROJECT_DIR/.venv"

MODELS=(xgboost lstm tcn tft)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo " HPO -- $MODEL"
echo " Job ID   : $SLURM_JOB_ID  Array ID : $SLURM_ARRAY_TASK_ID"
echo " Hostname : $(hostname)"
echo " GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo " Venv     : $VENV_DIR"
echo "============================================================"

romeo_load_armgpu_env
spack load /iw66xwz
spack load /oxq4fb7
spack load py-torch

source "$VENV_DIR/bin/activate"

export PYTHONUTF8=1
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export OPTUNA_STORAGE="sqlite:///optuna.db"
export PIP_NO_CACHE_DIR=1

python scripts/hpo.py --model "$MODEL" --asset all --trials 100

echo ""
echo "============================================================"
echo " Fin HPO $MODEL -- $(date)"
echo "============================================================"
