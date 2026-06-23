#!/usr/bin/env bash
#SBATCH --account=r250123
#SBATCH --job-name=proj140-hpo
#SBATCH --output=logs/hpo_%A_%a.out
#SBATCH --error=logs/hpo_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --chdir=/project/r250123/CHPS1010/CHPS1010

set -euo pipefail

PROJECT_DIR=/project/r250123/CHPS1010/CHPS1010
VENV_DIR="$PROJECT_DIR/.venv"

MODELS=(xgboost lstm tcn tft)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo " HPO -- $MODEL | Job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID"
echo " Host : $(hostname)"
echo "============================================================"

spack load /2celb2j
source "$VENV_DIR/bin/activate"

export PYTHONUTF8=1
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export OPTUNA_STORAGE="sqlite:///optuna.db"

python scripts/hpo.py --model "$MODEL" --asset all --trials 100

echo "Fin HPO $MODEL -- $(date)"
