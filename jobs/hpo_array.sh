#!/usr/bin/env bash
#SBATCH --account=r250123
#SBATCH --job-name=proj140-hpo
#SBATCH --output=logs/hpo_%A_%a.out
#SBATCH --error=logs/hpo_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --constraint="armgpu"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3

# ============================================================================
# hpo_array.sh -- HPO Optuna en parallele (un job par modele).
#
# SLURM array : 4 copies tournent simultanement avec SLURM_ARRAY_TASK_ID
# valant 0, 1, 2 ou 3. Chaque copie traite un modele different.
#
# Pre-requis : setup.sh puis train_all.sh
# ============================================================================

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p logs

MODELS=(xgboost lstm tcn tft)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo " HPO -- $MODEL"
echo " Job ID   : $SLURM_JOB_ID  Array ID : $SLURM_ARRAY_TASK_ID"
echo " Hostname : $(hostname)"
echo " GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

romeo_load_armgpu_env
spack load /iw66xwz
spack load /oxq4fb7

source .venv/bin/activate

export PYTHONUTF8=1
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export OPTUNA_STORAGE="sqlite:///optuna.db"

python scripts/hpo.py --model $MODEL --asset all --trials 100

echo ""
echo "============================================================"
echo " Fin HPO $MODEL -- $(date)"
echo "============================================================"
