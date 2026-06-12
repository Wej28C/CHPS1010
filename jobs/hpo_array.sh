#!/bin/bash
#SBATCH --job-name=proj140_hpo
#SBATCH --output=logs/hpo_%A_%a.out
#SBATCH --error=logs/hpo_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3          # 4 jobs en parallèle, un par modèle

# Array job SLURM : SLURM lance 4 copies de ce script simultanément,
# chacune avec un SLURM_ARRAY_TASK_ID différent (0, 1, 2, 3).
# Chaque copie traite un modèle différent en parallèle → 4x plus rapide
# que de les lancer séquentiellement.

spack load python@3.11
spack load cuda@11.8

source ~/Projet_140/.venv/bin/activate
cd ~/Projet_140

export PYTHONUTF8=1
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export OPTUNA_STORAGE=sqlite:///optuna.db

MODELS=(xgboost lstm tcn tft)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=== HPO $MODEL — $(date) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

python scripts/hpo.py --model $MODEL --asset all --trials 100

echo "=== Fin HPO $MODEL — $(date) ==="
