#!/bin/bash
#SBATCH --job-name=proj140_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Charger l'environnement via Spack
spack load python@3.11
spack load cuda@11.8

source ~/Projet_140/.venv/bin/activate
cd ~/Projet_140

export PYTHONUTF8=1
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

echo "=== Début entraînement — $(date) ==="
echo "GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

for MODEL in xgboost lstm tcn tft; do
    echo "--- $MODEL ---"
    python scripts/train.py --model $MODEL --asset all
done

echo "=== Fin entraînement — $(date) ==="
