#!/usr/bin/env bash
#SBATCH --account=r250123
#SBATCH --job-name=proj140-train
#SBATCH --output=logs/train_%J.out
#SBATCH --error=logs/train_%J.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --constraint="armgpu"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

# ============================================================================
# train_all.sh — Entraîne les 4 modèles sur les 5 actifs.
# Pré-requis : sbatch jobs/setup.sh doit avoir été exécuté.
# ============================================================================

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "============================================================"
echo " ENTRAÎNEMENT — Projet 140"
echo " Job ID   : $SLURM_JOB_ID"
echo " Hostname : $(hostname)"
echo " GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

romeo_load_armgpu_env
spack load /<HASH_ARM_PY311>

source .venv/bin/activate

export PYTHONUTF8=1
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

for MODEL in xgboost lstm tcn tft; do
    echo ""
    echo "--- $MODEL --- $(date)"
    python scripts/train.py --model $MODEL --asset all
done

echo ""
echo "============================================================"
echo " Fin entraînement — $(date)"
echo "============================================================"
