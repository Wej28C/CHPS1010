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

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

VENV_DIR="/project/r250123/proj140_venv"

echo "============================================================"
echo " ENTRAINEMENT -- Projet 140"
echo " Job ID   : $SLURM_JOB_ID"
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
export PIP_NO_CACHE_DIR=1

for MODEL in xgboost lstm tcn tft; do
    echo ""
    echo "--- $MODEL --- $(date)"
    python scripts/train.py --model "$MODEL" --asset all
done

echo ""
echo "============================================================"
echo " Fin entrainement -- $(date)"
echo "============================================================"
