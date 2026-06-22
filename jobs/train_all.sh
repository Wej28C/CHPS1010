#!/usr/bin/env bash
#SBATCH --account=r250123
#SBATCH --job-name=proj140-train
#SBATCH --output=logs/train_%J.out
#SBATCH --error=logs/train_%J.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --constraint=armgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/project/r250123/CHPS1010/CHPS1010

set -euo pipefail

PROJECT_DIR=/project/r250123/CHPS1010/CHPS1010
VENV_DIR="$PROJECT_DIR/.venv"

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
