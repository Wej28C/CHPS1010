#!/usr/bin/env bash
#SBATCH --account=r250123
#SBATCH --job-name=proj140-train
#SBATCH --output=logs/train_%J.out
#SBATCH --error=logs/train_%J.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/project/r250123/CHPS1010/CHPS1010

set -euo pipefail

PROJECT_DIR=/project/r250123/CHPS1010/CHPS1010
VENV_DIR="$PROJECT_DIR/.venv"

echo "============================================================"
echo " ENTRAINEMENT -- Projet 140"
echo " Job ID  : $SLURM_JOB_ID"
echo " Host    : $(hostname)"
echo "============================================================"

source "$VENV_DIR/bin/activate"

export PYTHONUTF8=1
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

for MODEL in xgboost lstm tcn tft; do
    echo "--- $MODEL --- $(date)"
    python scripts/train.py --model "$MODEL" --asset all
done

echo "Fin entrainement -- $(date)"
