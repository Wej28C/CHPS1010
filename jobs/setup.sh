#!/usr/bin/env bash
#SBATCH --account=r250123
#SBATCH --job-name=proj140-setup-venv
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/setup_venv_%J.out
#SBATCH --error=logs/setup_venv_%J.err
#SBATCH --chdir=/project/r250123/CHPS1010/CHPS1010

set -euo pipefail

PROJECT_DIR=/project/r250123/CHPS1010/CHPS1010
VENV_DIR="$PROJECT_DIR/.venv"

export TMPDIR="$PROJECT_DIR/.tmp_pip"
mkdir -p "$TMPDIR"
export PIP_NO_CACHE_DIR=1

echo "============================================================"
echo " SETUP VENV -- Projet 140 MLOps"
echo " Job ID  : $SLURM_JOB_ID"
echo " Host    : $(hostname)"
echo " Arch    : $(uname -m)"
echo "============================================================"

spack load /2celb2j

PYTHON=$(which python3)
echo "Python : $PYTHON ($($PYTHON --version))"

if [[ -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
fi

$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip --no-cache-dir --quiet
pip install --no-cache-dir -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "======== Verification ========"
python3 -c "
import sys, torch, numpy, pandas, mlflow, optuna, xgboost, sklearn, ta
print('Python      : ' + sys.version.split()[0])
print('torch       : ' + torch.__version__)
print('numpy       : ' + numpy.__version__)
print('pandas      : ' + pandas.__version__)
print('mlflow      : ' + mlflow.__version__)
print('optuna      : ' + optuna.__version__)
print('xgboost     : ' + xgboost.__version__)
print('scikit-learn: ' + sklearn.__version__)
print('ta          : ' + ta.__version__)
print('TOUS OK')
"

deactivate
rm -rf "$TMPDIR"

echo ""
echo "============================================================"
echo " VENV PRET : $VENV_DIR"
echo " Lancer ensuite : sbatch jobs/train_all.sh"
echo "============================================================"
