#!/usr/bin/env bash
#SBATCH --account=r250123
#SBATCH --job-name=proj140-setup-venv
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --constraint=armgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/setup_venv_%J.out
#SBATCH --error=logs/setup_venv_%J.err
#SBATCH --chdir=/project/r250123/CHPS1010/CHPS1010

# ============================================================================
# setup.sh
# A LANCER UNE SEULE FOIS avant les jobs d'entrainement ou d'HPO.
#
# Strategie torch sur ROMEO armgpu (aarch64) :
#   PyPI ne fournit pas de wheels torch pour ARM -> pip install torch echoue.
#   Solution : spack load py-torch (compile pour ARM + CUDA sur ROMEO) puis
#   venv avec --system-site-packages pour heriter de torch via spack.
#   Le reste des dependances (mlflow, optuna, etc.) vient de pip.
#
# Usage :
#   mkdir -p logs
#   sbatch jobs/setup.sh
# ============================================================================

set -euo pipefail

PROJECT_DIR=/project/r250123/CHPS1010/CHPS1010
VENV_DIR="$PROJECT_DIR/.venv"

export PIP_NO_CACHE_DIR=1
export PIP_CACHE_DIR="/tmp/pip_cache_$$"

echo "============================================================"
echo " SETUP VENV Python -- Projet 140 MLOps"
echo " Job ID    : $SLURM_JOB_ID"
echo " Hostname  : $(hostname)"
echo " Arch      : $(uname -m)"
echo " Project   : $PROJECT_DIR"
echo " Venv dir  : $VENV_DIR"
echo "============================================================"

romeo_load_armgpu_env
spack load /iw66xwz
spack load /oxq4fb7
spack load py-torch

PYTHON=$(which python3)
echo "Python     : $PYTHON ($($PYTHON --version))"
echo "torch spack: $(python3 -c 'import torch; print(torch.__version__)')"

if [[ -d "$VENV_DIR" ]]; then
    echo "Suppression de l'ancien venv..."
    rm -rf "$VENV_DIR"
fi

echo "Creation du venv (--system-site-packages pour heriter de torch spack)..."
$PYTHON -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "Venv actif : $(which python3)"

pip install --upgrade pip --no-cache-dir --quiet

echo ""
echo "Installation des dependances (torch exclu -- fourni par spack)..."
grep -vE "^torch" "$PROJECT_DIR/requirements.txt" > /tmp/req_no_torch_$$.txt
pip install --no-cache-dir --quiet -r /tmp/req_no_torch_$$.txt
rm /tmp/req_no_torch_$$.txt
rm -rf "$PIP_CACHE_DIR"

echo ""
echo "======== Verification ========"
python3 -c "
import sys, torch, numpy, pandas, mlflow, optuna, xgboost, sklearn, ta

print('Python      : ' + sys.version.split()[0])
print('torch       : ' + torch.__version__)
cuda_ok = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_ok else 'N/A'
print('CUDA dispo  : ' + str(cuda_ok) + ' (' + gpu_name + ')')
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
echo ""
echo "============================================================"
echo " VENV PRET dans : $VENV_DIR"
echo " Lancer ensuite :"
echo "   sbatch jobs/train_all.sh"
echo "   sbatch jobs/hpo_array.sh"
echo "============================================================"
