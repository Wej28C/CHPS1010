#!/usr/bin/env bash
#SBATCH --account=r250123
#SBATCH --job-name=proj140-setup-venv
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --constraint="armgpu"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/setup_venv_%J.out
#SBATCH --error=logs/setup_venv_%J.err

# ============================================================================
# setup.sh
# A LANCER UNE SEULE FOIS avant tout job d'entrainement ou d'HPO.
# Cree un venv Python dans .venv/ avec toutes les dependances de
# requirements.txt. PyTorch est installe avec le support CUDA (noeuds armgpu).
#
# Usage :
#   mkdir -p logs
#   sbatch jobs/setup.sh
# ============================================================================

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p logs

VENV_DIR=".venv"

echo "============================================================"
echo " SETUP VENV Python -- Projet 140 MLOps"
echo " Job ID    : $SLURM_JOB_ID"
echo " Hostname  : $(hostname)"
echo " Arch      : $(uname -m)"
echo " Workdir   : $(pwd)"
echo " Venv dir  : $VENV_DIR"
echo "============================================================"

# Charger l'environnement ARM GPU via Spack
romeo_load_armgpu_env

# /iw66xwz = cuda ou toolkit armgpu (hash trouve avec : spack find --long)
# /oxq4fb7 = python@3.11 pour armgpu
spack load /iw66xwz
spack load /oxq4fb7

PYTHON=$(which python3)
echo "Python     : $PYTHON ($($PYTHON --version))"
echo "CUDA       : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'pas de GPU sur noeud de login')"

# Recreer le venv proprement
if [[ -d "$VENV_DIR" ]]; then
    echo "Suppression de l'ancien venv..."
    rm -rf "$VENV_DIR"
fi

echo "Creation du venv..."
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "Venv actif : $(which python3)"

pip install --upgrade pip --quiet

# Installer PyTorch CUDA EN PREMIER avant requirements.txt
# requirements.txt contient "torch==2.4.1" sans index URL -- si pip
# le voit en premier il prend la version CPU. En l'installant ici avec
# l'index CUDA, pip le detecte comme deja satisfait et le skip ensuite.
# Verifier la version CUDA avec : nvidia-smi | grep "CUDA Version"
# cu118 -> 11.8 / cu121 -> 12.1 / cu124 -> 12.4
echo ""
echo "Installation de PyTorch avec support CUDA..."
pip install --quiet \
    torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Installer le reste des dependances depuis requirements.txt
echo ""
echo "Installation de requirements.txt..."
pip install --quiet -r requirements.txt

# Verification
echo ""
echo "======== Verification ========"
python3 -c "
import sys, torch, numpy, pandas, mlflow, optuna, xgboost, sklearn, ta

print('Python     : ' + sys.version.split()[0])
print('torch      : ' + torch.__version__)
cuda_ok = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_ok else 'N/A'
print('CUDA dispo : ' + str(cuda_ok) + ' (' + gpu_name + ')')
print('numpy      : ' + numpy.__version__)
print('pandas     : ' + pandas.__version__)
print('mlflow     : ' + mlflow.__version__)
print('optuna     : ' + optuna.__version__)
print('xgboost    : ' + xgboost.__version__)
print('scikit-learn: ' + sklearn.__version__)
print('ta         : ' + ta.__version__)
print('TOUS OK')
"

deactivate
echo ""
echo "============================================================"
echo " VENV PRET dans : $(pwd)/$VENV_DIR"
echo " Lancer ensuite :"
echo "   sbatch jobs/train_all.sh"
echo "   sbatch jobs/hpo_array.sh"
echo "============================================================"
