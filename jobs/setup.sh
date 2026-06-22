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
# Strategie quota : venv installe sur SCRATCH (pas sur HOME).
# Le HOME ROMEO a un quota tres limite ; le SCRATCH est genereux.
#
# torch est fourni par spack (ARM+CUDA, pas disponible sur PyPI aarch64).
# Tous les autres paquets vont dans $VENV_DIR sur SCRATCH, sans cache pip.
#
# Usage :
#   mkdir -p logs
#   sbatch jobs/setup.sh
# ============================================================================

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Repertoire du venv sur GPFS (2.3 PB libres) hors du HOME systeme
# /gpfs/home/wbouchhioua est le HOME GPFS -- filesystem distinct du HOME systeme
VENV_DIR="/project/r250123/proj140_venv"

# Forcer pip a ne pas ecrire de cache (evite de remplir HOME)
export PIP_NO_CACHE_DIR=1
export PIP_CACHE_DIR="/tmp/pip_cache_$$"

echo "============================================================"
echo " SETUP VENV Python -- Projet 140 MLOps"
echo " Job ID    : $SLURM_JOB_ID"
echo " Hostname  : $(hostname)"
echo " Arch      : $(uname -m)"
echo " Workdir   : $(pwd)"
echo " Venv dir  : $VENV_DIR"
echo " SCRATCH   : /project/r250123"
echo " Quota HOME : $(quota -s 2>/dev/null | tail -1 || echo 'N/A')"
echo "============================================================"

# Charger l'environnement ARM GPU via Spack
romeo_load_armgpu_env
spack load /iw66xwz
spack load /oxq4fb7

# Charger PyTorch depuis Spack (compile ARM + CUDA)
spack load py-torch

PYTHON=$(which python3)
echo "Python     : $PYTHON ($($PYTHON --version))"
echo "torch spack: $(python3 -c 'import torch; print(torch.__version__)')"

# Supprimer l'ancien venv si necessaire
if [[ -d "$VENV_DIR" ]]; then
    echo "Suppression de l'ancien venv sur SCRATCH..."
    rm -rf "$VENV_DIR"
fi

# Creer le venv sur SCRATCH avec --system-site-packages
# --system-site-packages : herite de torch (et autres) depuis spack
echo "Creation du venv sur SCRATCH..."
$PYTHON -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "Venv actif : $(which python3)"

# Mettre a jour pip sans cache
pip install --upgrade pip --no-cache-dir --quiet

# Installer uniquement les paquets PAS disponibles via spack
# torch est exclu (fourni par spack)
# numpy, scipy peuvent etre deja la via spack -- on installe au cas ou
echo ""
echo "Installation des dependances sans cache (SCRATCH)..."
grep -vE "^torch" "$SLURM_SUBMIT_DIR/requirements.txt" > /tmp/req_no_torch_$$.txt

pip install --no-cache-dir --quiet -r /tmp/req_no_torch_$$.txt
rm /tmp/req_no_torch_$$.txt

# Nettoyer le cache temporaire
rm -rf "$PIP_CACHE_DIR"

# Creer un fichier d'activation facile a sourcer
ACTIVATE_SCRIPT="$SLURM_SUBMIT_DIR/jobs/activate_venv.sh"
cat > "$ACTIVATE_SCRIPT" << ACTIVATE_EOF
#!/usr/bin/env bash
# Source ce fichier pour activer le venv du projet
romeo_load_armgpu_env
spack load /iw66xwz
spack load /oxq4fb7
spack load py-torch
source "${VENV_DIR}/bin/activate"
export PYTHONUTF8=1
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export OPTUNA_STORAGE="sqlite:///optuna.db"
export PIP_NO_CACHE_DIR=1
ACTIVATE_EOF
chmod +x "$ACTIVATE_SCRIPT"

# Verification finale
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
echo ""
echo " Pour activer manuellement sur le login node :"
echo "   source jobs/activate_venv.sh"
echo ""
echo " Lancer ensuite :"
echo "   sbatch jobs/train_all.sh"
echo "   sbatch jobs/hpo_array.sh"
echo "============================================================"
