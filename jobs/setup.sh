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
# ----------------------------------------------------------------------------
# A LANCER UNE SEULE FOIS avant tout job d'entraînement ou d'HPO.
# Crée un venv Python dans .venv/ avec toutes les dépendances de
# requirements.txt. PyTorch est installé avec le support CUDA (nœuds armgpu).
#
# Usage :
#   mkdir -p logs
#   sbatch jobs/setup.sh
#
# Ensuite :
#   sbatch jobs/train_all.sh
#   sbatch jobs/hpo_array.sh
# ============================================================================

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p logs

VENV_DIR=".venv"

echo "============================================================"
echo " SETUP VENV Python — Projet 140 MLOps"
echo " Job ID    : $SLURM_JOB_ID"
echo " Hostname  : $(hostname)"
echo " Arch      : $(uname -m)"
echo " Workdir   : $(pwd)"
echo " Venv dir  : $VENV_DIR"
echo "============================================================"

# ── Charger l'environnement ARM GPU ────────────────────────────────────────
# romeo_load_armgpu_env configure les variables Spack pour les nœuds armgpu.
romeo_load_armgpu_env

# Charger Python 3.11 compilé pour armgpu.
# Pour trouver le bon hash sur ROMEO :
#   spack find --long python arch=linux-rhel8-aarch64
# Remplacer le hash ci-dessous par celui retourné.
spack load /<HASH_ARM_PY311>

PYTHON=$(which python3)
echo "Python     : $PYTHON ($($PYTHON --version))"
echo "CUDA       : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'nvidia-smi indisponible sur nœud setup')"

# ── Recréer le venv proprement ─────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    echo "Suppression de l'ancien venv..."
    rm -rf "$VENV_DIR"
fi

echo "Création du venv..."
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "Venv actif : $(which python3)"

pip install --upgrade pip --quiet

# ── PyTorch avec support CUDA ──────────────────────────────────────────────
# On installe torch EN PREMIER avec l'index CUDA avant requirements.txt.
# Raison : requirements.txt contient "torch==2.4.1" sans index URL.
# Si pip le voit en premier il installe la version CPU (~200 Mo).
# En l'installant ici avec l'index CUDA, pip détecte qu'il est déjà
# satisfait et le skip lors du `pip install -r requirements.txt`.
#
# Adapter cu121/cu118 selon la version CUDA du nœud :
#   nvidia-smi | grep "CUDA Version"
#   cu118 → CUDA 11.8  /  cu121 → CUDA 12.1  /  cu124 → CUDA 12.4
echo ""
echo "Installation de PyTorch avec support CUDA..."
pip install --quiet \
    torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Reste des dépendances ──────────────────────────────────────────────────
echo ""
echo "Installation de requirements.txt..."
pip install --quiet -r requirements.txt

# ── Vérification ───────────────────────────────────────────────────────────
echo ""
echo "======== Vérification ========"
python3 -c "
import sys, torch, numpy, pandas, mlflow, optuna, xgboost, sklearn, ta

print(f'Python     : {sys.version.split()[0]}')
print(f'torch      : {torch.__version__}')
print(f'CUDA dispo : {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
print(f'numpy      : {numpy.__version__}')
print(f'pandas     : {pandas.__version__}')
print(f'mlflow     : {mlflow.__version__}')
print(f'optuna     : {optuna.__version__}')
print(f'xgboost    : {xgboost.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'ta         : {ta.__version__}')
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
