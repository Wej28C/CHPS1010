#!/bin/bash
# setup_romeo.sh — Initialise l'environnement sur ROMEO (Spack)
#
# Usage : source jobs/setup_romeo.sh
# (source et non bash — pour que les exports restent dans le shell courant)
#
# À exécuter UNE FOIS par session SSH, avant tout sbatch.

# ── 1. Charger Python et CUDA via Spack ────────────────────────────────────
# Adapter les versions selon : spack find python && spack find cuda
spack load python@3.11
spack load cuda@11.8       # vérifier avec nvidia-smi la version disponible

# ── 2. Activer le venv ──────────────────────────────────────────────────────
PROJET_DIR="$HOME/Projet_140"
source "$PROJET_DIR/.venv/bin/activate"

# ── 3. Variables d'environnement ─────────────────────────────────────────────
export PYTHONUTF8=1
export MLFLOW_TRACKING_URI="sqlite:///$PROJET_DIR/mlflow.db"
export OPTUNA_STORAGE="sqlite:///$PROJET_DIR/optuna.db"
export PYTHONPATH="$PROJET_DIR:$PYTHONPATH"

echo "[OK] Environnement ROMEO prêt — Python $(python --version)"
echo "[OK] GPU disponible : $(python -c 'import torch; print(torch.cuda.is_available())')"
