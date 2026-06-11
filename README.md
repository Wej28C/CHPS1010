# Pipeline MLOps — Prédiction de séries temporelles financières (Luxe/Joaillerie)

Université de Reims Champagne-Ardenne — Projet Intégrateur 140h — 2025/2026

## Description

Pipeline automatisé de prédiction de rendements directionnels J+1 sur 5 actifs du secteur luxe coté,
comparant 4 architectures Deep Learning (XGBoost, LSTM, TCN, TFT) avec orchestration MLOps complète.

## Installation

```bash
git clone <repo-url>
cd Projet_140

# Créer et activer le venv
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux/Mac (ROMEO)

pip install -r requirements.txt
```

## Lancer MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Ouvrir http://localhost:5000
```

## Reproduire le pipeline complet

```bash
dvc repro
```

## Structure

```
Projet_140/
├── data/           # Données (versionnées DVC)
│   ├── raw/        # OHLCV bruts
│   └── processed/  # Features calculées
├── models/         # Architectures ML
├── scripts/        # fetch_data, preprocess, train, report
├── tests/          # Tests unitaires pytest
├── slurm/          # Jobs SLURM pour ROMEO
├── notebooks/      # EDA
├── reports/        # Rapports générés
└── .github/        # CI/CD GitHub Actions
```

## Actifs étudiés

| Ticker | Entreprise | Bourse |
|--------|-----------|--------|
| MC.PA | LVMH | Euronext Paris |
| CFR.SW | Richemont | SIX Swiss |
| RMS.PA | Hermès | Euronext Paris |
| BRBY.L | Burberry | LSE |
| MONO.PA | L'Oréal (référence) | Euronext Paris |
