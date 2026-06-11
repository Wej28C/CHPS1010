# start.ps1 — A lancer au debut de chaque session de travail
# Usage : .\start.ps1

$env:PYTHONUTF8 = "1"
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# Alias "py" -> le bon python du venv
Set-Alias py ".\.venv\Scripts\python.exe" -Scope Global
function python { & ".\.venv\Scripts\python.exe" @args }

Write-Host ""
Write-Host "=== Session Projet_140 demarree ==="
Write-Host ""
Write-Host "  Python : $( & ".\.venv\Scripts\python.exe" --version)"
Write-Host ""
Write-Host "  Commandes disponibles :"
Write-Host "    py scripts/fetch_data.py"
Write-Host "    py scripts/preprocess.py"
Write-Host "    py scripts/train.py --model xgboost --asset MC.PA"
Write-Host ""
Write-Host "  MLflow UI (dans un 2e terminal) :"
Write-Host "    py -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"
Write-Host "    Puis ouvrir : http://localhost:5000"
Write-Host ""
