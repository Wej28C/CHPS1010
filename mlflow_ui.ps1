# Lance le serveur MLflow UI
# Usage : .\mlflow_ui.ps1
# Puis ouvrir http://localhost:5000

$env:PYTHONUTF8 = "1"
Write-Host "Demarrage MLflow UI sur http://localhost:5000 ..."
Write-Host "Ctrl+C pour arreter"
& ".\.venv\Scripts\python.exe" -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
