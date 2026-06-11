"""
train.py — Point d'entrée unifié pour entraîner n'importe quel modèle.

Usage :
    python scripts/train.py --model xgboost --asset MC.PA
    python scripts/train.py --model lstm --asset RMS.PA --window 30
    python scripts/train.py --model xgboost --asset all   # tous les actifs

Ce script :
  1. Charge les séquences prétraitées depuis data/processed/
  2. Instancie le bon modèle selon --model
  3. Entraîne le modèle
  4. Évalue sur le test set
  5. Logue tous les paramètres, métriques et artefacts dans MLflow
  6. Sauvegarde le modèle entraîné dans models/saved/
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
from dotenv import load_dotenv

# Ajouter la racine du projet au PYTHONPATH pour les imports relatifs
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from models.tcn_model import TCNModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models/saved")
TICKERS = ["MC.PA", "CFR.SW", "RMS.PA", "BRBY.L", "OR.PA"]

MODEL_REGISTRY = {
    "xgboost": XGBoostModel,
    "lstm": LSTMModel,
    "tcn": TCNModel,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Entraîner un modèle de prédiction")
    parser.add_argument(
        "--model",
        required=True,
        choices=["xgboost", "lstm", "tcn", "tft"],
        help="Architecture du modèle",
    )
    parser.add_argument(
        "--asset",
        required=True,
        help='Ticker de l\'actif (ex: MC.PA) ou "all" pour tous',
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Taille de la fenêtre glissante (doit correspondre au prétraitement)",
    )
    parser.add_argument(
        "--experiment",
        default="projet140",
        help="Nom de l'expérience MLflow",
    )
    return parser.parse_args()


def load_sequences(ticker: str, processed_dir: Path):
    """
    Charge les séquences numpy depuis data/processed/<TICKER>/.

    Ces fichiers ont été créés par preprocess.py :
      X_train.npy : (N_train, window, n_features)
      y_train.npy : (N_train,)
      etc.
    """
    safe = ticker.replace(".", "_")
    ticker_dir = processed_dir / safe

    if not ticker_dir.exists():
        raise FileNotFoundError(
            f"Données non trouvées pour {ticker} dans {ticker_dir}. "
            "Lance d'abord : python scripts/preprocess.py"
        )

    data = {}
    for split in ["train", "val", "test"]:
        data[f"X_{split}"] = np.load(ticker_dir / f"X_{split}.npy")
        data[f"y_{split}"] = np.load(ticker_dir / f"y_{split}.npy")

    logger.info(
        f"{ticker} — X_train{data['X_train'].shape}, "
        f"X_val{data['X_val'].shape}, X_test{data['X_test'].shape}"
    )
    return data


def get_git_commit() -> str:
    """Retourne le hash court du commit courant pour la traçabilité MLflow."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def train_one(model_name: str, ticker: str, window: int,
              experiment_name: str, processed_dir: Path):
    """
    Entraîne un modèle sur un actif et logue tout dans MLflow.

    Structure d'un run MLflow :
    ────────────────────────────
    Run "xgboost_MC.PA"
    ├── Parameters  : hyperparamètres + contexte (actif, fenêtre, commit git)
    ├── Metrics     : train_acc, val_acc, test_acc, sharpe, mae, rmse
    └── Artifacts   : modèle sérialisé (.pkl), feature importances (.npy)

    Chaque run est identifié par son nom et son experiment.
    L'UI MLflow permet de les comparer visuellement.
    """
    logger.info(f"\n{'='*55}")
    logger.info(f"  Modèle: {model_name.upper()} | Actif: {ticker}")
    logger.info(f"{'='*55}")

    # 1. Charger les données
    data = load_sequences(ticker, processed_dir)

    # 2. Configurer MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    run_name = f"{model_name}_{ticker.replace('.', '_')}"

    with mlflow.start_run(run_name=run_name):

        # ── Paramètres de contexte ────────────────────────────────────────
        # On logue tout ce qui permet de reproduire l'expérience
        mlflow.log_params({
            "model":    model_name,
            "asset":    ticker,
            "window":   window,
            "git_commit": get_git_commit(),
            "n_train":  data["X_train"].shape[0],
            "n_val":    data["X_val"].shape[0],
            "n_test":   data["X_test"].shape[0],
            "n_features": data["X_train"].shape[2],
        })

        # 3. Instancier le modèle
        ModelClass = MODEL_REGISTRY[model_name]
        model = ModelClass()

        # ── Hyperparamètres du modèle ─────────────────────────────────────
        mlflow.log_params(model.config)

        # 4. Entraîner
        train_results = model.train(
            data["X_train"], data["y_train"],
            data["X_val"],   data["y_val"],
        )

        # ── Métriques d'entraînement ──────────────────────────────────────
        mlflow.log_metrics({k: v for k, v in train_results.items()
                            if isinstance(v, (int, float))})

        # 5. Évaluer sur le TEST SET (la donnée jamais vue)
        test_metrics = model.evaluate(data["X_test"], data["y_test"])
        logger.info(f"Test — {test_metrics}")

        # Préfixer avec "test_" pour distinguer dans l'UI MLflow
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # ── Résumé console ────────────────────────────────────────────────
        da = test_metrics["directional_accuracy"]
        sharpe = test_metrics["sharpe_ratio"]
        logger.info(f"RESULTAT — Dir. Accuracy: {da:.1%} | Sharpe: {sharpe:.2f}")

        # 6. Sauvegarder le modèle comme artefact MLflow
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{run_name}.pkl"
        saved_path = model.save(model_path)   # retourne le chemin réel (.pt ou .pkl)
        mlflow.log_artifact(str(saved_path), artifact_path="model")

        # 7. Sauvegarder les feature importances (XGBoost uniquement)
        if hasattr(model, "feature_importances"):
            fi = model.feature_importances()
            fi_path = MODELS_DIR / f"{run_name}_feature_importances.npy"
            np.save(fi_path, fi)
            mlflow.log_artifact(str(fi_path), artifact_path="model")
            logger.info(f"Feature importances sauvegardées → {fi_path}")

        # 8. Logger les courbes d'apprentissage epoch par epoch (LSTM/TCN/TFT)
        # MLflow permet de visualiser train_loss vs val_loss dans l'UI
        if hasattr(model, "_history"):
            for step, (tl, vl) in enumerate(zip(
                model._history["train_loss"],
                model._history["val_loss"]
            )):
                mlflow.log_metrics(
                    {"epoch_train_loss": tl, "epoch_val_loss": vl},
                    step=step
                )

    logger.info(f"Run MLflow enregistre : {run_name}")
    return test_metrics


def main():
    args = parse_args()

    # Déterminer les actifs à traiter
    tickers = TICKERS if args.asset == "all" else [args.asset]

    # Valider les tickers
    invalid = [t for t in tickers if t not in TICKERS]
    if invalid:
        logger.error(f"Tickers invalides : {invalid}. Valides : {TICKERS}")
        sys.exit(1)

    results = {}
    for ticker in tickers:
        metrics = train_one(
            model_name=args.model,
            ticker=ticker,
            window=args.window,
            experiment_name=args.experiment,
            processed_dir=PROCESSED_DIR,
        )
        results[ticker] = metrics

    # ── Résumé final ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  RESULTATS {args.model.upper()}")
    print("=" * 60)
    print(f"  {'Actif':<12} {'Dir.Acc':>9} {'Sharpe':>8} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*12} {'-'*9} {'-'*8} {'-'*8} {'-'*8}")
    for ticker, m in results.items():
        print(
            f"  {ticker:<12} "
            f"{m['directional_accuracy']:>8.1%} "
            f"{m['sharpe_ratio']:>8.2f} "
            f"{m['mae']:>8.4f} "
            f"{m['rmse']:>8.4f}"
        )
    print("=" * 60)
    print("  Voir les details : http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
