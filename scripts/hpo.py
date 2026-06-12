"""
hpo.py — Optimisation automatique des hyperparamètres avec Optuna.

Usage :
    python scripts/hpo.py --model xgboost --asset MC.PA --trials 50
    python scripts/hpo.py --model lstm    --asset RMS.PA --trials 30
    python scripts/hpo.py --model tcn     --asset all    --trials 40
    python scripts/hpo.py --model tft     --asset MC.PA  --trials 30

Ce script :
  1. Crée une étude Optuna (stockée dans optuna.db — persistante)
  2. Lance N essais : chaque essai entraîne le modèle avec des hyperparamètres
     différents et retourne la val_directional_accuracy
  3. Optuna choisit intelligemment les hyperparamètres suivants
     (algorithme TPE — Tree-structured Parzen Estimator)
  4. À la fin : ré-entraîne le meilleur modèle et évalue sur le TEST SET
  5. Logue tout dans MLflow sous l'experiment "projet140_hpo"

IMPORTANT — Règle anti-leakage :
  L'optimisation se fait UNIQUEMENT sur val_directional_accuracy.
  Le test set n'est touché qu'une seule fois, à la toute fin,
  pour rapporter les métriques finales honnêtes.
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import mlflow
import numpy as np
import optuna
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from models.tcn_model import TCNModel
from models.tft_model import TFTModel

load_dotenv()

# Réduire le bruit dans la console — Optuna est très verbeux par défaut
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models/saved")
TICKERS = ["MC.PA", "CFR.SW", "RMS.PA", "BRBY.L", "OR.PA"]
OPTUNA_DB = "sqlite:///optuna.db"
MLFLOW_URI = "sqlite:///mlflow.db"


# ─────────────────────────────────────────────────────────────────────────────
# Espaces de recherche par modèle
# ─────────────────────────────────────────────────────────────────────────────
# trial.suggest_* définit le type et la plage de chaque hyperparamètre.
# Optuna explore intelligemment ces espaces avec l'algorithme TPE.
#
# TPE (Tree-structured Parzen Estimator) :
#   Au lieu d'explorer aléatoirement, TPE modélise la distribution des
#   hyperparamètres qui donnent de bons résultats et échantillonne
#   préférentiellement dans les régions prometteuses.
#   → Converge en 30-100 essais vs 300+ pour la recherche aléatoire.

def suggest_xgboost(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        # log=True : explore exponentiellement (0.001, 0.003, 0.01, 0.03, 0.1...)
        # plutôt que linéairement — mieux pour les lr qui varient sur des ordres
        # de grandeur
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "early_stopping_rounds": 30,
    }


def suggest_lstm(trial: optuna.Trial) -> dict:
    return {
        "hidden_size": trial.suggest_categorical("hidden_size",
                                                  [32, 64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate",
                                              1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 5.0),
        "epochs": 50,
        "patience": 10,
        "random_state": 42,
    }


def suggest_tcn(trial: optuna.Trial) -> dict:
    # Nombre de niveaux de dilation : génère automatiquement [1, 2, 4, ...]
    n_levels = trial.suggest_int("n_dilation_levels", 3, 6)
    dilations = [2 ** i for i in range(n_levels)]
    return {
        "num_channels": trial.suggest_categorical("num_channels",
                                                   [32, 64, 128]),
        "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 5]),
        "dilations": dilations,
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate",
                                              1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 5.0),
        "epochs": 50,
        "patience": 10,
        "random_state": 42,
    }


def suggest_tft(trial: optuna.Trial) -> dict:
    # d_model doit être divisible par n_heads — on construit ça proprement
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    # d_model est un multiple de n_heads dans [32, 256]
    d_model_factor = trial.suggest_int("d_model_factor", 1, 4)
    d_model = n_heads * (8 * d_model_factor)   # ex: 4 heads × 16 = 64

    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "learning_rate": trial.suggest_float("learning_rate",
                                              1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 5.0),
        "epochs": 50,
        "patience": 10,
        "random_state": 42,
    }


SEARCH_SPACES = {
    "xgboost": suggest_xgboost,
    "lstm": suggest_lstm,
    "tcn": suggest_tcn,
    "tft": suggest_tft,
}

MODEL_REGISTRY = {
    "xgboost": XGBoostModel,
    "lstm": LSTMModel,
    "tcn": TCNModel,
    "tft": TFTModel,
}


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────────────────────────────────────

def load_sequences(ticker: str) -> dict:
    safe = ticker.replace(".", "_")
    d = PROCESSED_DIR / safe
    if not d.exists():
        raise FileNotFoundError(
            f"Données introuvables pour {ticker}. "
            "Lance d'abord : python scripts/preprocess.py"
        )
    return {
        split: (np.load(d / f"X_{split}.npy"), np.load(d / f"y_{split}.npy"))
        for split in ["train", "val", "test"]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fonction objectif Optuna
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(model_name: str, ticker: str, data: dict):
    """
    Retourne la fonction objectif pour un (modèle, actif) donné.

    Cette fonction est appelée par Optuna pour chaque essai.
    Elle reçoit un objet `trial` qui propose des hyperparamètres,
    entraîne le modèle, et retourne la métrique à maximiser.

    IMPORTANT : on retourne val_directional_accuracy, PAS test.
    Le test set est réservé pour l'évaluation finale.
    """
    X_train, y_train = data["train"]
    X_val,   y_val   = data["val"]

    suggest_fn = SEARCH_SPACES[model_name]
    ModelClass = MODEL_REGISTRY[model_name]

    def objective(trial: optuna.Trial) -> float:
        config = suggest_fn(trial)

        # Supprimer les warnings de convergence pour garder la console lisible
        try:
            model = ModelClass(config)
            model.train(X_train, y_train, X_val, y_val)
            metrics = model.evaluate(X_val, y_val)
            val_acc = metrics["directional_accuracy"]

            # Stocker les métriques dans le trial pour pouvoir les récupérer
            trial.set_user_attr("val_sharpe", metrics["sharpe_ratio"])
            trial.set_user_attr("val_mae", metrics["mae"])

        except Exception as e:
            # Si le modèle plante (ex: NaN dans les gradients), on pénalise
            # plutôt que de faire crasher toute l'étude
            logger.warning(f"Trial {trial.number} échoué : {e}")
            return 0.0

        return val_acc

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction de la config depuis les params du meilleur trial
# ─────────────────────────────────────────────────────────────────────────────

def params_to_config(model_name: str, params: dict) -> dict:
    """
    Reconstruit la config complète à partir des params Optuna.

    Nécessaire car certains hyperparamètres sont DÉRIVÉS d'autres
    (ex: dilations TCN = [1, 2, 4, ...] calculé depuis n_dilation_levels)
    et ne sont pas directement dans best.params.
    """
    base = {"epochs": 50, "patience": 10, "random_state": 42}

    if model_name == "xgboost":
        return {**base, **params, "early_stopping_rounds": 30}

    if model_name == "lstm":
        return {**base, **params}

    if model_name == "tcn":
        p = dict(params)
        n_levels = p.pop("n_dilation_levels")
        p["dilations"] = [2 ** i for i in range(n_levels)]
        return {**base, **p}

    if model_name == "tft":
        p = dict(params)
        n_heads = p["n_heads"]
        factor = p.pop("d_model_factor")
        p["d_model"] = n_heads * (8 * factor)
        return {**base, **p}

    raise ValueError(f"Modèle inconnu : {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# HPO pour un (modèle, actif)
# ─────────────────────────────────────────────────────────────────────────────

def run_hpo(model_name: str, ticker: str, n_trials: int) -> dict:
    """
    Lance l'HPO pour un modèle sur un actif.

    Étapes :
    1. Créer ou charger l'étude Optuna (reprise possible si interrompue)
    2. Optimiser pendant n_trials
    3. Afficher les meilleurs hyperparamètres
    4. Ré-entraîner le meilleur modèle sur train+val
    5. Évaluer sur test
    6. Logguer dans MLflow

    Retourne les métriques test du meilleur modèle.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  HPO | Modèle: {model_name.upper()} | Actif: {ticker}")
    logger.info(f"  {n_trials} essais — stockés dans optuna.db")
    logger.info(f"{'='*60}")

    data = load_sequences(ticker)
    X_train, y_train = data["train"]
    X_val,   y_val   = data["val"]
    X_test,  y_test  = data["test"]

    # ── 1. Créer/charger l'étude ───────────────────────────────────────────
    # load_if_exists=True : si l'étude existe déjà (run précédent interrompu),
    # Optuna recharge l'historique et continue depuis où on en était.
    # C'est fondamental sur ROMEO où les jobs peuvent être préemptés.
    study_name = f"{model_name}_{ticker.replace('.', '_')}"
    study = optuna.create_study(
        study_name=study_name,
        storage=OPTUNA_DB,
        direction="maximize",       # on maximise val_directional_accuracy
        load_if_exists=True,        # reprise si interrompu
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        # MedianPruner : arrête tôt les essais clairement mauvais
        # (ceux dont la val_loss est dans la moitié inférieure des essais précédents)
        # Économise du temps en ne finissant pas les runs perdants.
    )

    n_already = len(study.trials)
    if n_already > 0:
        logger.info(f"Étude existante avec {n_already} essais — reprise.")

    n_remaining = max(0, n_trials - n_already)
    if n_remaining == 0:
        logger.info("Tous les essais sont déjà terminés.")
    else:
        logger.info(f"Lancement de {n_remaining} essais...")

        objective = make_objective(model_name, ticker, data)
        study.optimize(
            objective,
            n_trials=n_remaining,
            show_progress_bar=True,
            gc_after_trial=True,    # libère la mémoire entre les essais
        )

    # ── 2. Afficher les résultats ──────────────────────────────────────────
    best = study.best_trial
    logger.info(f"\nMeilleur essai : #{best.number}")
    logger.info(f"  val_directional_accuracy : {best.value:.4f}")
    logger.info(f"  Hyperparamètres :")
    for k, v in best.params.items():
        logger.info(f"    {k:30s} = {v}")

    # ── 3. Ré-entraîner avec les meilleurs hyperparamètres ─────────────────
    logger.info("\nRé-entraînement avec les meilleurs hyperparamètres...")
    best_config = params_to_config(model_name, best.params)

    ModelClass = MODEL_REGISTRY[model_name]
    final_model = ModelClass(best_config)
    final_model.train(X_train, y_train, X_val, y_val)

    # ── 4. Évaluation finale sur le TEST SET ───────────────────────────────
    test_metrics = final_model.evaluate(X_test, y_test)
    logger.info(f"\nRésultats TEST (jamais vus pendant l'HPO) :")
    logger.info(f"  directional_accuracy : {test_metrics['directional_accuracy']:.4f}")
    logger.info(f"  sharpe_ratio         : {test_metrics['sharpe_ratio']:.4f}")
    logger.info(f"  mae                  : {test_metrics['mae']:.4f}")

    # ── 5. Logguer dans MLflow ─────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("projet140_hpo")
    run_name = f"hpo_{model_name}_{ticker.replace('.', '_')}"

    with mlflow.start_run(run_name=run_name):
        # Contexte
        mlflow.log_params({
            "model": model_name,
            "asset": ticker,
            "n_trials": len(study.trials),
            "best_trial": best.number,
        })

        # Meilleurs hyperparamètres
        mlflow.log_params({f"hp_{k}": v for k, v in best.params.items()})

        # Métriques val du meilleur trial
        mlflow.log_metric("best_val_directional_accuracy", best.value)

        # Métriques test finales
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Sauvegarder le modèle optimisé
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"best_{run_name}.pkl"
        saved_path = final_model.save(model_path)
        mlflow.log_artifact(str(saved_path), artifact_path="model")

        # Pour TFT : logguer les importances de features
        if hasattr(final_model, "get_feature_importance"):
            fi = final_model.get_feature_importance(X_test)
            fi_path = MODELS_DIR / f"best_{run_name}_feature_importance.npy"
            np.save(fi_path, fi)
            mlflow.log_artifact(str(fi_path), artifact_path="model")

    logger.info(f"Run MLflow enregistré : {run_name} (experiment: projet140_hpo)")
    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="HPO avec Optuna — recherche automatique d'hyperparamètres"
    )
    parser.add_argument(
        "--model", required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Architecture à optimiser",
    )
    parser.add_argument(
        "--asset", required=True,
        help='Ticker (ex: MC.PA) ou "all" pour tous les actifs',
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Nombre d'essais Optuna (défaut: 50)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tickers = TICKERS if args.asset == "all" else [args.asset]

    invalid = [t for t in tickers if t not in TICKERS]
    if invalid:
        logger.error(f"Tickers invalides : {invalid}. Valides : {TICKERS}")
        sys.exit(1)

    all_results = {}
    for ticker in tickers:
        metrics = run_hpo(args.model, ticker, args.trials)
        all_results[ticker] = metrics

    # ── Tableau récapitulatif ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  RESULTATS HPO — {args.model.upper()} ({args.trials} essais)")
    print("=" * 65)
    print(f"  {'Actif':<12} {'Dir.Acc':>9} {'Sharpe':>8} {'MAE':>8}")
    print(f"  {'-'*12} {'-'*9} {'-'*8} {'-'*8}")
    for ticker, m in all_results.items():
        print(
            f"  {ticker:<12} "
            f"{m['directional_accuracy']:>8.1%} "
            f"{m['sharpe_ratio']:>8.2f} "
            f"{m['mae']:>8.4f}"
        )
    print("=" * 65)
    print("  Voir les details : http://localhost:5000")
    print("  Experiment MLflow : projet140_hpo")
    print("=" * 65)


if __name__ == "__main__":
    main()
