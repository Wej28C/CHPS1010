"""
base_model.py — Interface commune pour tous les modèles du projet.

Chaque modèle (XGBoost, LSTM, TCN, TFT) hérite de BaseModel et doit
implémenter les méthodes abstraites : train(), predict().

La méthode evaluate() est implémentée ici une seule fois et réutilisée
par tous les modèles — pas de duplication de code.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    """
    Classe de base abstraite pour tous les modèles de prédiction.

    ABC = Abstract Base Class : Python empêche d'instancier cette classe
    directement. On ne peut instancier que les sous-classes qui implémentent
    toutes les méthodes abstraites (@abstractmethod).

    Paramètres
    ----------
    config : dict
        Hyperparamètres du modèle. Chaque modèle définit ses propres clés.
        Exemple XGBoost : {"n_estimators": 200, "max_depth": 6, "lr": 0.05}
        Exemple LSTM    : {"hidden_size": 128, "num_layers": 2, "lr": 1e-3}
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None        # sera assigné dans train()
        self.is_trained = False

    # ─────────────────────────────────────────────────────────────────────────
    # Méthodes abstraites — OBLIGATOIRES dans chaque sous-classe
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """
        Entraîne le modèle.

        Paramètres
        ----------
        X_train : (N_train, window, n_features)  — séquences d'entraînement
        y_train : (N_train,)                     — cibles (0 ou 1)
        X_val   : (N_val, window, n_features)    — séquences de validation
        y_val   : (N_val,)                       — cibles validation

        Retourne
        --------
        dict : métriques d'entraînement {"train_acc": ..., "val_acc": ...}
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les prédictions pour X.

        Paramètres
        ----------
        X : (N, window, n_features)

        Retourne
        --------
        np.ndarray de shape (N,) avec des valeurs 0 ou 1
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne la probabilité de la classe 1 (hausse).

        Paramètres
        ----------
        X : (N, window, n_features)

        Retourne
        --------
        np.ndarray de shape (N,) avec des valeurs dans [0, 1]
        """

    # ─────────────────────────────────────────────────────────────────────────
    # Méthodes communes — implémentées ici, héritées par tous
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Calcule toutes les métriques d'évaluation sur un split donné.

        Métriques calculées :
        ─────────────────────
        - directional_accuracy : % de bonnes prédictions de direction
              C'est la métrique principale pour un modèle de trading.
              Un modèle aléatoire obtient ~50%. Seuil "utile" : >55%.

        - mae : Mean Absolute Error sur les probabilités prédites
              Mesure l'erreur moyenne entre P(hausse) prédite et la vraie direction.

        - rmse : Root Mean Squared Error — pénalise plus les grandes erreurs
              Utile pour comparer la calibration des modèles probabilistes.

        - sharpe_ratio : rendement moyen / écart-type du rendement × √252
              Mesure la viabilité financière de la stratégie.
              Un Sharpe > 1 est considéré bon, > 2 excellent.
              Ici : stratégie long si prédiction=1, short si prédiction=0.

        Paramètres
        ----------
        X : (N, window, n_features)
        y : (N,)  — vraies cibles (0 ou 1)

        Retourne
        --------
        dict avec toutes les métriques
        """
        preds = self.predict(X)           # prédictions binaires (0 ou 1)
        probas = self.predict_proba(X)    # probabilités P(hausse)

        # Directional Accuracy
        directional_accuracy = float(np.mean(preds == y))

        # MAE et RMSE sur les probabilités vs cibles réelles
        mae = float(np.mean(np.abs(probas - y)))
        rmse = float(np.sqrt(np.mean((probas - y) ** 2)))

        # Sharpe Ratio (backtest simplifié)
        # Signal : +1 si on prédit hausse, -1 si on prédit baisse
        # En pratique : on achète si preds=1, on vend si preds=0
        # Le rendement réel de la journée est approximé par (2*y - 1)
        # car y=1 → hausse (+1), y=0 → baisse (-1)
        signals = 2 * preds.astype(float) - 1      # +1 ou -1
        returns = signals * (2 * y.astype(float) - 1)  # +1 si correct, -1 sinon

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret > 0 else 0.0

        return {
            "directional_accuracy": round(directional_accuracy, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "sharpe_ratio": round(sharpe, 4),
        }

    def save(self, path: Path):
        """Sérialise le modèle. Implémentation par défaut via joblib."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path, config: dict):
        """Charge un modèle sérialisé."""
        import joblib
        instance = cls(config)
        instance.model = joblib.load(path)
        instance.is_trained = True
        return instance

    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"
