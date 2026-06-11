"""
xgboost_model.py — Modèle baseline XGBoost pour la prédiction directionnelle.

XGBoost (eXtreme Gradient Boosting) est un ensemble d'arbres de décision
entraînés en séquence, chaque arbre corrigeant les erreurs du précédent.

Rôle dans le projet : BASELINE
────────────────────────────────
C'est le modèle de référence. Sa performance définit le seuil minimal
que LSTM, TCN et TFT doivent dépasser pour justifier leur complexité.

Spécificité XGBoost vs LSTM/TCN :
───────────────────────────────────
XGBoost est un modèle tabulaire : il ne comprend pas l'ordre temporel.
On lui fournit les séquences aplaties (flatten) :
  Entrée LSTM/TCN : (N, window=30, features=18)  → tenseur 3D
  Entrée XGBoost  : (N, 30×18=540)               → tableau 2D

Il dispose quand même de l'historique des 30 derniers jours,
mais sous forme de colonnes indépendantes.
"""

import logging

import numpy as np
from xgboost import XGBClassifier

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


# Hyperparamètres par défaut — seront optimisés par Optuna à l'étape 8
DEFAULT_CONFIG = {
    # Nombre d'arbres dans l'ensemble
    # Plus d'arbres = meilleur apprentissage mais risque d'overfitting
    "n_estimators": 300,

    # Profondeur maximale de chaque arbre
    # Une profondeur de 6 est un bon compromis biais/variance en finance
    "max_depth": 6,

    # Taux d'apprentissage (learning rate)
    # Petit LR + beaucoup d'arbres > grand LR + peu d'arbres (règle générale)
    "learning_rate": 0.05,

    # Fraction de features à considérer à chaque split
    # Réduction aléatoire → régularisation, meilleure généralisation
    "colsample_bytree": 0.8,

    # Fraction des exemples utilisés pour entraîner chaque arbre
    # Subsampling → réduit le surapprentissage
    "subsample": 0.8,

    # Régularisation L1 (lasso) — force certains poids à zéro
    "reg_alpha": 0.1,

    # Régularisation L2 (ridge) — pénalise les grands poids
    "reg_lambda": 1.0,

    # Nombre de rounds sans amélioration avant d'arrêter (early stopping)
    "early_stopping_rounds": 30,

    # Seed pour reproductibilité
    "random_state": 42,

    # Utiliser tous les CPU disponibles pour paralléliser la construction des arbres
    "n_jobs": -1,
}


class XGBoostModel(BaseModel):
    """
    Modèle XGBoost pour classification binaire (direction J+1).

    Hérite de BaseModel et implémente :
    - train()        : entraînement avec early stopping sur val set
    - predict()      : prédictions binaires (0 ou 1)
    - predict_proba(): probabilités P(hausse)
    """

    def __init__(self, config: dict = None):
        # Si aucun config fourni, on utilise les valeurs par défaut
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """
        Aplatit les séquences 3D en tableau 2D pour XGBoost.

        Transformation :
          (N, window, n_features) → (N, window × n_features)
          Ex: (1730, 30, 18)      → (1730, 540)

        L'ordre de l'aplatissement est préservé (C order = row-major) :
        les features du jour le plus ancien sont en tête, les plus récentes
        à la fin — XGBoost peut ainsi exploiter les patterns temporels
        via des features "lag".
        """
        return X.reshape(X.shape[0], -1)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """
        Entraîne XGBoost avec early stopping sur le validation set.

        Early stopping : si la métrique de validation ne s'améliore pas
        pendant `early_stopping_rounds` rounds consécutifs, l'entraînement
        s'arrête. Cela évite l'overfitting sans avoir à fixer n_estimators
        manuellement.

        L'eval_metric "logloss" (log-loss binaire) est standard pour la
        classification binaire. Elle mesure la qualité des probabilités
        prédites, pas seulement les classes.
        """
        logger.info(
            f"XGBoost — entraînement sur {X_train.shape[0]} exemples, "
            f"validation sur {X_val.shape[0]} exemples"
        )

        # Aplatir les séquences 3D → 2D
        X_tr = self._flatten(X_train)
        X_vl = self._flatten(X_val)

        logger.info(f"Shape après flatten — train: {X_tr.shape}, val: {X_vl.shape}")

        # Vérifier le déséquilibre de classes
        # En finance, les marchés haussiers sont souvent plus fréquents (~53%)
        # scale_pos_weight compense ce déséquilibre pour éviter que le modèle
        # prédise toujours "hausse" (biais trivial)
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        scale = n_neg / n_pos if n_pos > 0 else 1.0
        logger.info(f"Classes — hausse: {n_pos}, baisse: {n_neg}, scale_pos_weight: {scale:.2f}")

        # Instancier le classifieur
        self.model = XGBClassifier(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            learning_rate=self.config["learning_rate"],
            colsample_bytree=self.config["colsample_bytree"],
            subsample=self.config["subsample"],
            reg_alpha=self.config["reg_alpha"],
            reg_lambda=self.config["reg_lambda"],
            scale_pos_weight=scale,
            random_state=self.config["random_state"],
            n_jobs=self.config["n_jobs"],
            eval_metric="logloss",
            early_stopping_rounds=self.config["early_stopping_rounds"],
            verbosity=0,  # silencieux (les logs passent par notre logger)
        )

        # Entraînement avec eval_set pour le early stopping
        self.model.fit(
            X_tr, y_train,
            eval_set=[(X_vl, y_val)],
            verbose=False,
        )

        self.is_trained = True

        best_round = self.model.best_iteration
        logger.info(f"XGBoost entraîné — meilleur round: {best_round}")

        # Métriques sur train et val
        train_metrics = self.evaluate(X_train, y_train)
        val_metrics = self.evaluate(X_val, y_val)

        logger.info(f"Train — {train_metrics}")
        logger.info(f"Val   — {val_metrics}")

        return {
            "best_round": best_round,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédictions binaires : 0 (baisse) ou 1 (hausse)."""
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        return self.model.predict(self._flatten(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilité P(hausse) pour chaque exemple."""
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        # predict_proba retourne (N, 2) : [:, 0] = P(baisse), [:, 1] = P(hausse)
        return self.model.predict_proba(self._flatten(X))[:, 1]

    def feature_importances(self) -> np.ndarray:
        """
        Retourne l'importance de chaque feature selon XGBoost.

        Utile pour le rapport : quels indicateurs techniques sont
        les plus prédictifs pour le secteur luxe ?
        """
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        return self.model.feature_importances_
