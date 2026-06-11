"""
Tests unitaires pour XGBoostModel.

On teste le COMPORTEMENT du modèle, pas ses performances.
Un test unitaire doit être :
  - Rapide   : données synthétiques petites, pas les vraies données
  - Isolé    : ne dépend d'aucun fichier externe
  - Déterministe : même résultat à chaque exécution (seed fixé)

Lancer : python -m pytest tests/test_xgboost.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgboost_model import XGBoostModel


# ── Fixtures pytest ───────────────────────────────────────────────────────────
# Une fixture est une fonction qui prépare les données pour les tests.
# pytest l'injecte automatiquement dans les tests qui en ont besoin.

@pytest.fixture
def synthetic_data():
    """
    Génère des données synthétiques de la même forme que les vraies données.
    Petit dataset pour que les tests soient rapides (<1 seconde).
    """
    rng = np.random.default_rng(42)  # seed fixé = résultats reproductibles
    N_train, N_val, N_test = 200, 50, 50
    window, n_features = 30, 18

    X_train = rng.random((N_train, window, n_features)).astype(np.float32)
    y_train = rng.integers(0, 2, N_train).astype(np.int64)

    X_val = rng.random((N_val, window, n_features)).astype(np.float32)
    y_val = rng.integers(0, 2, N_val).astype(np.int64)

    X_test = rng.random((N_test, window, n_features)).astype(np.float32)
    y_test = rng.integers(0, 2, N_test).astype(np.int64)

    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture
def trained_model(synthetic_data):
    """Fournit un modèle déjà entraîné pour les tests d'inférence."""
    X_train, y_train, X_val, y_val, _, _ = synthetic_data
    model = XGBoostModel({"n_estimators": 20, "early_stopping_rounds": 5})
    model.train(X_train, y_train, X_val, y_val)
    return model


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestXGBoostInterface:
    """Vérifie que XGBoostModel respecte le contrat de BaseModel."""

    def test_instantiation(self):
        """Le modèle se crée sans erreur avec la config par défaut."""
        model = XGBoostModel()
        assert model.config is not None
        assert not model.is_trained

    def test_custom_config(self):
        """Les hyperparamètres personnalisés écrasent les valeurs par défaut."""
        model = XGBoostModel({"n_estimators": 50, "max_depth": 3})
        assert model.config["n_estimators"] == 50
        assert model.config["max_depth"] == 3
        # Les autres clés gardent leurs valeurs par défaut
        assert "learning_rate" in model.config

    def test_predict_before_train_raises(self):
        """predict() doit lever une erreur si le modèle n'est pas entraîné."""
        model = XGBoostModel()
        X = np.random.random((10, 30, 18)).astype(np.float32)
        with pytest.raises(RuntimeError, match="pas encore entraîné"):
            model.predict(X)

    def test_train_returns_dict(self, synthetic_data):
        """train() doit retourner un dict avec les métriques."""
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        model = XGBoostModel({"n_estimators": 10, "early_stopping_rounds": 3})
        result = model.train(X_train, y_train, X_val, y_val)
        assert isinstance(result, dict)
        assert "best_round" in result
        assert "train_directional_accuracy" in result
        assert "val_directional_accuracy" in result


class TestXGBoostPredictions:
    """Vérifie la forme et la validité des prédictions."""

    def test_predict_output_shape(self, trained_model, synthetic_data):
        """predict() doit retourner un vecteur de longueur N."""
        _, _, _, _, X_test, _ = synthetic_data
        preds = trained_model.predict(X_test)
        assert preds.shape == (len(X_test),), \
            f"Forme attendue ({len(X_test)},), obtenu {preds.shape}"

    def test_predict_binary_values(self, trained_model, synthetic_data):
        """Les prédictions doivent être strictement 0 ou 1."""
        _, _, _, _, X_test, _ = synthetic_data
        preds = trained_model.predict(X_test)
        unique_values = set(np.unique(preds))
        assert unique_values.issubset({0, 1}), \
            f"Valeurs inattendues dans les prédictions : {unique_values}"

    def test_predict_no_nan(self, trained_model, synthetic_data):
        """Aucun NaN dans les prédictions."""
        _, _, _, _, X_test, _ = synthetic_data
        preds = trained_model.predict(X_test)
        assert not np.any(np.isnan(preds)), "NaN détectés dans predict()"

    def test_predict_proba_range(self, trained_model, synthetic_data):
        """Les probabilités doivent être dans [0, 1]."""
        _, _, _, _, X_test, _ = synthetic_data
        probas = trained_model.predict_proba(X_test)
        assert probas.shape == (len(X_test),)
        assert np.all(probas >= 0) and np.all(probas <= 1), \
            "Probabilités hors de [0, 1]"

    def test_predict_proba_no_nan(self, trained_model, synthetic_data):
        """Aucun NaN dans les probabilités."""
        _, _, _, _, X_test, _ = synthetic_data
        probas = trained_model.predict_proba(X_test)
        assert not np.any(np.isnan(probas)), "NaN détectés dans predict_proba()"


class TestXGBoostEvaluation:
    """Vérifie la méthode evaluate() héritée de BaseModel."""

    def test_evaluate_returns_all_metrics(self, trained_model, synthetic_data):
        """evaluate() doit retourner les 4 métriques attendues."""
        _, _, _, _, X_test, y_test = synthetic_data
        metrics = trained_model.evaluate(X_test, y_test)
        for key in ["directional_accuracy", "mae", "rmse", "sharpe_ratio"]:
            assert key in metrics, f"Métrique manquante : {key}"

    def test_directional_accuracy_range(self, trained_model, synthetic_data):
        """La directional accuracy doit être dans [0, 1]."""
        _, _, _, _, X_test, y_test = synthetic_data
        metrics = trained_model.evaluate(X_test, y_test)
        da = metrics["directional_accuracy"]
        assert 0.0 <= da <= 1.0, f"Directional accuracy hors de [0,1] : {da}"

    def test_mae_positive(self, trained_model, synthetic_data):
        """MAE doit être positive."""
        _, _, _, _, X_test, y_test = synthetic_data
        metrics = trained_model.evaluate(X_test, y_test)
        assert metrics["mae"] >= 0, "MAE négative !"

    def test_rmse_geq_mae(self, trained_model, synthetic_data):
        """RMSE >= MAE (propriété mathématique fondamentale)."""
        _, _, _, _, X_test, y_test = synthetic_data
        metrics = trained_model.evaluate(X_test, y_test)
        assert metrics["rmse"] >= metrics["mae"] - 1e-6, \
            f"RMSE ({metrics['rmse']}) < MAE ({metrics['mae']})"


class TestXGBoostSaveLoad:
    """Vérifie la sérialisation et désérialisation du modèle."""

    def test_save_and_load(self, trained_model, synthetic_data, tmp_path):
        """
        Le modèle sauvegardé puis rechargé doit produire
        exactement les mêmes prédictions.
        """
        _, _, _, _, X_test, _ = synthetic_data
        preds_before = trained_model.predict(X_test)

        # Sauvegarder
        model_path = tmp_path / "xgboost_test.pkl"
        trained_model.save(model_path)
        assert model_path.exists()

        # Recharger
        loaded = XGBoostModel.load(model_path, trained_model.config)
        preds_after = loaded.predict(X_test)

        np.testing.assert_array_equal(preds_before, preds_after,
                                      err_msg="Prédictions différentes après save/load")
