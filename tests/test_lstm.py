"""
Tests unitaires pour LSTMModel.

Même stratégie que pour XGBoost : données synthétiques petites,
tests rapides, pas de dépendances externes.

Lancer : python -m pytest tests/test_lstm.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.lstm_model import LSTMModel, LSTMNet


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    N_train, N_val, N_test = 150, 40, 40
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
    """LSTM entraîné avec config minimale pour que les tests soient rapides."""
    X_train, y_train, X_val, y_val, _, _ = synthetic_data
    model = LSTMModel({
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.0,
        "epochs": 3,
        "patience": 2,
        "batch_size": 32,
    })
    model.train(X_train, y_train, X_val, y_val)
    return model


class TestLSTMNet:
    """Tests sur l'architecture PyTorch directement."""

    def test_forward_shape(self):
        """La sortie du réseau a la bonne forme : (batch,)."""
        net = LSTMNet(input_size=18, hidden_size=32, num_layers=1, dropout=0.0)
        x = torch.randn(10, 30, 18)   # (batch=10, window=30, features=18)
        out = net(x)
        assert out.shape == (10,), f"Forme attendue (10,), obtenu {out.shape}"

    def test_output_range(self):
        """La sortie est une probabilité dans [0, 1] (grâce à Sigmoid)."""
        net = LSTMNet(input_size=18, hidden_size=32, num_layers=1, dropout=0.0)
        x = torch.randn(20, 30, 18)
        out = net(x)
        assert torch.all(out >= 0) and torch.all(out <= 1), \
            "Sigmoid doit garantir des sorties dans [0, 1]"

    def test_no_nan_in_output(self):
        """Aucun NaN dans les sorties du réseau."""
        net = LSTMNet(input_size=18, hidden_size=32, num_layers=1, dropout=0.0)
        x = torch.randn(10, 30, 18)
        out = net(x)
        assert not torch.any(torch.isnan(out)), "NaN détectés dans la sortie"

    def test_multilayer(self):
        """Le réseau fonctionne avec plusieurs couches LSTM."""
        net = LSTMNet(input_size=18, hidden_size=64, num_layers=3, dropout=0.2)
        x = torch.randn(8, 30, 18)
        out = net(x)
        assert out.shape == (8,)


class TestLSTMModel:
    """Tests sur le wrapper LSTMModel (interface BaseModel)."""

    def test_instantiation(self):
        model = LSTMModel()
        assert not model.is_trained
        assert model.config["hidden_size"] == 128  # valeur par défaut

    def test_predict_before_train_raises(self):
        model = LSTMModel()
        X = np.random.random((10, 30, 18)).astype(np.float32)
        with pytest.raises(RuntimeError, match="pas encore entraîné"):
            model.predict(X)

    def test_train_sets_is_trained(self, synthetic_data):
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        model = LSTMModel({"hidden_size": 16, "num_layers": 1,
                           "epochs": 2, "patience": 1, "batch_size": 32})
        model.train(X_train, y_train, X_val, y_val)
        assert model.is_trained

    def test_train_returns_dict_with_metrics(self, synthetic_data):
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        model = LSTMModel({"hidden_size": 16, "num_layers": 1,
                           "epochs": 2, "patience": 1, "batch_size": 32})
        result = model.train(X_train, y_train, X_val, y_val)
        assert "best_val_loss" in result
        assert "train_directional_accuracy" in result
        assert "val_directional_accuracy" in result

    def test_history_recorded(self, trained_model):
        """L'historique des losses est enregistré pour les courbes MLflow."""
        assert hasattr(trained_model, "_history")
        assert "train_loss" in trained_model._history
        assert "val_loss" in trained_model._history
        assert len(trained_model._history["train_loss"]) > 0


class TestLSTMPredictions:
    """Tests sur les prédictions du modèle entraîné."""

    def test_predict_shape(self, trained_model, synthetic_data):
        _, _, _, _, X_test, _ = synthetic_data
        preds = trained_model.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_predict_binary(self, trained_model, synthetic_data):
        _, _, _, _, X_test, _ = synthetic_data
        preds = trained_model.predict(X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_range(self, trained_model, synthetic_data):
        _, _, _, _, X_test, _ = synthetic_data
        probas = trained_model.predict_proba(X_test)
        assert np.all(probas >= 0) and np.all(probas <= 1)

    def test_no_nan(self, trained_model, synthetic_data):
        _, _, _, _, X_test, _ = synthetic_data
        assert not np.any(np.isnan(trained_model.predict(X_test)))
        assert not np.any(np.isnan(trained_model.predict_proba(X_test)))

    def test_evaluate_all_metrics(self, trained_model, synthetic_data):
        _, _, _, _, X_test, y_test = synthetic_data
        metrics = trained_model.evaluate(X_test, y_test)
        for key in ["directional_accuracy", "mae", "rmse", "sharpe_ratio"]:
            assert key in metrics


class TestLSTMSaveLoad:
    """Vérification save/load avec format .pt (PyTorch)."""

    def test_save_creates_file(self, trained_model, tmp_path):
        model_path = tmp_path / "lstm_test.pt"
        trained_model.save(model_path)
        assert model_path.exists()

    def test_load_same_predictions(self, trained_model, synthetic_data, tmp_path):
        """Après save/load, les prédictions doivent être identiques."""
        _, _, _, _, X_test, _ = synthetic_data
        preds_before = trained_model.predict(X_test)

        model_path = tmp_path / "lstm_test.pt"
        trained_model.save(model_path)

        loaded = LSTMModel.load(model_path, trained_model.config)
        preds_after = loaded.predict(X_test)

        np.testing.assert_array_equal(preds_before, preds_after,
                                      err_msg="Prédictions différentes après save/load")
