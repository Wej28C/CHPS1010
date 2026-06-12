"""
Tests unitaires pour TFTModel.

Points spécifiques au TFT vs les autres modèles :
  - forward() retourne un TUPLE (probs, vsn_weights)
  - get_feature_importance() expose les poids VSN
  - Les poids VSN somment à 1 (softmax)
  - d_model doit être divisible par n_heads (contrainte Multi-Head Attention)

Lancer : python -m pytest tests/test_tft.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.tft_model import TFTModel, TFTNet, GatedLinearUnit, GatedResidualNetwork


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

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
    """TFT entraîné avec config minimale — rapide."""
    X_train, y_train, X_val, y_val, _, _ = synthetic_data
    model = TFTModel({
        "d_model": 16,
        "n_heads": 4,       # 16 % 4 == 0 ✓
        "lstm_layers": 1,
        "dropout": 0.0,
        "epochs": 3,
        "patience": 2,
        "batch_size": 32,
    })
    model.train(X_train, y_train, X_val, y_val)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Tests sur les blocs de base
# ─────────────────────────────────────────────────────────────────────────────

class TestGatedLinearUnit:

    def test_output_shape(self):
        """GLU produit un vecteur de la dimension de sortie spécifiée."""
        glu = GatedLinearUnit(input_dim=32, output_dim=16)
        x = torch.randn(4, 10, 32)     # (batch, seq, input_dim)
        out = glu(x)
        assert out.shape == (4, 10, 16)

    def test_output_range(self):
        """
        La sortie du GLU n'est pas bornée (contrairement à Sigmoid).
        Elle peut être négative ou > 1 — la porte module l'amplitude
        mais ne restreint pas la plage.
        """
        glu = GatedLinearUnit(input_dim=8, output_dim=8)
        x = torch.randn(100, 8)
        out = glu(x)
        # Vérifier absence de NaN plutôt qu'une plage fixe
        assert not torch.any(torch.isnan(out))


class TestGatedResidualNetwork:

    def test_output_shape_same_dim(self):
        grn = GatedResidualNetwork(input_dim=32, hidden_dim=32,
                                   output_dim=32, dropout=0.0)
        x = torch.randn(4, 10, 32)
        out = grn(x)
        assert out.shape == (4, 10, 32)

    def test_output_shape_different_dim(self):
        """GRN adapte les dimensions via la connexion résiduelle."""
        grn = GatedResidualNetwork(input_dim=18, hidden_dim=64,
                                   output_dim=32, dropout=0.0)
        x = torch.randn(4, 10, 18)
        out = grn(x)
        assert out.shape == (4, 10, 32)

    def test_no_nan(self):
        grn = GatedResidualNetwork(input_dim=16, hidden_dim=32,
                                   output_dim=16, dropout=0.0)
        x = torch.randn(8, 30, 16)
        assert not torch.any(torch.isnan(grn(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Tests sur TFTNet (réseau complet)
# ─────────────────────────────────────────────────────────────────────────────

class TestTFTNet:

    def test_forward_returns_tuple(self):
        """
        forward() retourne (probs, vsn_weights) — différent des autres modèles
        qui retournent directement un tenseur.
        """
        net = TFTNet(n_features=18, d_model=16, n_heads=4,
                     lstm_layers=1, dropout=0.0)
        x = torch.randn(4, 30, 18)
        output = net(x)
        assert isinstance(output, tuple) and len(output) == 2

    def test_probs_shape(self):
        net = TFTNet(n_features=18, d_model=16, n_heads=4,
                     lstm_layers=1, dropout=0.0)
        x = torch.randn(8, 30, 18)
        probs, _ = net(x)
        assert probs.shape == (8,)

    def test_probs_range(self):
        """Sigmoid garantit des probabilités dans [0, 1]."""
        net = TFTNet(n_features=18, d_model=16, n_heads=4,
                     lstm_layers=1, dropout=0.0)
        x = torch.randn(20, 30, 18)
        probs, _ = net(x)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_vsn_weights_shape(self):
        """Les poids VSN ont la forme (batch, window, n_features)."""
        net = TFTNet(n_features=18, d_model=16, n_heads=4,
                     lstm_layers=1, dropout=0.0)
        x = torch.randn(4, 30, 18)
        _, vsn_weights = net(x)
        assert vsn_weights.shape == (4, 30, 18)

    def test_vsn_weights_sum_to_one(self):
        """
        Les poids VSN sont produits par softmax → ils somment à 1
        sur la dimension des features pour chaque (batch, time).
        C'est la propriété fondamentale qui rend les poids interprétables.
        """
        net = TFTNet(n_features=18, d_model=16, n_heads=4,
                     lstm_layers=1, dropout=0.0)
        net.eval()
        x = torch.randn(4, 30, 18)
        with torch.no_grad():
            _, vsn_weights = net(x)
        sums = vsn_weights.sum(dim=-1)      # (batch, window) — doit être ≈ 1
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Les poids VSN doivent sommer à 1 (softmax)"

    def test_d_model_not_divisible_raises(self):
        """d_model non divisible par n_heads doit lever une AssertionError."""
        with pytest.raises(AssertionError):
            TFTNet(n_features=18, d_model=17, n_heads=4,
                   lstm_layers=1, dropout=0.0)

    def test_no_nan(self):
        net = TFTNet(n_features=18, d_model=16, n_heads=4,
                     lstm_layers=1, dropout=0.0)
        x = torch.randn(8, 30, 18)
        probs, weights = net(x)
        assert not torch.any(torch.isnan(probs))
        assert not torch.any(torch.isnan(weights))


# ─────────────────────────────────────────────────────────────────────────────
# Tests sur TFTModel (wrapper BaseModel)
# ─────────────────────────────────────────────────────────────────────────────

class TestTFTModel:

    def test_instantiation(self):
        model = TFTModel()
        assert not model.is_trained
        assert model.config["d_model"] == 64
        assert model.config["n_heads"] == 4

    def test_predict_before_train_raises(self):
        model = TFTModel()
        X = np.random.random((10, 30, 18)).astype(np.float32)
        with pytest.raises(RuntimeError, match="pas encore entraîné"):
            model.predict(X)

    def test_train_sets_is_trained(self, synthetic_data):
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        model = TFTModel({
            "d_model": 16, "n_heads": 4, "lstm_layers": 1,
            "epochs": 2, "patience": 1, "batch_size": 32,
        })
        model.train(X_train, y_train, X_val, y_val)
        assert model.is_trained

    def test_train_returns_dict_with_metrics(self, synthetic_data):
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        model = TFTModel({
            "d_model": 16, "n_heads": 4, "lstm_layers": 1,
            "epochs": 2, "patience": 1, "batch_size": 32,
        })
        result = model.train(X_train, y_train, X_val, y_val)
        assert "best_val_loss" in result
        assert "train_directional_accuracy" in result
        assert "val_directional_accuracy" in result

    def test_history_recorded(self, trained_model):
        assert hasattr(trained_model, "_history")
        assert len(trained_model._history["train_loss"]) > 0

    def test_get_feature_importance_shape(self, trained_model, synthetic_data):
        """get_feature_importance retourne un vecteur (n_features,)."""
        _, _, _, _, X_test, _ = synthetic_data
        importance = trained_model.get_feature_importance(X_test)
        assert importance.shape == (18,)

    def test_get_feature_importance_sums_to_one(self, trained_model, synthetic_data):
        """Les importances somment à 1 (héritent du softmax VSN)."""
        _, _, _, _, X_test, _ = synthetic_data
        importance = trained_model.get_feature_importance(X_test)
        assert abs(importance.sum() - 1.0) < 1e-4, \
            f"Importances doivent sommer à 1, obtenu {importance.sum()}"


class TestTFTPredictions:

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


class TestTFTSaveLoad:

    def test_save_creates_pt_file(self, trained_model, tmp_path):
        saved = trained_model.save(tmp_path / "tft_test.pt")
        assert saved.exists() and saved.suffix == ".pt"

    def test_load_same_predictions(self, trained_model, synthetic_data, tmp_path):
        """Après save/load, les prédictions sont identiques."""
        _, _, _, _, X_test, _ = synthetic_data
        preds_before = trained_model.predict(X_test)

        path = tmp_path / "tft_test.pt"
        trained_model.save(path)
        loaded = TFTModel.load(path, trained_model.config)
        preds_after = loaded.predict(X_test)

        np.testing.assert_array_equal(preds_before, preds_after,
                                      err_msg="Prédictions différentes après save/load")
