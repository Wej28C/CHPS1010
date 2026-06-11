"""
Tests unitaires pour TCNModel.

Points clés à tester par rapport à LSTM :
  1. Causalité : le modèle ne doit pas voir dans le futur
  2. Connexions résiduelles : gradient doit circuler
  3. Mêmes dimensions entrée/sortie que LSTM → compatible avec train.py

Lancer : python -m pytest tests/test_tcn.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.tcn_model import TCNModel, TCNNet, CausalConv1d, TCNBlock


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
    """TCN entraîné avec config minimale — rapide."""
    X_train, y_train, X_val, y_val, _, _ = synthetic_data
    model = TCNModel({
        "num_channels": 16,
        "kernel_size": 3,
        "dilations": [1, 2],
        "dropout": 0.0,
        "epochs": 3,
        "patience": 2,
        "batch_size": 32,
    })
    model.train(X_train, y_train, X_val, y_val)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Tests sur CausalConv1d
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalConv1d:

    def test_output_length_preserved(self):
        """
        La convolution causale doit CONSERVER la longueur temporelle.

        Une conv standard avec padding=(kernel-1)*dilation ajoute des pas des
        deux côtés, allongeant la séquence. On coupe le surplus à droite pour
        revenir à la longueur d'entrée. Ce test vérifie ce comportement.
        """
        layer = CausalConv1d(in_channels=8, out_channels=16,
                             kernel_size=3, dilation=2)
        # x : (batch=4, channels=8, length=30)
        x = torch.randn(4, 8, 30)
        out = layer(x)
        assert out.shape == (4, 16, 30), \
            f"La longueur doit rester 30, obtenu {out.shape}"

    def test_causal_no_future_leak(self):
        """
        TEST DE CAUSALITÉ — le plus important pour les séries temporelles.

        Principe : si on change la valeur à la position t=5 dans l'entrée,
        les sorties aux positions t < 5 ne doivent PAS changer.

        Si elles changent, c'est que la conv regarde vers le futur → fuite !

        Méthode : comparer les sorties pour deux entrées identiques sauf
        à partir de t=5. Toutes les sorties à t < 5 doivent être identiques.
        """
        torch.manual_seed(42)
        layer = CausalConv1d(in_channels=4, out_channels=8,
                             kernel_size=2, dilation=1)
        layer.eval()

        x1 = torch.randn(1, 4, 20)
        x2 = x1.clone()
        x2[:, :, 5:] = torch.randn(1, 4, 15)  # modifier à partir de t=5

        with torch.no_grad():
            out1 = layer(x1)
            out2 = layer(x2)

        # Les sorties aux positions 0..4 (avant la modification) doivent être
        # identiques — sinon la conv regarde dans le futur
        assert torch.allclose(out1[:, :, :5], out2[:, :, :5], atol=1e-6), \
            "FUITE CAUSALE : la conv regarde dans le futur !"


# ─────────────────────────────────────────────────────────────────────────────
# Tests sur TCNBlock
# ─────────────────────────────────────────────────────────────────────────────

class TestTCNBlock:

    def test_output_shape_same_channels(self):
        """Le bloc conserve les dimensions quand in_channels == out_channels."""
        block = TCNBlock(in_channels=16, out_channels=16,
                         kernel_size=3, dilation=1, dropout=0.0)
        x = torch.randn(4, 16, 30)
        out = block(x)
        assert out.shape == (4, 16, 30)

    def test_output_shape_different_channels(self):
        """Le bloc adapte les canaux quand in_channels != out_channels."""
        block = TCNBlock(in_channels=18, out_channels=64,
                         kernel_size=3, dilation=1, dropout=0.0)
        x = torch.randn(4, 18, 30)
        out = block(x)
        assert out.shape == (4, 64, 30)

    def test_residual_identity_used(self):
        """Quand in_channels == out_channels, on utilise nn.Identity."""
        block = TCNBlock(in_channels=32, out_channels=32,
                         kernel_size=3, dilation=1, dropout=0.0)
        assert isinstance(block.residual, torch.nn.Identity)

    def test_residual_conv_used(self):
        """Quand in_channels != out_channels, on utilise une conv 1x1."""
        block = TCNBlock(in_channels=18, out_channels=64,
                         kernel_size=3, dilation=1, dropout=0.0)
        assert isinstance(block.residual, torch.nn.Conv1d)
        assert block.residual.kernel_size == (1,)


# ─────────────────────────────────────────────────────────────────────────────
# Tests sur TCNNet (réseau complet)
# ─────────────────────────────────────────────────────────────────────────────

class TestTCNNet:

    def test_forward_shape(self):
        """La sortie a la bonne forme : (batch,)."""
        net = TCNNet(input_size=18, num_channels=32,
                     kernel_size=3, dilations=[1, 2, 4], dropout=0.0)
        x = torch.randn(10, 30, 18)   # (batch, window, features)
        out = net(x)
        assert out.shape == (10,), f"Forme attendue (10,), obtenu {out.shape}"

    def test_output_range(self):
        """La sortie est dans [0, 1] (Sigmoid en sortie)."""
        net = TCNNet(input_size=18, num_channels=32,
                     kernel_size=3, dilations=[1, 2], dropout=0.0)
        x = torch.randn(20, 30, 18)
        out = net(x)
        assert torch.all(out >= 0) and torch.all(out <= 1)

    def test_no_nan(self):
        """Aucun NaN dans la sortie."""
        net = TCNNet(input_size=18, num_channels=32,
                     kernel_size=3, dilations=[1, 2, 4, 8], dropout=0.0)
        x = torch.randn(8, 30, 18)
        out = net(x)
        assert not torch.any(torch.isnan(out))

    def test_receptive_field(self):
        """
        Test du champ réceptif.

        Avec kernel=2 et dilations=[1,2,4], le champ réceptif théorique est
        (kernel-1) * sum(dilations) + 1 = 1 * 7 + 1 = 8 pas de temps.

        On ne peut pas tester ça directement sur le modèle, mais on peut
        vérifier que différentes longueurs de séquence fonctionnent (grâce
        au Global Average Pooling qui est invariant à la longueur).
        """
        net = TCNNet(input_size=18, num_channels=16,
                     kernel_size=2, dilations=[1, 2, 4], dropout=0.0)
        # Séquence courte
        x_short = torch.randn(4, 15, 18)
        out_short = net(x_short)
        assert out_short.shape == (4,)

        # Séquence longue
        x_long = torch.randn(4, 60, 18)
        out_long = net(x_long)
        assert out_long.shape == (4,)


# ─────────────────────────────────────────────────────────────────────────────
# Tests sur TCNModel (wrapper BaseModel)
# ─────────────────────────────────────────────────────────────────────────────

class TestTCNModel:

    def test_instantiation(self):
        model = TCNModel()
        assert not model.is_trained
        assert model.config["num_channels"] == 64
        assert model.config["kernel_size"] == 3
        assert model.config["dilations"] == [1, 2, 4, 8, 16]

    def test_predict_before_train_raises(self):
        model = TCNModel()
        X = np.random.random((10, 30, 18)).astype(np.float32)
        with pytest.raises(RuntimeError, match="pas encore entraîné"):
            model.predict(X)

    def test_train_sets_is_trained(self, synthetic_data):
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        model = TCNModel({
            "num_channels": 16, "dilations": [1, 2],
            "epochs": 2, "patience": 1, "batch_size": 32,
        })
        model.train(X_train, y_train, X_val, y_val)
        assert model.is_trained

    def test_train_returns_dict_with_metrics(self, synthetic_data):
        X_train, y_train, X_val, y_val, _, _ = synthetic_data
        model = TCNModel({
            "num_channels": 16, "dilations": [1, 2],
            "epochs": 2, "patience": 1, "batch_size": 32,
        })
        result = model.train(X_train, y_train, X_val, y_val)
        assert "best_val_loss" in result
        assert "train_directional_accuracy" in result
        assert "val_directional_accuracy" in result

    def test_history_recorded(self, trained_model):
        assert hasattr(trained_model, "_history")
        assert "train_loss" in trained_model._history
        assert len(trained_model._history["train_loss"]) > 0


class TestTCNPredictions:

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


class TestTCNSaveLoad:

    def test_save_creates_pt_file(self, trained_model, tmp_path):
        path = tmp_path / "tcn_test.pt"
        saved = trained_model.save(path)
        assert saved.exists()
        assert saved.suffix == ".pt"

    def test_save_converts_pkl_to_pt(self, trained_model, tmp_path):
        """save() doit convertir .pkl → .pt automatiquement."""
        path = tmp_path / "tcn_test.pkl"
        saved = trained_model.save(path)
        assert saved.suffix == ".pt"

    def test_load_same_predictions(self, trained_model, synthetic_data, tmp_path):
        """Après save/load, les prédictions doivent être identiques."""
        _, _, _, _, X_test, _ = synthetic_data
        preds_before = trained_model.predict(X_test)

        path = tmp_path / "tcn_test.pt"
        trained_model.save(path)

        loaded = TCNModel.load(path, trained_model.config)
        preds_after = loaded.predict(X_test)

        np.testing.assert_array_equal(preds_before, preds_after,
                                      err_msg="Prédictions différentes après save/load")
