"""
lstm_model.py — LSTM (Long Short-Term Memory) pour prédiction directionnelle.

Le LSTM est un réseau de neurones récurrent conçu pour apprendre des
dépendances à long terme dans les séquences. Contrairement à XGBoost,
il traite les données dans l'ordre temporel et maintient un état mémoire.

Architecture utilisée :
──────────────────────
Input (batch, 30, 18)
    ↓
LSTM (N couches, hidden_size neurones)   ← apprend les patterns temporels
    ↓
Dropout                                  ← régularisation (évite l'overfitting)
    ↓
Linear(hidden_size → 1)                  ← projection vers une valeur scalaire
    ↓
Sigmoid                                  ← convertit en probabilité [0, 1]
    ↓
Output : P(hausse J+1)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    # Dimension de l'état caché du LSTM
    # Plus grand = plus de capacité mais plus lent et risque d'overfitting
    "hidden_size": 128,

    # Nombre de couches LSTM empilées
    # 2 couches capturent des patterns plus abstraits qu'une seule
    "num_layers": 2,

    # Dropout entre les couches LSTM (régularisation)
    # 0.2 = 20% des neurones désactivés aléatoirement pendant l'entraînement
    "dropout": 0.2,

    # Taux d'apprentissage de l'optimiseur Adam
    "learning_rate": 1e-3,

    # Nombre d'exemples traités en parallèle à chaque étape
    # 64 est un bon compromis vitesse/stabilité pour nos ~1700 exemples
    "batch_size": 64,

    # Nombre maximum d'époques (early stopping arrêtera avant si nécessaire)
    "epochs": 50,

    # Arrêter si pas d'amélioration après N époques consécutives
    "patience": 10,

    # Gradient clipping : valeur maximale de la norme du gradient
    # ESSENTIEL pour les LSTM : sans ça, les gradients peuvent exploser
    # (gradient explosion), rendant l'entraînement instable
    "grad_clip": 1.0,

    # Seed pour reproductibilité
    "random_state": 42,
}


# ─────────────────────────────────────────────────────────────────────────────
# Architecture du réseau de neurones
# ─────────────────────────────────────────────────────────────────────────────

class LSTMNet(nn.Module):
    """
    Réseau LSTM PyTorch.

    On sépare l'architecture (LSTMNet) du wrapper ML (LSTMModel)
    pour deux raisons :
    1. LSTMNet est un nn.Module pur → facile à sérialiser avec torch.save()
    2. LSTMModel gère la logique train/predict/evaluate commune à BaseModel
    """

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()

        # La couche LSTM principale
        # batch_first=True : l'entrée est (batch, temps, features)
        #   au lieu de (temps, batch, features) — plus intuitif
        # dropout s'applique ENTRE les couches (pas sur la dernière)
        # → c'est pourquoi dropout=0 si num_layers=1 (pas "entre" couches)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dropout supplémentaire après la dernière couche LSTM
        self.dropout = nn.Dropout(dropout)

        # Couche linéaire : projette hidden_size → 1 scalaire
        self.fc = nn.Linear(hidden_size, 1)

        # Sigmoid : convertit le scalaire en probabilité [0, 1]
        # P=0.5 → incertain, P>0.5 → prédit hausse, P<0.5 → prédit baisse
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant (forward pass).

        x : (batch, window, n_features)

        On passe la séquence entière dans le LSTM.
        On ne garde que la sortie du DERNIER pas de temps (out[:, -1, :])
        car c'est là que toute l'information de la séquence est condensée.

        C'est l'approche "many-to-one" : N entrées → 1 sortie.
        """
        # out : (batch, window, hidden_size) — sortie à chaque pas de temps
        # _   : (h_n, c_n) — états finaux (non utilisés ici)
        out, _ = self.lstm(x)

        # Prendre uniquement le dernier pas de temps
        # out[:, -1, :] → (batch, hidden_size)
        last_hidden = out[:, -1, :]

        # Régularisation + projection
        last_hidden = self.dropout(last_hidden)
        logit = self.fc(last_hidden)           # (batch, 1)

        # Probabilité finale
        prob = self.sigmoid(logit)             # (batch, 1) dans [0, 1]
        return prob.squeeze(1)                 # (batch,)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper ML (hérite de BaseModel)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMModel(BaseModel):
    """
    Wrapper autour de LSTMNet pour l'intégration avec BaseModel et MLflow.

    Gère :
    - La boucle d'entraînement avec early stopping
    - Le gradient clipping
    - La conversion numpy ↔ torch.Tensor
    - La sauvegarde / chargement du modèle
    """

    def __init__(self, config: dict = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)
        # Utiliser GPU si disponible, sinon CPU
        # Sur PC local : CPU. Sur ROMEO avec GPU : CUDA automatiquement
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"LSTM — device: {self.device}")

    def _to_tensor(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Convertit numpy arrays en tenseurs PyTorch sur le bon device."""
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        if y is not None:
            y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
            return X_t, y_t
        return X_t

    def _build_loader(self, X: np.ndarray, y: np.ndarray,
                      shuffle: bool) -> DataLoader:
        """
        Crée un DataLoader PyTorch à partir des arrays numpy.

        DataLoader gère automatiquement :
        - Le découpage en mini-batches (batch_size=64)
        - Le mélange aléatoire (shuffle=True pour train, False pour val/test)
        - Le chargement en mémoire efficace

        shuffle=True sur le train : on mélange l'ordre des exemples à chaque
        époque pour éviter que le modèle apprenne l'ordre spécifique
        (tout en respectant que X[i] correspond à y[i])
        """
        X_t, y_t = self._to_tensor(X, y)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=self.config["batch_size"],
                          shuffle=shuffle)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """
        Entraîne le LSTM avec early stopping.

        Boucle d'entraînement standard PyTorch :
        ──────────────────────────────────────────
        Pour chaque époque :
          1. Mode train : model.train()
          2. Pour chaque mini-batch :
             a. Forward pass : prédictions = model(X_batch)
             b. Calcul de la loss : BCELoss(prédictions, y_batch)
             c. Backward pass : calcul des gradients
             d. Gradient clipping : évite l'explosion des gradients
             e. Mise à jour des poids : optimizer.step()
          3. Mode eval : model.eval()
          4. Calcul de la val_loss
          5. Early stopping si pas d'amélioration
        """
        torch.manual_seed(self.config["random_state"])

        input_size = X_train.shape[2]   # nombre de features (18)

        # Instancier le réseau et le déplacer sur le device
        self.model = LSTMNet(
            input_size=input_size,
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
        ).to(self.device)

        logger.info(
            f"LSTM — {sum(p.numel() for p in self.model.parameters()):,} paramètres | "
            f"device: {self.device}"
        )

        # BCELoss = Binary Cross-Entropy Loss
        # Mesure la différence entre P(hausse) prédite et la vraie cible (0 ou 1)
        # C'est la loss standard pour la classification binaire
        criterion = nn.BCELoss()

        # Adam = Adaptive Moment Estimation
        # Optimiseur standard, adapte le taux d'apprentissage par paramètre
        # Bien meilleur que SGD simple pour les LSTM
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"]
        )

        # ReduceLROnPlateau : réduit le LR si la val_loss stagne
        # Permet de "zoomer" vers le minimum sans osciller
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        train_loader = self._build_loader(X_train, y_train, shuffle=True)
        val_loader = self._build_loader(X_val, y_val, shuffle=False)

        # ── Early stopping ────────────────────────────────────────────────
        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config["epochs"]):

            # ── Phase entraînement ────────────────────────────────────────
            self.model.train()   # active dropout, batch norm, etc.
            train_losses = []

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()           # réinitialiser les gradients

                preds = self.model(X_batch)     # forward pass
                loss = criterion(preds, y_batch)  # calcul loss

                loss.backward()                 # backward pass (calcul gradients)

                # Gradient clipping : si la norme du gradient dépasse grad_clip,
                # on la ramène à grad_clip. Empêche les "gradient explosions"
                # fréquents dans les LSTM sur de longues séquences.
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["grad_clip"]
                )

                optimizer.step()                # mise à jour des poids
                train_losses.append(loss.item())

            # ── Phase validation ──────────────────────────────────────────
            self.model.eval()    # désactive dropout pour l'évaluation
            val_losses = []

            with torch.no_grad():   # pas de calcul de gradient en validation
                for X_batch, y_batch in val_loader:
                    preds = self.model(X_batch)
                    loss = criterion(preds, y_batch)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            # Log toutes les 5 époques pour ne pas surcharger la console
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Époque {epoch+1:3d}/{self.config['epochs']} — "
                    f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                    f"lr: {optimizer.param_groups[0]['lr']:.2e}"
                )

            # ── Early stopping ────────────────────────────────────────────
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                # Sauvegarder les meilleurs poids en mémoire
                best_weights = {k: v.cpu().clone()
                                for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config["patience"]:
                    logger.info(
                        f"Early stopping à l'époque {epoch+1} "
                        f"(patience={self.config['patience']})"
                    )
                    break

        # Restaurer les meilleurs poids trouvés pendant l'entraînement
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            logger.info(f"Meilleurs poids restaurés (val_loss={best_val_loss:.4f})")

        self.is_trained = True
        self._history = history   # gardé pour les graphiques (rapport)

        train_metrics = self.evaluate(X_train, y_train)
        val_metrics = self.evaluate(X_val, y_val)
        logger.info(f"Train — {train_metrics}")
        logger.info(f"Val   — {val_metrics}")

        return {
            "best_epoch": len(history["train_loss"]),
            "best_val_loss": round(best_val_loss, 4),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédictions binaires (0 ou 1) avec seuil à 0.5."""
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        self.model.eval()
        X_t = self._to_tensor(X)
        probas = self.model(X_t).cpu().numpy()
        return (probas >= 0.5).astype(int)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilité P(hausse) pour chaque exemple."""
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        self.model.eval()
        X_t = self._to_tensor(X)
        return self.model(X_t).cpu().numpy()

    def save(self, path: Path) -> Path:
        """Sauvegarde le state_dict PyTorch. Retourne le chemin réel (.pt)."""
        path = Path(str(path).replace(".pkl", ".pt"))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"Modèle LSTM sauvegardé → {path}")
        return path

    @classmethod
    def load(cls, path: Path, config: dict):
        """Charge un modèle LSTM sauvegardé."""
        path = Path(str(path).replace(".pkl", ".pt"))
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        instance = cls(checkpoint["config"])
        input_size = checkpoint["state_dict"]["lstm.weight_ih_l0"].shape[1]
        instance.model = LSTMNet(
            input_size=input_size,
            hidden_size=instance.config["hidden_size"],
            num_layers=instance.config["num_layers"],
            dropout=instance.config["dropout"],
        )
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.is_trained = True
        return instance
