"""
tcn_model.py — Temporal Convolutional Network pour prédiction directionnelle.

Le TCN remplace la récurrence du LSTM par des convolutions dilatées causales.
Avantages sur LSTM :
  - Entièrement parallélisable (pas de dépendance séquentielle)
  - Champ réceptif contrôlable via les dilatations
  - Moins de paramètres pour une capacité équivalente
  - Gradient stable (pas d'explosion/disparition comme dans les RNN)

Architecture :
─────────────
Input (batch, n_features, window)     ← TCN attend (batch, channels, length)
    ↓
TCN Block (dilation=1)                ← voit 2 pas de temps
    ↓
TCN Block (dilation=2)                ← voit 4 pas de temps
    ↓
TCN Block (dilation=4)                ← voit 8 pas de temps
    ↓
TCN Block (dilation=8)                ← voit 16 pas de temps
    ↓
TCN Block (dilation=16)               ← voit 32 pas de temps
    ↓
Global Average Pooling                ← résume la séquence entière
    ↓
Linear → Sigmoid                      ← probabilité de hausse
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    # Nombre de canaux (filtres) dans chaque couche convolutive
    # Analogue à hidden_size du LSTM — plus grand = plus de capacité
    "num_channels": 64,

    # Taille du noyau de convolution
    # kernel=3 : chaque neurone combine 3 positions temporelles adjacentes
    "kernel_size": 3,

    # Niveaux de dilation — doublement exponentiel
    # [1, 2, 4, 8, 16] → champ réceptif = (3-1)*(1+2+4+8+16)+1 = 63 jours
    # Couvre largement notre fenêtre de 30 jours
    "dilations": [1, 2, 4, 8, 16],

    # Dropout entre les couches convolutives
    "dropout": 0.2,

    # Taux d'apprentissage
    "learning_rate": 1e-3,

    # Taille des mini-batches
    "batch_size": 64,

    # Epochs max (early stopping arrêtera avant)
    "epochs": 50,

    # Patience early stopping
    "patience": 10,

    # Gradient clipping
    "grad_clip": 1.0,

    # Seed reproductibilité
    "random_state": 42,
}


# ─────────────────────────────────────────────────────────────────────────────
# Blocs de base du TCN
# ─────────────────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """
    Convolution 1D causale avec padding asymétrique.

    Une convolution standard avec kernel=3 et dilation=2 regarderait
    [t-2, t, t+2] — elle voit dans le futur, ce qui est interdit.

    Une convolution CAUSALE ne regarde que le passé [t-4, t-2, t].
    Pour l'obtenir, on ajoute du padding uniquement à GAUCHE (le passé)
    et on coupe ce qui dépasse à droite.

    Padding nécessaire = (kernel_size - 1) × dilation
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        # Padding à gauche seulement = (kernel-1) × dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,   # on surpadde des deux côtés...
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, channels, length)
        out = self.conv(x)
        # ...puis on coupe le padding droit pour rendre la conv causale
        # out[:, :, :-self.padding] supprime les self.padding dernières positions
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    Bloc résiduel TCN : CausalConv → BatchNorm → ReLU → Dropout × 2
    avec connexion résiduelle (skip connection).

    La connexion résiduelle (residual connection) est inspirée de ResNet.
    Elle additionne l'entrée du bloc à sa sortie :
        output = F(x) + x

    Avantages :
    1. Gradient fluide : le gradient peut "court-circuiter" les blocs
       via la connexion résiduelle → pas de gradient qui disparaît
    2. Apprentissage des résidus : le bloc apprend la DIFFÉRENCE par rapport
       à l'identité, pas la fonction complète — plus facile à optimiser
    3. Si un bloc est inutile, il apprend à produire F(x)≈0 → x+0 = x

    Si in_channels ≠ out_channels, une conv 1×1 adapte les dimensions
    pour que l'addition soit possible.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float):
        super().__init__()

        # Deux couches convolutives causales (standard dans les TCN)
        self.conv1 = CausalConv1d(in_channels, out_channels,
                                  kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = CausalConv1d(out_channels, out_channels,
                                  kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Connexion résiduelle : conv 1×1 si dimensions différentes
        # Conv 1×1 = transformation linéaire sans regarder les voisins
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()   # pas de transformation nécessaire

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branche principale
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Connexion résiduelle : additionner avec l'entrée adaptée
        return self.relu(out + self.residual(x))


class TCNNet(nn.Module):
    """
    Réseau TCN complet : pile de blocs TCN + classification finale.
    """

    def __init__(self, input_size: int, num_channels: int,
                 kernel_size: int, dilations: List[int], dropout: float):
        super().__init__()

        # Construction de la pile de blocs TCN
        # Chaque bloc a la même largeur (num_channels canaux)
        # sauf le premier qui prend input_size en entrée
        layers = []
        in_ch = input_size
        for dilation in dilations:
            layers.append(
                TCNBlock(in_ch, num_channels, kernel_size, dilation, dropout)
            )
            in_ch = num_channels   # les blocs suivants reçoivent num_channels

        self.network = nn.Sequential(*layers)

        # Global Average Pooling temporel
        # Réduit (batch, channels, length) → (batch, channels)
        # en moyennant sur la dimension temporelle.
        # Cela rend le modèle indépendant de la longueur de séquence
        # et condense toute l'information temporelle apprise.
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifieur final
        self.classifier = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, window, n_features)

        TCN attend (batch, channels, length) → on transpose :
        (batch, window, n_features) → (batch, n_features, window)
        """
        x = x.transpose(1, 2)          # (batch, n_features, window)
        out = self.network(x)           # (batch, num_channels, window)
        out = self.gap(out)             # (batch, num_channels, 1)
        out = out.squeeze(-1)           # (batch, num_channels)
        out = self.classifier(out)      # (batch, 1)
        return out.squeeze(1)           # (batch,)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper ML
# ─────────────────────────────────────────────────────────────────────────────

class TCNModel(BaseModel):
    """
    Wrapper TCN compatible BaseModel.
    Boucle d'entraînement identique au LSTM — seule l'architecture change.
    """

    def __init__(self, config: dict = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TCN — device: {self.device}")

    def _to_tensor(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        if y is not None:
            y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
            return X_t, y_t
        return X_t

    def _build_loader(self, X: np.ndarray, y: np.ndarray,
                      shuffle: bool) -> DataLoader:
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
        torch.manual_seed(self.config["random_state"])
        input_size = X_train.shape[2]

        self.model = TCNNet(
            input_size=input_size,
            num_channels=self.config["num_channels"],
            kernel_size=self.config["kernel_size"],
            dilations=self.config["dilations"],
            dropout=self.config["dropout"],
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"TCN — {n_params:,} paramètres | "
            f"champ réceptif ≈ {(self.config['kernel_size']-1) * sum(self.config['dilations']) + 1} jours | "
            f"device: {self.device}"
        )

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        train_loader = self._build_loader(X_train, y_train, shuffle=True)
        val_loader = self._build_loader(X_val, y_val, shuffle=False)

        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config["epochs"]):

            # ── Train ──────────────────────────────────────────────────────
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["grad_clip"]
                )
                optimizer.step()
                train_losses.append(loss.item())

            # ── Validation ─────────────────────────────────────────────────
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    preds = self.model(X_batch)
                    loss = criterion(preds, y_batch)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Époque {epoch+1:3d}/{self.config['epochs']} — "
                    f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                    f"lr: {optimizer.param_groups[0]['lr']:.2e}"
                )

            # ── Early stopping ─────────────────────────────────────────────
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
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

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        self.is_trained = True
        self._history = history

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
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        self.model.eval()
        return (self.model(self._to_tensor(X)).cpu().numpy() >= 0.5).astype(int)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        self.model.eval()
        return self.model(self._to_tensor(X)).cpu().numpy()

    def save(self, path: Path) -> Path:
        path = Path(str(path).replace(".pkl", ".pt"))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"Modèle TCN sauvegardé → {path}")
        return path

    @classmethod
    def load(cls, path: Path, config: dict):
        path = Path(str(path).replace(".pkl", ".pt"))
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        instance = cls(checkpoint["config"])
        input_size = checkpoint["state_dict"][
            "network.0.conv1.conv.weight"
        ].shape[1]
        instance.model = TCNNet(
            input_size=input_size,
            num_channels=instance.config["num_channels"],
            kernel_size=instance.config["kernel_size"],
            dilations=instance.config["dilations"],
            dropout=instance.config["dropout"],
        )
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.is_trained = True
        return instance
