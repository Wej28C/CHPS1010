"""
tft_model.py — Temporal Fusion Transformer (TFT) simplifié.

Référence : Lim et al., 2019 — "Temporal Fusion Transformers for
Interpretable Multi-horizon Time Series Forecasting"

Simplifications par rapport au papier original :
  - Pas de covariables futures connues (on n'a que le passé)
  - Pas de covariables statiques (on entraîne un modèle par actif)
  - Classification binaire au lieu de prédiction quantile
  - Moins de têtes d'attention pour rester léger sur CPU

Ce qui est conservé du papier :
  - Variable Selection Network (VSN) : sélection des features pertinentes
  - Gated Residual Network (GRN) : transformations avec portes GLU
  - LSTM encoder sur les représentations sélectionnées
  - Multi-Head Self-Attention sur les sorties LSTM
  - Add & Norm à chaque étape (stabilité de l'entraînement)

Avantage différenciateur vs LSTM/TCN :
  L'attention produit des poids INTERPRÉTABLES : on peut visualiser quels
  jours du passé ont le plus influencé la prédiction. C'est un argument
  fort pour le rapport et pour expliquer le modèle à un investisseur.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    # Dimension des représentations internes (hidden dim pour tout le modèle)
    # Analogue à hidden_size du LSTM — tous les vecteurs internes ont cette taille
    "d_model": 64,

    # Nombre de têtes d'attention dans le Multi-Head Attention
    # Chaque tête apprend un type de dépendance temporelle différent
    # d_model doit être divisible par n_heads
    "n_heads": 4,

    # Nombre de couches LSTM dans l'encoder
    "lstm_layers": 1,

    # Dropout général
    "dropout": 0.1,

    # Taux d'apprentissage
    "learning_rate": 1e-3,

    # Taille des mini-batches
    "batch_size": 64,

    # Epochs max
    "epochs": 50,

    # Patience early stopping
    "patience": 10,

    # Gradient clipping
    "grad_clip": 1.0,

    # Seed reproductibilité
    "random_state": 42,
}


# ─────────────────────────────────────────────────────────────────────────────
# Blocs de base du TFT
# ─────────────────────────────────────────────────────────────────────────────

class GatedLinearUnit(nn.Module):
    """
    GLU — Gated Linear Unit.

    Mécanisme de porte inspiré des LSTMs mais plus simple.
    Divise la dimension en deux moitiés :
      - La première moitié porte l'information (valeurs)
      - La seconde moitié est une porte sigmoid (0 à 1)

    output = valeurs × sigmoid(porte)

    Intuition : la porte apprend à "ouvrir" ou "fermer" le flux
    d'information pour chaque dimension. Si la porte ≈ 0, l'information
    est bloquée. Si la porte ≈ 1, elle passe intégralement.

    Avantage : le modèle peut ignorer des transformations inutiles
    plutôt que d'être forcé de les propager.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # La projection est 2x la taille de sortie car on va la diviser en 2
        self.linear = nn.Linear(input_dim, output_dim * 2)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.linear(x)                          # (..., output_dim*2)
        values, gate = projected.split(self.output_dim, dim=-1)
        return values * torch.sigmoid(gate)                 # (..., output_dim)


class GatedResidualNetwork(nn.Module):
    """
    GRN — Gated Residual Network.

    Bloc central du TFT. Combine :
    1. Une transformation non-linéaire avec ELU (Exponential Linear Unit)
    2. Une porte GLU pour filtrer l'information
    3. Une connexion résiduelle + LayerNorm pour la stabilité

    Schéma :
        x ──→ Linear → ELU → Linear → GLU → Add → LayerNorm → output
        ↑                                    ↑
        └────────────────────────────────────┘  (résiduelle)

    Si input_dim ≠ output_dim, une projection linéaire adapte les dimensions
    pour que l'addition résiduelle soit possible.

    Le GRN est utilisé à plusieurs endroits dans le TFT :
    - Dans le Variable Selection Network (par feature)
    - Avant l'attention
    - Après l'attention
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.glu = GatedLinearUnit(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Projection résiduelle si les dimensions diffèrent
        self.residual_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)        # adapte les dimensions

        h = F.elu(self.linear1(x))              # transformation non-linéaire
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.glu(h)                         # filtrage par porte

        return self.norm(h + residual)          # Add & Norm


class VariableSelectionNetwork(nn.Module):
    """
    VSN — Variable Selection Network.

    Apprend à pondérer chaque feature selon son importance.

    Architecture :
    1. Chaque feature est projetée individuellement vers d_model dimensions
       (chaque feature a sa propre couche linéaire)
    2. Toutes les features projetées sont concaténées et passées dans un GRN
    3. Un softmax produit un vecteur de poids (un poids par feature)
    4. Les features projetées sont combinées avec ces poids (somme pondérée)

    Sortie :
    - La représentation combinée (batch, window, d_model)
    - Les poids d'attention par feature (batch, window, n_features)
      → Ces poids sont INTERPRÉTABLES : on voit quelles features comptent

    Paramètres
    ----------
    n_features : nombre de features d'entrée (18 dans notre cas)
    d_model    : dimension des représentations internes
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Une projection linéaire par feature : shape (1,) → (d_model,)
        # Pourquoi des projections séparées ?
        # Chaque feature a une échelle et une interprétation différente
        # (RSI ∈ [0,100], volume_norm ≈ [0,5], log_return ≈ [-0.1, 0.1])
        # Une projection dédiée permet au modèle d'apprendre la
        # "bonne façon" de représenter chaque feature.
        self.feature_projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])

        # GRN qui calcule les poids de sélection à partir de toutes les features
        # Entrée : concaténation de toutes les projections = n_features * d_model
        self.selection_grn = GRN_wrapper(
            input_dim=n_features * d_model,
            hidden_dim=d_model,
            output_dim=n_features,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (batch, window, n_features)

        Retourne :
          combined : (batch, window, d_model)  — représentation pondérée
          weights  : (batch, window, n_features) — poids par feature
        """
        batch, window, _ = x.shape

        # Projeter chaque feature séparément
        # x[:, :, i] → (batch, window) → unsqueeze → (batch, window, 1) → proj → (batch, window, d_model)
        projections = []
        for i, proj in enumerate(self.feature_projections):
            feat = x[:, :, i].unsqueeze(-1)      # (batch, window, 1)
            projections.append(proj(feat))        # (batch, window, d_model)

        # Stack : liste de (batch, window, d_model) → (batch, window, n_features, d_model)
        stacked = torch.stack(projections, dim=2)

        # Concaténer pour le GRN de sélection : (batch, window, n_features*d_model)
        flat = stacked.reshape(batch, window, self.n_features * self.d_model)

        # Calculer les poids de sélection : (batch, window, n_features)
        weights = torch.softmax(self.selection_grn(flat), dim=-1)

        # Combinaison pondérée : somme(weight_i * projection_i) sur les features
        # weights.unsqueeze(-1) → (batch, window, n_features, 1)
        # stacked               → (batch, window, n_features, d_model)
        combined = (weights.unsqueeze(-1) * stacked).sum(dim=2)

        return combined, weights


class GRN_wrapper(nn.Module):
    """Wrapper léger du GRN pour usage dans le VSN (sans connexion résiduelle complexe)."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.elu(self.linear1(x))))


# ─────────────────────────────────────────────────────────────────────────────
# Réseau TFT complet
# ─────────────────────────────────────────────────────────────────────────────

class TFTNet(nn.Module):
    """
    Réseau TFT complet.

    Pipeline :
      Input → VSN → LSTM Encoder → Multi-Head Attention → GRN → Classifieur
    """

    def __init__(self, n_features: int, d_model: int, n_heads: int,
                 lstm_layers: int, dropout: float):
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) doit être divisible par n_heads ({n_heads})"

        # 1. Variable Selection Network
        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)

        # 2. GRN après VSN (transformation supplémentaire)
        self.input_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        # 3. LSTM Encoder
        # Encode la séquence de représentations sélectionnées
        # batch_first=True : (batch, window, d_model) → (batch, window, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # 4. Multi-Head Self-Attention
        # Le modèle "regarde toute la séquence" et pondère les positions
        # batch_first=True : attend (batch, seq, d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # 5. GRN post-attention
        self.output_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        # 6. Pooling temporel + classifieur
        # On moyenne sur la dimension temporelle pour condenser en vecteur fixe
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (batch, window, n_features)

        Retourne :
          output   : (batch,) — probabilité de hausse
          vsn_weights : (batch, window, n_features) — poids d'interprétabilité
        """
        # ── 1. Variable Selection ──────────────────────────────────────────
        # Quelles features sont importantes à chaque pas de temps ?
        selected, vsn_weights = self.vsn(x)         # (batch, window, d_model)

        # ── 2. GRN d'entrée ────────────────────────────────────────────────
        encoded = self.input_grn(selected)           # (batch, window, d_model)

        # ── 3. LSTM Encoder ────────────────────────────────────────────────
        # Résume la séquence en tenant compte de l'ordre temporel
        lstm_out, _ = self.lstm(encoded)             # (batch, window, d_model)

        # ── 4. Multi-Head Self-Attention ───────────────────────────────────
        # Q, K, V = lstm_out (self-attention : le modèle se regarde lui-même)
        # attn_weights : (batch, window, window) — qui regarde qui
        attn_out, _ = self.attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
        )
        # Add & Norm : additionner avec le résidu LSTM + normaliser
        attn_out = self.attn_norm(attn_out + lstm_out)

        # ── 5. GRN post-attention ──────────────────────────────────────────
        out = self.output_grn(attn_out)              # (batch, window, d_model)

        # ── 6. Pooling temporel + classification ──────────────────────────
        # Moyenne sur la dimension temporelle
        pooled = out.mean(dim=1)                     # (batch, d_model)
        prob = self.classifier(pooled).squeeze(1)    # (batch,)

        return prob, vsn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper ML
# ─────────────────────────────────────────────────────────────────────────────

class TFTModel(BaseModel):
    """
    Wrapper TFT compatible BaseModel.
    Boucle d'entraînement identique aux autres modèles DL.
    Expose en plus get_feature_importance() pour l'interprétabilité.
    """

    def __init__(self, config: dict = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        super().__init__(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vsn_weights_history = None
        logger.info(f"TFT — device: {self.device}")

    def _to_tensor(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        if y is not None:
            y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
            return X_t, y_t
        return X_t

    def _build_loader(self, X: np.ndarray, y: np.ndarray,
                      shuffle: bool) -> DataLoader:
        X_t, y_t = self._to_tensor(X, y)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        torch.manual_seed(self.config["random_state"])
        n_features = X_train.shape[2]

        self.model = TFTNet(
            n_features=n_features,
            d_model=self.config["d_model"],
            n_heads=self.config["n_heads"],
            lstm_layers=self.config["lstm_layers"],
            dropout=self.config["dropout"],
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"TFT — {n_params:,} paramètres | device: {self.device}")

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
                preds, _ = self.model(X_batch)
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
                    preds, _ = self.model(X_batch)
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
        probs, _ = self.model(self._to_tensor(X))
        return (probs.cpu().numpy() >= 0.5).astype(int)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        self.model.eval()
        probs, _ = self.model(self._to_tensor(X))
        return probs.cpu().numpy()

    @torch.no_grad()
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les poids VSN moyennés sur le temps et les exemples.

        Shape retournée : (n_features,)
        Interprétation : poids[i] = importance moyenne de la feature i.
        Ces poids somment à 1 (softmax dans le VSN).

        Usage dans le rapport :
          importances = model.get_feature_importance(X_test)
          # importances[0] = importance de 'open'
          # importances[6] = importance de 'sma_10'
          # etc.
        """
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        self.model.eval()
        X_t = self._to_tensor(X)
        _, vsn_weights = self.model(X_t)
        # vsn_weights : (batch, window, n_features)
        # Moyenne sur batch et window → (n_features,)
        return vsn_weights.mean(dim=(0, 1)).cpu().numpy()

    def save(self, path: Path) -> Path:
        path = Path(str(path).replace(".pkl", ".pt"))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"Modèle TFT sauvegardé → {path}")
        return path

    @classmethod
    def load(cls, path: Path, config: dict):
        path = Path(str(path).replace(".pkl", ".pt"))
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        instance = cls(checkpoint["config"])

        # Déduire n_features depuis les poids du VSN
        # La première projection feature a shape (d_model, 1) → on compte les projections
        n_features = sum(
            1 for k in checkpoint["state_dict"]
            if k.startswith("vsn.feature_projections.")
            and k.endswith(".weight")
        )
        instance.model = TFTNet(
            n_features=n_features,
            d_model=instance.config["d_model"],
            n_heads=instance.config["n_heads"],
            lstm_layers=instance.config["lstm_layers"],
            dropout=instance.config["dropout"],
        )
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.is_trained = True
        return instance
