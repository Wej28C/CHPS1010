"""
preprocess.py — Calcul des features techniques et préparation des séquences.

Pour chaque actif, ce script :
  1. Charge les données OHLCV brutes depuis data/raw/
  2. Calcule les indicateurs techniques (SMA, RSI, MACD, volatilité, volume norm.)
  3. Crée la cible binaire : direction J+1 (1=hausse, 0=baisse)
  4. Normalise les features (MinMaxScaler ajusté sur le train uniquement)
  5. Crée les séquences fenêtres glissantes pour LSTM/TCN/TFT
  6. Effectue le walk-forward validation split (pas de data leakage)
  7. Sauvegarde en Parquet + CSV dans data/processed/

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --window 30 --ticker MC.PA
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SCALERS_DIR = Path("data/processed/scalers")

# Tickers du projet (OR.PA remplace MONO.PA introuvable sur Yahoo Finance)
DEFAULT_TICKERS = ["MC.PA", "CFR.SW", "RMS.PA", "BRBY.L", "OR.PA"]

# Taille de la fenêtre glissante :
# 30 jours = ~6 semaines de bourse, capture les tendances court/moyen terme
# sans être trop long (plus la fenêtre est grande, plus on perd de données au début)
DEFAULT_WINDOW = 30

# Fraction du dataset réservée au test (jamais vue pendant l'entraînement)
TEST_SIZE = 0.15   # 15% = ~375 jours sur 2500

# Fraction pour la validation (pour le early stopping / sélection de modèle)
VAL_SIZE = 0.15    # 15% = ~375 jours


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prétraitement features + séquences")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="Taille de la fenêtre glissante (jours)")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--output-dir", default=str(PROCESSED_DIR))
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 : Calcul des indicateurs techniques
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Calcule tous les indicateurs techniques à partir des colonnes OHLCV.

    On travaille sur une copie pour ne pas modifier les données brutes.
    Chaque indicateur est expliqué ci-dessous.
    """
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    # ── Log-return ────────────────────────────────────────────────────────────
    # ln(P_t / P_{t-1}) — rendement logarithmique journalier.
    # Propriété clé : additivité — le rendement sur N jours = somme des log-returns.
    # Les prix boursiers sont non-stationnaires (racine unitaire), les log-returns
    # sont (en général) stationnaires → les modèles convergent beaucoup mieux.
    df["log_return"] = np.log(close / close.shift(1))

    # ── Moyennes mobiles simples (SMA) ────────────────────────────────────────
    # SMA_n = moyenne arithmétique des n derniers prix de clôture.
    # - SMA10 : tendance très court terme (2 semaines)
    # - SMA20 : tendance court terme (1 mois boursier)
    # - SMA50 : tendance moyen terme (2,5 mois)
    # Le ratio Close/SMA indique si le prix est au-dessus ou en dessous
    # de sa moyenne → signal directionnel pour les modèles.
    for window in [10, 20, 50]:
        df[f"sma_{window}"] = close.rolling(window).mean()
        # Ratio normalisé : plus stable numériquement que la différence brute
        df[f"sma_{window}_ratio"] = close / df[f"sma_{window}"]

    # ── RSI — Relative Strength Index ────────────────────────────────────────
    # RSI = 100 - 100 / (1 + RS) où RS = gain moyen / perte moyenne sur 14 jours
    # Interprétation :
    #   RSI > 70 → suracheté (le prix a monté trop vite → retournement possible)
    #   RSI < 30 → survendu  (le prix a trop baissé  → rebond possible)
    #   RSI ≈ 50 → neutre
    delta = close.diff()
    gain = delta.clip(lower=0)     # ne garde que les hausses
    loss = -delta.clip(upper=0)    # ne garde que les baisses (positif)

    # Moyenne exponentielle (EWM) plutôt que rolling : plus réactive aux
    # changements récents, standard dans l'industrie financière.
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── MACD — Moving Average Convergence Divergence ──────────────────────────
    # MACD = EMA(12) - EMA(26)
    # Signal = EMA(9) du MACD
    # Histogramme = MACD - Signal
    #
    # Interprétation :
    #   MACD croise Signal par le haut → signal d'achat (momentum haussier)
    #   MACD croise Signal par le bas  → signal de vente (momentum baissier)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ── Volatilité glissante ──────────────────────────────────────────────────
    # Écart-type des log-returns sur 20 jours, annualisé (×√252).
    # 252 = nombre moyen de jours de bourse par an.
    # La volatilité annualisée permet de comparer avec les benchmarks de marché.
    # Une forte volatilité = contexte incertain = prédiction plus difficile.
    df["volatility_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)

    # ── Volume normalisé ──────────────────────────────────────────────────────
    # Ratio Volume_t / moyenne_20j(Volume).
    # Un ratio > 2 signifie un volume inhabituel → confirmation d'un mouvement.
    # On normalise par la moyenne glissante (pas la moyenne globale) pour
    # éviter que les changements structurels de liquidité sur 10 ans ne biaisent
    # la feature.
    vol_mean = volume.rolling(20).mean().replace(0, np.nan)
    df["volume_norm"] = volume / vol_mean

    # ── Cible : direction J+1 ─────────────────────────────────────────────────
    # C'est ce qu'on cherche à prédire.
    # 1 = le prix de clôture de DEMAIN sera supérieur à celui d'AUJOURD'HUI
    # 0 = il sera inférieur ou égal
    # On utilise le log-return de J+1 (shift(-1) = regarder en avant dans le temps)
    # ATTENTION : cette colonne doit absolument être exclue des features d'entrée !
    # Elle ne sert qu'à construire le vecteur y (cible).
    df["target"] = (df["log_return"].shift(-1) > 0).astype(int)

    # ── Suppression des NaN créés par les indicateurs ─────────────────────────
    # Les SMA50 nécessitent 50 jours, la volatilité 20 jours, etc.
    # On perd les premières lignes — c'est inévitable et intentionnel.
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    logger.info(f"{ticker} : {n_dropped} lignes supprimées (warmup indicateurs), "
                f"{len(df)} lignes restantes")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 : Walk-forward validation split
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_split(df: pd.DataFrame, ticker: str):
    """
    Découpe le dataset en train / val / test de façon temporellement correcte.

    POURQUOI PAS UN SPLIT ALÉATOIRE ?
    ──────────────────────────────────
    Un split aléatoire sur des séries temporelles crée du DATA LEAKAGE :
    le modèle voit des données "futures" pendant l'entraînement.

    Exemple de leakage : si la date 2020-03-10 (krach COVID) est dans le train
    et 2020-03-09 dans le test, le modèle "sait" qu'un krach arrive.
    En production réelle, ce n'est pas possible → les métriques seraient
    optimistes artificiellement.

    LA BONNE APPROCHE :
    ──────────────────
    |─── TRAIN (70%) ──────────|─── VAL (15%) ──|─── TEST (15%) ──|
    2015                     2021.5           2023              2025

    - TRAIN : le modèle apprend dessus
    - VAL   : sert au early stopping et à la sélection des hyperparamètres
    - TEST  : évaluation finale, touché UNE SEULE FOIS à la fin du projet

    Le test est toujours la période la plus récente (2023-2025).
    """
    n = len(df)
    n_test = int(n * TEST_SIZE)
    n_val = int(n * VAL_SIZE)
    n_train = n - n_val - n_test

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train: n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    # Vérification anti-leakage : les dates ne doivent pas se chevaucher
    assert train_df.index.max() < val_df.index.min(), "Leakage train/val !"
    assert val_df.index.max() < test_df.index.min(), "Leakage val/test !"

    logger.info(
        f"{ticker} split — "
        f"train: {len(train_df)} ({train_df.index.min().date()}→{train_df.index.max().date()}) | "
        f"val: {len(val_df)} ({val_df.index.min().date()}→{val_df.index.max().date()}) | "
        f"test: {len(test_df)} ({test_df.index.min().date()}→{test_df.index.max().date()})"
    )
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 : Normalisation
# ─────────────────────────────────────────────────────────────────────────────

# Colonnes qui seront normalisées (features d'entrée des modèles)
# La cible (target) et le ticker ne sont PAS normalisés.
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "log_return",
    "sma_10", "sma_20", "sma_50",
    "sma_10_ratio", "sma_20_ratio", "sma_50_ratio",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "volatility_20",
    "volume_norm",
]


def normalize(train_df, val_df, test_df, ticker: str, scalers_dir: Path):
    """
    Normalise les features avec MinMaxScaler.

    RÈGLE CRITIQUE : le scaler est ajusté (fit) UNIQUEMENT sur le train set.
    Ensuite on applique (transform) ce même scaler sur val et test.

    Pourquoi ? Parce que le scaler apprend les min/max des données.
    Si on l'ajuste sur tout le dataset, le modèle dispose implicitement
    d'informations sur la plage des données futures → data leakage.

    Le scaler est sauvegardé dans data/processed/scalers/<TICKER>_scaler.pkl
    pour être réutilisé à l'inférence (prédiction sur de nouvelles données).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    # fit() sur train uniquement
    scaler.fit(train_df[FEATURE_COLS])

    # transform() sur les trois splits
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[FEATURE_COLS] = scaler.transform(train_df[FEATURE_COLS])
    val_scaled[FEATURE_COLS] = scaler.transform(val_df[FEATURE_COLS])
    test_scaled[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])

    # Sauvegarder le scaler pour l'inférence future
    scalers_dir.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace(".", "_")
    scaler_path = scalers_dir / f"{safe}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"{ticker} : scaler sauvegardé → {scaler_path}")

    return train_scaled, val_scaled, test_scaled


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 : Séquences fenêtres glissantes
# ─────────────────────────────────────────────────────────────────────────────

def create_sequences(df: pd.DataFrame, window: int, ticker: str):
    """
    Transforme le DataFrame en séquences (X, y) pour LSTM/TCN/TFT.

    PRINCIPE DU FENÊTRAGE GLISSANT :
    ─────────────────────────────────
    Pour chaque position t dans le temps :
      X[t] = features des jours [t-window, ..., t-1]  (shape: window × n_features)
      y[t] = target du jour t (direction J+1 : 0 ou 1)

    Exemple avec window=3 et 6 jours de données :
      Jour :   1    2    3    4    5    6
      X[3] = [j1, j2, j3]  →  y[3] = direction_j4
      X[4] = [j2, j3, j4]  →  y[4] = direction_j5
      X[5] = [j3, j4, j5]  →  y[5] = direction_j6

    POURQUOI DES SÉQUENCES POUR LSTM/TCN et pas pour XGBoost ?
    ────────────────────────────────────────────────────────────
    LSTM et TCN sont des modèles séquentiels : ils traitent les données
    dans l'ordre temporel et apprennent les dépendances entre les pas de temps.
    Ils ont besoin d'un tenseur 3D : (batch, temps, features).

    XGBoost est un modèle tabulaire : il ne comprend pas l'ordre temporel.
    On lui donnera les features "aplaties" (un vecteur 1D par exemple).

    Retourne :
      X : numpy array de shape (N, window, n_features)
      y : numpy array de shape (N,)
    """
    features = df[FEATURE_COLS].values   # shape (T, n_features)
    targets = df["target"].values        # shape (T,)

    X, y = [], []
    for i in range(window, len(features)):
        # Fenêtre de window jours avant la position i
        X.append(features[i - window: i])
        # Cible : direction du jour i (qui est J+1 par rapport au dernier jour de X)
        y.append(targets[i])

    X = np.array(X, dtype=np.float32)   # float32 requis par PyTorch
    y = np.array(y, dtype=np.int64)     # int64 requis par CrossEntropyLoss

    logger.info(f"{ticker} : séquences créées — X{X.shape}, y{y.shape}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 : Sauvegarde
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(
    train_df, val_df, test_df,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    ticker: str,
    output_dir: Path,
):
    """
    Sauvegarde les données prétraitées dans data/processed/<TICKER>/ :

    Structure :
      data/processed/MC_PA/
      ├── train.parquet  ─ DataFrame train normalisé (pour XGBoost et analyse)
      ├── val.parquet
      ├── test.parquet
      ├── train.csv      ─ même chose en CSV (lisibilité / prof)
      ├── val.csv
      ├── test.csv
      ├── X_train.npy    ─ séquences 3D (N, window, features) pour LSTM/TCN/TFT
      ├── y_train.npy
      ├── X_val.npy
      ├── y_val.npy
      ├── X_test.npy
      └── y_test.npy
    """
    safe = ticker.replace(".", "_")
    ticker_dir = output_dir / safe
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # DataFrames en Parquet + CSV
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.to_parquet(ticker_dir / f"{name}.parquet", engine="pyarrow")
        df.to_csv(ticker_dir / f"{name}.csv", float_format="%.6f")
        logger.info(f"{ticker} : {name}.parquet + {name}.csv sauvegardés ({len(df)} lignes)")

    # Séquences numpy
    for name, arr in [
        ("X_train", X_train), ("y_train", y_train),
        ("X_val", X_val), ("y_val", y_val),
        ("X_test", X_test), ("y_test", y_test),
    ]:
        np.save(ticker_dir / f"{name}.npy", arr)

    logger.info(
        f"{ticker} : séquences numpy sauvegardées — "
        f"X_train{X_train.shape}, X_val{X_val.shape}, X_test{X_test.shape}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def process_ticker(ticker: str, raw_dir: Path, output_dir: Path,
                   window: int, scalers_dir: Path):
    """Pipeline complet pour un actif."""
    safe = ticker.replace(".", "_")
    raw_path = raw_dir / f"{safe}.parquet"

    if not raw_path.exists():
        logger.error(f"{ticker} : fichier brut introuvable ({raw_path})")
        return False

    logger.info(f"\n{'='*55}\nTraitement {ticker}\n{'='*55}")

    # 1. Charger
    df = pd.read_parquet(raw_path)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df = df.sort_index()

    # 2. Calculer les features
    df = compute_features(df, ticker)

    # 3. Walk-forward split
    train_df, val_df, test_df = walk_forward_split(df, ticker)

    # 4. Normaliser (fit sur train uniquement)
    train_sc, val_sc, test_sc = normalize(train_df, val_df, test_df, ticker, scalers_dir)

    # 5. Créer les séquences
    X_train, y_train = create_sequences(train_sc, window, ticker)
    X_val, y_val = create_sequences(val_sc, window, ticker)
    X_test, y_test = create_sequences(test_sc, window, ticker)

    # 6. Sauvegarder
    save_processed(
        train_sc, val_sc, test_sc,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        ticker, output_dir,
    )
    return True


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    scalers_dir = output_dir / "scalers"
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for ticker in args.tickers:
        ok = process_ticker(ticker, raw_dir, output_dir, args.window, scalers_dir)
        if not ok:
            failed.append(ticker)

    print("\n" + "=" * 55)
    print("  RESUME PRETRAITEMENT")
    print("=" * 55)
    for ticker in args.tickers:
        status = "[FAIL]" if ticker in failed else "[OK]"
        print(f"  {status}  {ticker}")
    print(f"\n  Donnees dans : {output_dir.resolve()}")
    print(f"  Scalers dans : {scalers_dir.resolve()}")
    print("=" * 55)

    if failed:
        logger.error(f"Tickers echoues : {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
