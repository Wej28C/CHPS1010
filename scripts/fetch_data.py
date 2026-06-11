"""
fetch_data.py — Collecte des données OHLCV via yfinance.

Sauvegarde chaque actif en :
  - CSV  → data/raw/<TICKER>.csv   (pour partager avec le prof / visualiser)
  - Parquet → data/raw/<TICKER>.parquet  (pour les modèles, plus rapide à charger)

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --tickers MC.PA RMS.PA --start 2020-01-01
"""

import argparse
import json
import logging
import sys
import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Logger ───────────────────────────────────────────────────────────────────
# On utilise logging plutôt que print() pour contrôler la verbosité
# et écrire dans un fichier si besoin (--log-file en option future).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["MC.PA", "CFR.SW", "RMS.PA", "BRBY.L", "OR.PA"]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-01-01"
RAW_DIR = Path("data/raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collecte OHLCV via yfinance")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output-dir", default=str(RAW_DIR))
    return parser.parse_args()


def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Télécharge les données OHLCV pour un seul ticker.

    auto_adjust=True : les prix sont corrigés des splits et dividendes.
    C'est indispensable pour que les modèles ne voient pas de sauts
    artificiels dans les prix (ex : LVMH a splitté 1:10 en 2021).
    """
    logger.info(f"Téléchargement {ticker} ({start} → {end})...")

    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as e:
        logger.error(f"{ticker} : échec téléchargement — {e}")
        return pd.DataFrame()

    if df.empty:
        logger.warning(f"{ticker} : aucune donnée retournée")
        return pd.DataFrame()

    # yfinance retourne parfois un MultiIndex (Price, Ticker) quand
    # auto_adjust=True. On l'aplatit pour avoir des colonnes simples.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.error(f"{ticker} : colonnes manquantes {missing}")
        return pd.DataFrame()

    df = df[cols].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df["Ticker"] = ticker

    logger.info(f"{ticker} : {len(df)} lignes récupérées")
    return df


def validate(df: pd.DataFrame, ticker: str) -> bool:
    """
    Contrôles qualité basiques avant de sauvegarder.

    Seuil NaN à 5% : au-delà, les données sont trop lacunaires
    pour être fiables dans un modèle de séries temporelles.
    """
    if df.empty:
        logger.error(f"{ticker} : DataFrame vide")
        return False

    nan_rate = df[["Open", "High", "Low", "Close", "Volume"]].isna().mean()
    bad_cols = nan_rate[nan_rate > 0.05]
    if not bad_cols.empty:
        logger.warning(f"{ticker} : trop de NaN dans {bad_cols.to_dict()}")
        return False

    logger.info(f"{ticker} : validation OK")
    return True


def clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Nettoyage des NaN et doublons.

    Forward fill (ffill) = propager la dernière valeur connue.
    C'est la convention standard en finance : si une bourse est fermée
    un jour (férié local), on considère que le prix est resté le même.
    Ex : la bourse de Paris est fermée le 14 juillet, pas Londres.
    """
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="all")
    df[["Open", "High", "Low", "Close"]] = (
        df[["Open", "High", "Low", "Close"]].ffill().bfill()
    )
    df["Volume"] = df["Volume"].fillna(0).astype("int64")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    logger.info(f"{ticker} : nettoyage terminé ({len(df)} lignes finales)")
    return df


def save(df: pd.DataFrame, output_dir: Path, ticker: str) -> dict:
    """
    Sauvegarde en CSV et Parquet.

    CSV  → lisible par Excel, OpenOffice, et partageable avec le prof.
    Parquet → format binaire compressé utilisé par les scripts de ML.
              Environ 10x plus petit qu'un CSV et 5x plus rapide à lire.

    Le nom de fichier remplace '.' par '_' (MC.PA → MC_PA) pour
    éviter les problèmes sur certains systèmes de fichiers.
    """
    safe = ticker.replace(".", "_")

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = output_dir / f"{safe}.csv"
    # index=True pour conserver la colonne Date dans le CSV
    # float_format="%.4f" pour 4 décimales (suffisant pour les prix boursiers)
    df.to_csv(csv_path, index=True, float_format="%.4f", encoding="utf-8")
    csv_kb = csv_path.stat().st_size / 1024

    # ── Parquet ──────────────────────────────────────────────────────────────
    parquet_path = output_dir / f"{safe}.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=True)
    parquet_kb = parquet_path.stat().st_size / 1024

    logger.info(
        f"{ticker} : CSV {csv_kb:.0f} KB → {csv_path.name} | "
        f"Parquet {parquet_kb:.0f} KB → {parquet_path.name}"
    )

    return {
        "status": "OK",
        "rows": len(df),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "csv": str(csv_path),
        "parquet": str(parquet_path),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    failed = []

    for ticker in args.tickers:
        df = fetch_ticker(ticker, args.start, args.end)

        if not validate(df, ticker):
            failed.append(ticker)
            results[ticker] = {"status": "FAILED"}
            continue

        df = clean(df, ticker)
        results[ticker] = save(df, output_dir, ticker)

    # ── Métadonnées ──────────────────────────────────────────────────────────
    # Traçabilité : on enregistre quand, avec quelle version de yfinance,
    # et quels paramètres ont été utilisés pour cette collecte.
    meta = {
        "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "yfinance_version": yf.__version__,
        "parameters": {
            "tickers": args.tickers,
            "start": args.start,
            "end": args.end,
        },
        "results": results,
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ── Résumé console ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESUME COLLECTE OHLCV")
    print("=" * 60)
    for ticker, info in results.items():
        if info["status"] == "OK":
            print(f"  [OK]   {ticker:<10}  {info['rows']:>5} lignes  "
                  f"{info['start']} -> {info['end']}")
        else:
            print(f"  [FAIL] {ticker:<10}  ECHEC")
    print("=" * 60)
    print(f"  Fichiers dans : {output_dir.resolve()}")
    print(f"  CSV     : partageables avec le prof")
    print(f"  Parquet : utilises par les scripts ML")
    print("=" * 60)

    if failed:
        logger.error(f"Tickers échoués : {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
