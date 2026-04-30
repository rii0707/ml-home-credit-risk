"""
data_preprocessing.py — Load all raw CSV files, reduce memory, persist to parquet.

Usage:
    python src/data_preprocessing.py
"""

from pathlib import Path

import pandas as pd

from utils import DATA_RAW, DATA_PROC, get_logger, load_csv

logger = get_logger(__name__)

# ── File manifest ─────────────────────────────────────────────────────────────
RAW_FILES = {
    "app_train":    "application_train.csv",
    "app_test":     "application_test.csv",
    "bureau":       "bureau.csv",
    "bureau_bal":   "bureau_balance.csv",
    "prev_app":     "previous_application.csv",
    "installments": "installments_payments.csv",
    "pos_cash":     "POS_CASH_balance.csv",
    "cc_balance":   "credit_card_balance.csv",
}


def load_all_raw(data_dir: Path = DATA_RAW) -> dict[str, pd.DataFrame]:
    """Load every raw CSV and return a name→DataFrame dict."""
    dfs: dict[str, pd.DataFrame] = {}
    for key, filename in RAW_FILES.items():
        path = data_dir / filename
        if not path.exists():
            logger.warning("File not found, skipping: %s", path)
            continue
        dfs[key] = load_csv(path)
    logger.info("Loaded %d / %d files.", len(dfs), len(RAW_FILES))
    return dfs


def save_processed(dfs: dict[str, pd.DataFrame], out_dir: Path = DATA_PROC) -> None:
    """Persist DataFrames as parquet for fast downstream reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in dfs.items():
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved %s → %s", name, out_path)


def load_processed(out_dir: Path = DATA_PROC) -> dict[str, pd.DataFrame]:
    """Reload previously saved parquet files."""
    dfs: dict[str, pd.DataFrame] = {}
    for key in RAW_FILES:
        path = out_dir / f"{key}.parquet"
        if path.exists():
            dfs[key] = pd.read_parquet(path)
            logger.info("Loaded processed %s  shape=%s", key, dfs[key].shape)
        else:
            logger.warning("Processed file missing: %s", path)
    return dfs


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=== Data preprocessing start ===")
    raw_dfs = load_all_raw()
    save_processed(raw_dfs)
    logger.info("=== Data preprocessing complete ===")
