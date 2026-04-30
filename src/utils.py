"""
utils.py — Shared utilities: logging, memory optimisation, paths, constants.
"""

import gc
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Silence noisy libraries ───────────────────────────────────────────────────
warnings.filterwarnings("ignore")

# ── Project-level paths (resolved relative to this file) ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
FIGURES_DIR  = REPORTS_DIR / "figures"

for _d in [DATA_RAW, DATA_PROC, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED    = 42
N_FOLDS = 5

# ── Matplotlib style constants ────────────────────────────────────────────────
PALETTE = ["#1D9E75", "#378ADD", "#E85D24", "#7F77DD", "#BA7517"]

PLOT_STYLE = {
    "figure.dpi":         120,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "font.size":          11,
}


# ── Logger factory ────────────────────────────────────────────────────────────
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Memory optimisation ───────────────────────────────────────────────────────
def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to the smallest safe dtype."""
    logger = get_logger(__name__)
    start_mb = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        s = df[col]
        col_type = s.dtype

        # Only downcast true numeric columns. This avoids NumPy 2 comparison
        # errors on string/extension dtypes that may appear in mixed CSV data.
        if not (
            pd.api.types.is_integer_dtype(col_type)
            or pd.api.types.is_float_dtype(col_type)
        ):
            continue

        c_min, c_max = s.min(skipna=True), s.max(skipna=True)
        if pd.isna(c_min) or pd.isna(c_max):
            continue

        if pd.api.types.is_integer_dtype(col_type):
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                if c_min >= np.iinfo(dtype).min and c_max <= np.iinfo(dtype).max:
                    df[col] = s.astype(dtype)
                    break
        elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
            df[col] = s.astype(np.float32)

    end_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        logger.info("Memory: %.1f MB → %.1f MB (%.1f%% reduction)",
                    start_mb, end_mb, 100 * (start_mb - end_mb) / (start_mb + 1e-9))
    gc.collect()
    return df


# ── Safe CSV loader ───────────────────────────────────────────────────────────
def load_csv(filepath: Path, verbose: bool = True, **kwargs) -> pd.DataFrame:
    """Load a CSV with memory optimisation and basic logging."""
    logger = get_logger(__name__)
    logger.info("Loading %s …", filepath.name)
    df = pd.read_csv(filepath, **kwargs)
    df = reduce_mem_usage(df, verbose=verbose)
    logger.info("  shape: %s", df.shape)
    return df
