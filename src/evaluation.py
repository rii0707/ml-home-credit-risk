"""
evaluation.py — Model evaluation: threshold analysis, error analysis, plots.

Usage:
    python src/evaluation.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

from utils import DATA_PROC, FIGURES_DIR, MODELS_DIR, PALETTE, PLOT_STYLE, get_logger

logger = get_logger(__name__)
plt.rcParams.update(PLOT_STYLE)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold analysis
# ─────────────────────────────────────────────────────────────────────────────

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """Return the threshold that maximises F1 (or minimises business cost)."""
    thresholds = np.arange(0.05, 0.60, 0.005)
    if metric == "f1":
        scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
                  for t in thresholds]
        best = thresholds[int(np.argmax(scores))]
    elif metric == "cost":
        FN_COST, FP_COST = 10, 1
        costs = []
        for t in thresholds:
            cm = confusion_matrix(y_true, (y_prob >= t).astype(int))
            tn, fp, fn, tp = cm.ravel()
            costs.append(FN_COST * fn + FP_COST * fp)
        best = thresholds[int(np.argmin(costs))]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    logger.info("Best %s threshold: %.4f", metric, best)
    return float(best)


# ─────────────────────────────────────────────────────────────────────────────
# Error analysis
# ─────────────────────────────────────────────────────────────────────────────

_ERROR_MAP = {(0, 0): "TN", (1, 1): "TP", (0, 1): "FP", (1, 0): "FN"}


def build_error_df(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    key_features: list[str] | None = None,
) -> pd.DataFrame:
    """Build a DataFrame labelled with TN / TP / FP / FN."""
    y_pred = (y_prob >= threshold).astype(int)
    err_df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred})
    err_df["error_type"] = [
        _ERROR_MAP[(int(t), int(p))] for t, p in zip(y_true, y_pred)
    ]
    if key_features:
        safe_feats = [f for f in key_features if f in X.columns]
        err_df = err_df.join(X[safe_feats].reset_index(drop=True))

    logger.info("Error distribution:\n%s", err_df["error_type"].value_counts().to_string())
    return err_df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_pr(
    y_true: np.ndarray,
    oof_preds: dict[str, np.ndarray],
    out_path=FIGURES_DIR / "roc_pr.png",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, (name, preds) in enumerate(oof_preds.items()):
        auc = roc_auc_score(y_true, preds)
        fpr, tpr, _ = roc_curve(y_true, preds)
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} AUC={auc:.4f}", color=PALETTE[i % len(PALETTE)])

        ap = average_precision_score(y_true, preds)
        prec, rec, _ = precision_recall_curve(y_true, preds)
        axes[1].plot(rec, prec, lw=2, label=f"{name} AP={ap:.4f}", color=PALETTE[i % len(PALETTE)])

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC curve (OOF)")
    axes[0].legend()

    axes[1].axhline(y_true.mean(), color="k", linestyle="--", lw=0.8,
                    label=f"Baseline ({y_true.mean():.3f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve (OOF)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC/PR plot → %s", out_path)


def plot_feature_importance(
    fi_df: pd.DataFrame,
    top_n: int = 30,
    out_path=FIGURES_DIR / "feature_importance.png",
) -> None:
    top = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=PALETTE[4])
    ax.set(xlabel="Mean importance (split count)",
           title=f"Top {top_n} feature importances — LightGBM")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved feature importance plot → %s", out_path)


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    best_thresh: float,
    out_path=FIGURES_DIR / "threshold_analysis.png",
) -> None:
    thresholds = np.arange(0.05, 0.60, 0.005)
    f1_scores  = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
                  for t in thresholds]

    FN_COST, FP_COST = 10, 1
    costs = []
    for t in thresholds:
        cm = confusion_matrix(y_true, (y_prob >= t).astype(int))
        tn, fp, fn, tp = cm.ravel()
        costs.append(FN_COST * fn + FP_COST * fp)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(thresholds, f1_scores, color=PALETTE[0], lw=2)
    axes[0].axvline(best_thresh, color=PALETTE[2], linestyle="--",
                    label=f"Best = {best_thresh:.3f}")
    axes[0].set(xlabel="Threshold", ylabel="F1", title="F1 vs threshold")
    axes[0].legend()

    best_cost_t = thresholds[int(np.argmin(costs))]
    axes[1].plot(thresholds, costs, color=PALETTE[3], lw=2)
    axes[1].axvline(best_cost_t, color=PALETTE[2], linestyle="--",
                    label=f"Min cost = {best_cost_t:.3f}")
    axes[1].set(xlabel="Threshold",
                ylabel=f"Cost (FN={FN_COST}×, FP={FP_COST}×)",
                title="Business cost vs threshold")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved threshold analysis → %s", out_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    out_path=FIGURES_DIR / "confusion_matrix.png",
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No default", "Default"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion matrix (threshold={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix → %s", out_path)


def print_classification_report(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    logger.info("Classification report (threshold=%.4f):\n%s",
                threshold,
                classification_report(y_true, y_pred,
                                      target_names=["No default", "Default"]))


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(results: dict, train_df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Generate all evaluation artefacts from a `train_all()` results dict."""
    TARGET = "TARGET"
    y = train_df[TARGET].values

    oof_lgbm = results["oof_preds"]["lgbm"]
    oof_lr   = results["oof_preds"]["lr"]
    oof_rf   = results["oof_preds"]["rf"]
    oof_ens  = results["oof_ensemble"]
    fi_df    = results["feature_importance"]

    # Plots
    plot_roc_pr(y, {"LightGBM": oof_lgbm, "LR": oof_lr, "RF": oof_rf, "Ensemble": oof_ens})
    plot_feature_importance(fi_df)

    best_thresh = find_best_threshold(y, oof_lgbm, metric="f1")
    plot_threshold_analysis(y, oof_lgbm, best_thresh)
    plot_confusion_matrix(y, oof_lgbm, best_thresh)
    print_classification_report(y, oof_lgbm, best_thresh)

    # Error analysis (key features)
    key_feats = [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "CREDIT_INCOME_RATIO",
    ]
    X = train_df[feature_cols]
    err_df = build_error_df(X, y, oof_lgbm, best_thresh, key_feats)
    logger.info("Mean feature values by error type:\n%s",
                err_df.groupby("error_type")[key_feats].mean().T.round(2).to_string())


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pickle
    train_df = pd.read_parquet(DATA_PROC / "train_features.parquet")
    TARGET, ID = "TARGET", "SK_ID_CURR"
    feature_cols = [c for c in train_df.columns if c not in [TARGET, ID]]

    with open(MODELS_DIR / "training_results.pkl", "rb") as f:
        results = pickle.load(f)

    evaluate(results, train_df, feature_cols)
