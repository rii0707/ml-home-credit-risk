"""
model_training.py — Baseline models, LightGBM CV, Optuna tuning, stacking ensemble.

Usage:
    python src/model_training.py
"""

import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import lightgbm as lgb

from utils import DATA_PROC, MODELS_DIR, SEED, N_FOLDS, get_logger

logger = get_logger(__name__)
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "30"))
CV_FOLDS_OVERRIDE = os.getenv("CV_FOLDS")


# ─────────────────────────────────────────────────────────────────────────────
# Default LightGBM hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "objective":         "binary",
    "metric":            "auc",
    "boosting_type":     "gbdt",
    "num_leaves":        64,
    "max_depth":         -1,
    "learning_rate":     0.05,
    "n_estimators":      1000,
    "min_child_samples": 20,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "n_jobs":            -1,
    "random_state":      SEED,
    "verbose":           -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn pipeline builders
# ─────────────────────────────────────────────────────────────────────────────

def build_lr_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                make_column_selector(dtype_exclude=np.number),
            ),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model",   LogisticRegression(
            max_iter=1000, C=0.05,
            class_weight="balanced",
            solver="saga", n_jobs=-1, random_state=SEED,
        )),
    ])


def build_rf_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                make_column_selector(dtype_exclude=np.number),
            ),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model",   RandomForestClassifier(
            n_estimators=300, max_depth=10,
            min_samples_leaf=50, n_jobs=-1,
            class_weight="balanced", random_state=SEED,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM input prep
# ─────────────────────────────────────────────────────────────────────────────

def prepare_lgb_inputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert non-numeric columns to consistent integer codes across train/test.
    LightGBM sklearn interface in this environment expects int/float/bool dtypes.
    """
    X_train_lgb = X_train.copy()
    X_test_lgb = X_test.copy()

    cat_cols = [c for c in X_train_lgb.columns if not pd.api.types.is_numeric_dtype(X_train_lgb[c])]
    for col in cat_cols:
        # Use shared categories so train/test map to the same integer codes.
        combined = pd.concat([X_train_lgb[col], X_test_lgb[col]], axis=0).astype("string")
        categories = pd.Index(combined.dropna().unique())

        train_cat = pd.Categorical(X_train_lgb[col].astype("string"), categories=categories)
        test_cat = pd.Categorical(X_test_lgb[col].astype("string"), categories=categories)

        X_train_lgb[col] = pd.Series(train_cat.codes, index=X_train_lgb.index).astype(np.int32)
        X_test_lgb[col] = pd.Series(test_cat.codes, index=X_test_lgb.index).astype(np.int32)

    return X_train_lgb, X_test_lgb


# ─────────────────────────────────────────────────────────────────────────────
# Generic OOF CV helper
# ─────────────────────────────────────────────────────────────────────────────

def run_oof_cv(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    skf: StratifiedKFold,
    model_name: str = "model",
    lgb_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Run stratified K-fold OOF CV for any sklearn-compatible model or LightGBM.

    Returns
    -------
    oof_preds   : shape (n_train,)
    test_preds  : shape (n_test,)  — mean of fold predictions
    fold_scores : list of per-fold AUC values
    """
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores: list[float] = []

    logger.info("%s — %d-fold CV", model_name, skf.n_splits)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
        y_tr,  y_val  = y[tr_idx],        y[val_idx]

        t0 = time.time()

        if lgb_params is not None:
            # LightGBM branch with early stopping
            m = lgb.LGBMClassifier(**lgb_params)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
        else:
            m = model
            m.fit(X_tr, y_tr)

        oof_preds[val_idx] = m.predict_proba(X_val)[:, 1]
        test_preds        += m.predict_proba(X_test)[:, 1] / skf.n_splits

        score = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(score)
        logger.info("  fold %d: AUC=%.5f  (%.0fs)", fold, score, time.time() - t0)

    oof_auc = roc_auc_score(y, oof_preds)
    logger.info("%s — OOF AUC: %.5f  ±  %.5f",
                model_name, np.mean(fold_scores), np.std(fold_scores))
    logger.info("%s — Overall OOF AUC: %.5f", model_name, oof_auc)
    return oof_preds, test_preds, fold_scores


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM feature importance
# ─────────────────────────────────────────────────────────────────────────────

def get_lgb_feature_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    skf: StratifiedKFold,
    lgb_params: dict,
) -> pd.DataFrame:
    """Return a DataFrame of mean feature importances across folds."""
    fi_list = []
    for tr_idx, val_idx in skf.split(X, y):
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
        )
        fi_list.append(m.feature_importances_)

    return (
        pd.DataFrame({"feature": X.columns, "importance": np.mean(fi_list, axis=0)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optuna tuning
# ─────────────────────────────────────────────────────────────────────────────

def tune_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    skf: StratifiedKFold,
    n_trials: int = 30,
) -> dict:
    """
    Run Optuna hyperparameter search.  Returns best_params merged with
    DEFAULT_LGB_PARAMS.  Falls back to defaults if Optuna is unavailable.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed. Using default LGB params.")
        return DEFAULT_LGB_PARAMS

    pos_weight = float((y == 0).sum() / (y == 1).sum())

    def objective(trial):
        params = {
            **DEFAULT_LGB_PARAMS,
            "scale_pos_weight": pos_weight,
            "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
        }
        scores = []
        for tr_idx, val_idx in skf.split(X, y):
            m = lgb.LGBMClassifier(**params)
            m.fit(
                X.iloc[tr_idx], y[tr_idx],
                eval_set=[(X.iloc[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            scores.append(roc_auc_score(y[val_idx], m.predict_proba(X.iloc[val_idx])[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name="lgbm_hc")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Optuna best AUC: %.5f", study.best_value)
    for k, v in study.best_params.items():
        logger.info("  %s: %s", k, v)

    return {**DEFAULT_LGB_PARAMS, "scale_pos_weight": pos_weight, **study.best_params}


# ─────────────────────────────────────────────────────────────────────────────
# Weighted ensemble
# ─────────────────────────────────────────────────────────────────────────────

def find_best_blend_weights(
    oof_preds: dict[str, np.ndarray],
    y: np.ndarray,
) -> dict[str, float]:
    """
    Grid-search blend weights over OOF predictions.

    Parameters
    ----------
    oof_preds : {"lgbm": ..., "lr": ..., "rf": ...}
    y         : ground-truth labels
    """
    names  = list(oof_preds.keys())
    arrays = list(oof_preds.values())
    best_w, best_auc = None, 0.0

    # 3-model grid search (works for arbitrary n with recursion, but kept simple here)
    for w0 in np.arange(0.0, 1.05, 0.05):
        for w1 in np.arange(0.0, 1.05 - w0, 0.05):
            w2 = round(1 - w0 - w1, 5)
            if w2 < 0:
                continue
            blend = w0 * arrays[0] + w1 * arrays[1] + w2 * arrays[2]
            auc   = roc_auc_score(y, blend)
            if auc > best_auc:
                best_auc = auc
                best_w   = [w0, w1, w2]

    weights = dict(zip(names, best_w))
    logger.info("Best ensemble weights: %s  →  AUC=%.5f", weights, best_auc)
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Save / load artefacts
# ─────────────────────────────────────────────────────────────────────────────

def save_model(obj: Any, name: str, out_dir: Path = MODELS_DIR) -> Path:
    path = out_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved %s → %s", name, path)
    return path


def load_model(name: str, model_dir: Path = MODELS_DIR) -> Any:
    path = model_dir / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrated training run
# ─────────────────────────────────────────────────────────────────────────────

def train_all(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    n_optuna_trials: int = OPTUNA_TRIALS,
) -> dict:
    """
    Full training pipeline:
      1. Baseline LR + RF
      2. LightGBM with Optuna tuning
      3. Weighted average ensemble

    Returns a results dict with OOF preds, test preds, weights, and AUC scores.
    """
    TARGET, ID = "TARGET", "SK_ID_CURR"

    X      = train_df[feature_cols]
    y      = train_df[TARGET].values
    X_test = test_df[feature_cols]
    X_lgb, X_test_lgb = prepare_lgb_inputs(X, X_test)

    n_splits = int(CV_FOLDS_OVERRIDE) if CV_FOLDS_OVERRIDE else (3 if FAST_MODE else N_FOLDS)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pos_weight = float((y == 0).sum() / (y == 1).sum())

    # ── Baselines ─────────────────────────────────────────────────────────────
    oof_lr, test_lr, _ = run_oof_cv(build_lr_pipeline(), X, y, X_test, skf, "LR")
    oof_rf, test_rf, _ = run_oof_cv(build_rf_pipeline(), X, y, X_test, skf, "RF")

    # ── LightGBM (tuned) ──────────────────────────────────────────────────────
    best_params = tune_lgbm(X_lgb, y, skf, n_trials=n_optuna_trials)
    best_params["scale_pos_weight"] = pos_weight

    oof_lgbm, test_lgbm, lgbm_folds = run_oof_cv(
        None, X_lgb, y, X_test_lgb, skf, "LightGBM", lgb_params=best_params
    )
    fi_df = get_lgb_feature_importance(X_lgb, y, skf, best_params)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    oof_preds_map = {"lgbm": oof_lgbm, "lr": oof_lr, "rf": oof_rf}
    test_preds_map = {"lgbm": test_lgbm, "lr": test_lr, "rf": test_rf}
    weights = find_best_blend_weights(oof_preds_map, y)

    oof_ensemble  = sum(w * oof_preds_map[k]  for k, w in weights.items())
    test_ensemble = sum(w * test_preds_map[k] for k, w in weights.items())

    # ── Summary ───────────────────────────────────────────────────────────────
    results = {
        "oof_preds":     oof_preds_map,
        "test_preds":    test_preds_map,
        "oof_ensemble":  oof_ensemble,
        "test_ensemble": test_ensemble,
        "blend_weights": weights,
        "best_lgb_params": best_params,
        "feature_importance": fi_df,
        "auc_scores": {
            "LR":       roc_auc_score(y, oof_lr),
            "RF":       roc_auc_score(y, oof_rf),
            "LightGBM": roc_auc_score(y, oof_lgbm),
            "Ensemble": roc_auc_score(y, oof_ensemble),
        },
    }

    save_model(results, "training_results")
    logger.info("=== AUC summary ===")
    for name, auc in results["auc_scores"].items():
        logger.info("  %-12s %.5f", name, auc)

    return results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_df = pd.read_parquet(DATA_PROC / "train_features.parquet")
    test_df  = pd.read_parquet(DATA_PROC / "test_features.parquet")
    TARGET, ID = "TARGET", "SK_ID_CURR"
    feature_cols = [c for c in train_df.columns if c not in [TARGET, ID]]

    folds_cfg = CV_FOLDS_OVERRIDE if CV_FOLDS_OVERRIDE else ("3" if FAST_MODE else str(N_FOLDS))
    logger.info(
        "Run config: FAST_MODE=%s, folds=%s, optuna_trials=%d",
        FAST_MODE, folds_cfg, OPTUNA_TRIALS
    )
    train_all(train_df, test_df, feature_cols, n_optuna_trials=OPTUNA_TRIALS)
