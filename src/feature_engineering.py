"""
feature_engineering.py — All feature-engineering functions.

Each public function is pure (no side effects) and returns a DataFrame.
The `build_feature_matrix()` entry point merges everything.

Usage:
    python src/feature_engineering.py
"""

import numpy as np
import pandas as pd

from utils import DATA_PROC, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Application table
# ─────────────────────────────────────────────────────────────────────────────

def engineer_application(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and create ratio / aggregate features from application_{train|test}."""
    df = df.copy()

    # Fix DAYS_EMPLOYED sentinel (365243 = unemployed flag)
    df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(np.int8)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Days → human-readable years (positive)
    df["AGE_YEARS"]      = -df["DAYS_BIRTH"]             / 365.25
    df["EMPLOYED_YEARS"] =  df["DAYS_EMPLOYED"].abs()    / 365.25

    # Credit / income ratios
    df["CREDIT_INCOME_RATIO"]  = df["AMT_CREDIT"]      / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"]     / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"]     / (df["AMT_CREDIT"]       + 1)
    df["GOODS_CREDIT_RATIO"]   = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"]       + 1)
    df["INCOME_PER_PERSON"]    = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["EMPLOYED_BIRTH_RATIO"] = df["DAYS_EMPLOYED"]   /  (df["DAYS_BIRTH"]      + 1e-6)
    df["PAYMENT_RATE"]         = df["AMT_ANNUITY"]     / (df["AMT_CREDIT"]       + 1)

    # External source aggregates
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_STD"]  = ext.std(axis=1)
    df["EXT_SOURCE_MIN"]  = ext.min(axis=1)
    df["EXT_SOURCE_MAX"]  = ext.max(axis=1)
    df["EXT_SOURCE_PROD"] = ext.prod(axis=1)

    # Document flags
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT")]
    df["DOCUMENTS_TOTAL"] = df[doc_cols].sum(axis=1)

    # Credit bureau enquiries
    enq_cols = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU")]
    df["ENQUIRIES_TOTAL"] = df[enq_cols].sum(axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bureau + bureau_balance
# ─────────────────────────────────────────────────────────────────────────────

def agg_bureau(bureau_df: pd.DataFrame, bb_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau and bureau_balance tables to one row per SK_ID_CURR."""
    bb_df = bb_df.copy()
    bb_df["STATUS_0"] = (bb_df["STATUS"] == "0").astype(np.int8)
    bb_df["STATUS_X"] = (bb_df["STATUS"] == "X").astype(np.int8)
    bb_df["STATUS_C"] = (bb_df["STATUS"] == "C").astype(np.int8)

    bb_agg = bb_df.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_CNT    =("MONTHS_BALANCE", "count"),
        BB_MONTHS_MIN    =("MONTHS_BALANCE", "min"),
        BB_STATUS_0_MEAN =("STATUS_0",       "mean"),
        BB_STATUS_C_MEAN =("STATUS_C",       "mean"),
        BB_STATUS_X_MEAN =("STATUS_X",       "mean"),
    ).reset_index()

    bureau_df = bureau_df.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau_df["CREDIT_ACTIVE_BINARY"] = (bureau_df["CREDIT_ACTIVE"] == "Active").astype(np.int8)

    num_aggs = {
        "DAYS_CREDIT":            ["max", "min", "mean", "std"],
        "CREDIT_DAY_OVERDUE":     ["max", "mean"],
        "DAYS_CREDIT_ENDDATE":    ["max", "min", "mean"],
        "DAYS_ENDDATE_FACT":      ["max", "min", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
        "CNT_CREDIT_PROLONG":     ["sum", "mean"],
        "AMT_CREDIT_SUM":         ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT":    ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_SUM_LIMIT":   ["max", "mean"],
        "DAYS_CREDIT_UPDATE":     ["max", "min", "mean"],
        "AMT_ANNUITY":            ["max", "mean"],
        "CREDIT_ACTIVE_BINARY":   ["sum", "mean"],
        "BB_MONTHS_CNT":          ["max", "mean"],
        "BB_STATUS_0_MEAN":       ["mean"],
        "BB_STATUS_C_MEAN":       ["mean"],
        "BB_STATUS_X_MEAN":       ["mean"],
    }
    cat_aggs = {
        "CREDIT_ACTIVE": ["nunique"],
        "CREDIT_TYPE":   ["nunique"],
    }

    bureau_agg = bureau_df.groupby("SK_ID_CURR").agg({**num_aggs, **cat_aggs})
    bureau_agg.columns = ["BUREAU_" + "_".join(c).upper() for c in bureau_agg.columns]
    bureau_agg["BUREAU_LOAN_COUNT"] = bureau_df.groupby("SK_ID_CURR").size()
    return bureau_agg.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Previous applications
# ─────────────────────────────────────────────────────────────────────────────

_SENTINEL_COLS = [
    "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE", "DAYS_TERMINATION",
]


def agg_previous(prev_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous_application to one row per SK_ID_CURR."""
    prev_df = prev_df.copy()

    for col in _SENTINEL_COLS:
        if col in prev_df.columns:
            prev_df[col] = prev_df[col].replace(365243, np.nan)

    prev_df["APP_CREDIT_PERC"]   = prev_df["AMT_APPLICATION"] / (prev_df["AMT_CREDIT"] + 1)
    prev_df["CREDIT_GOODS_DIFF"] = prev_df["AMT_CREDIT"]      -  prev_df["AMT_GOODS_PRICE"]

    num_aggs = {
        "AMT_ANNUITY":               ["max", "mean", "min"],
        "AMT_APPLICATION":           ["max", "mean", "min"],
        "AMT_CREDIT":                ["max", "mean", "sum"],
        "AMT_DOWN_PAYMENT":          ["max", "mean"],
        "AMT_GOODS_PRICE":           ["max", "mean"],
        "RATE_DOWN_PAYMENT":         ["max", "mean", "min"],
        "DAYS_DECISION":             ["max", "mean", "min"],
        "CNT_PAYMENT":               ["mean", "sum"],
        "APP_CREDIT_PERC":           ["max", "mean", "min", "std"],
        "CREDIT_GOODS_DIFF":         ["max", "mean"],
        "DAYS_FIRST_DUE":            ["min", "mean"],
        "DAYS_LAST_DUE_1ST_VERSION": ["max", "mean"],
        "DAYS_LAST_DUE":             ["max", "mean"],
        "DAYS_TERMINATION":          ["max", "mean"],
        "NFLAG_INSURED_ON_APPROVAL": ["mean"],
    }
    cat_aggs = {
        "NAME_CONTRACT_TYPE":         ["nunique"],
        "NAME_CONTRACT_STATUS":       ["nunique"],
        "NAME_YIELD_GROUP":           ["nunique"],
        "WEEKDAY_APPR_PROCESS_START": ["nunique"],
    }

    prev_agg = prev_df.groupby("SK_ID_CURR").agg({**num_aggs, **cat_aggs})
    prev_agg.columns = ["PREV_" + "_".join(c).upper() for c in prev_agg.columns]
    prev_agg["PREV_LOAN_COUNT"] = prev_df.groupby("SK_ID_CURR").size()

    approved = prev_df[prev_df["NAME_CONTRACT_STATUS"] == "Approved"]
    refused  = prev_df[prev_df["NAME_CONTRACT_STATUS"] == "Refused"]
    prev_agg["PREV_APPROVED_COUNT"] = approved.groupby("SK_ID_CURR").size()
    prev_agg["PREV_REFUSED_COUNT"]  = refused.groupby("SK_ID_CURR").size()
    prev_agg["PREV_APPROVAL_RATE"]  = (
        prev_agg["PREV_APPROVED_COUNT"] / (prev_agg["PREV_LOAN_COUNT"] + 1)
    )
    return prev_agg.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Installments payments
# ─────────────────────────────────────────────────────────────────────────────

def agg_installments(inst_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments_payments to one row per SK_ID_CURR."""
    inst_df = inst_df.copy()
    inst_df["PAYMENT_DIFF"]      = inst_df["AMT_INSTALMENT"] - inst_df["AMT_PAYMENT"]
    inst_df["PAYMENT_RATIO"]     = inst_df["AMT_PAYMENT"]    / (inst_df["AMT_INSTALMENT"] + 1)
    inst_df["PAYMENT_DELAY"]     = inst_df["DAYS_ENTRY_PAYMENT"] - inst_df["DAYS_INSTALMENT"]
    inst_df["PAYMENT_DELAY_POS"] = inst_df["PAYMENT_DELAY"].clip(lower=0)
    inst_df["DPD"]               = inst_df["PAYMENT_DELAY"].apply(lambda x: max(x, 0))

    agg = inst_df.groupby("SK_ID_CURR").agg(
        INSTAL_COUNT              =("SK_ID_PREV",      "count"),
        INSTAL_AMT_PAYMENT_SUM    =("AMT_PAYMENT",     "sum"),
        INSTAL_AMT_PAYMENT_MEAN   =("AMT_PAYMENT",     "mean"),
        INSTAL_AMT_INSTALMENT_MAX =("AMT_INSTALMENT",  "max"),
        INSTAL_PAYMENT_DIFF_MEAN  =("PAYMENT_DIFF",    "mean"),
        INSTAL_PAYMENT_DIFF_MAX   =("PAYMENT_DIFF",    "max"),
        INSTAL_PAYMENT_DIFF_STD   =("PAYMENT_DIFF",    "std"),
        INSTAL_PAYMENT_RATIO_MEAN =("PAYMENT_RATIO",   "mean"),
        INSTAL_PAYMENT_DELAY_MEAN =("PAYMENT_DELAY",   "mean"),
        INSTAL_PAYMENT_DELAY_MAX  =("PAYMENT_DELAY",   "max"),
        INSTAL_DPD_MEAN           =("DPD",             "mean"),
        INSTAL_DPD_MAX            =("DPD",             "max"),
        INSTAL_LATE_COUNT         =("PAYMENT_DELAY_POS", "sum"),
    ).reset_index()
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 5. POS CASH balance
# ─────────────────────────────────────────────────────────────────────────────

def agg_pos_cash(pos_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance to one row per SK_ID_CURR."""
    pos_df = pos_df.copy()
    pos_df["DPD_BINARY"] = (pos_df["SK_DPD"] > 0).astype(np.int8)

    agg = pos_df.groupby("SK_ID_CURR").agg(
        POS_COUNT              =("SK_ID_PREV",              "count"),
        POS_MONTHS_MIN         =("MONTHS_BALANCE",          "min"),
        POS_CNT_INSTALMENT_MAX =("CNT_INSTALMENT",          "max"),
        POS_CNT_FUTURE_MEAN    =("CNT_INSTALMENT_FUTURE",   "mean"),
        POS_SK_DPD_MAX         =("SK_DPD",                  "max"),
        POS_SK_DPD_MEAN        =("SK_DPD",                  "mean"),
        POS_SK_DPD_DEF_MAX     =("SK_DPD_DEF",              "max"),
        POS_DPD_BINARY_MEAN    =("DPD_BINARY",              "mean"),
        POS_DPD_BINARY_SUM     =("DPD_BINARY",              "sum"),
    ).reset_index()
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 6. Credit card balance
# ─────────────────────────────────────────────────────────────────────────────

def agg_credit_card(cc_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit_card_balance to one row per SK_ID_CURR."""
    cc_df = cc_df.copy()
    cc_df["DRAWING_RATIO"] = cc_df["AMT_DRAWINGS_CURRENT"]        / (cc_df["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc_df["BALANCE_RATIO"] = cc_df["AMT_BALANCE"]                 / (cc_df["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc_df["PAYMENT_RATIO"] = cc_df["AMT_PAYMENT_TOTAL_CURRENT"]   / (cc_df["AMT_TOTAL_RECEIVABLE"]    + 1)

    agg = cc_df.groupby("SK_ID_CURR").agg(
        CC_COUNT                  =("SK_ID_PREV",                 "count"),
        CC_AMT_BALANCE_MAX        =("AMT_BALANCE",                "max"),
        CC_AMT_BALANCE_MEAN       =("AMT_BALANCE",                "mean"),
        CC_AMT_CREDIT_LIMIT_MAX   =("AMT_CREDIT_LIMIT_ACTUAL",    "max"),
        CC_AMT_DRAWINGS_MEAN      =("AMT_DRAWINGS_CURRENT",       "mean"),
        CC_AMT_DRAWINGS_ATM_MEAN  =("AMT_DRAWINGS_ATM_CURRENT",   "mean"),
        CC_AMT_PAYMENT_MEAN       =("AMT_PAYMENT_TOTAL_CURRENT",  "mean"),
        CC_SK_DPD_MAX             =("SK_DPD",                     "max"),
        CC_SK_DPD_DEF_MAX         =("SK_DPD_DEF",                 "max"),
        CC_SK_DPD_MEAN            =("SK_DPD",                     "mean"),
        CC_DRAWING_RATIO_MEAN     =("DRAWING_RATIO",              "mean"),
        CC_BALANCE_RATIO_MEAN     =("BALANCE_RATIO",              "mean"),
        CC_PAYMENT_RATIO_MEAN     =("PAYMENT_RATIO",              "mean"),
        CC_CNT_INSTALMENT_CUM_MAX =("CNT_INSTALMENT_MATURE_CUM",  "max"),
    ).reset_index()
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 7. Encode categoricals
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Label-encode object columns that appear in both train and test.
    Returns (encoded_train, encoded_test).
    """
    from sklearn.preprocessing import LabelEncoder

    cat_cols = [c for c in train_df.columns if train_df[c].dtype == object]
    le = LabelEncoder()

    for col in cat_cols:
        if col not in test_df.columns:
            continue
        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))

    logger.info("Encoded %d categorical columns.", len(cat_cols))
    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 8. Master builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    dfs: dict,
    save: bool = True,
    out_dir=DATA_PROC,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Orchestrate all feature-engineering steps and return:
        (X_train, X_test, feature_cols)

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Raw DataFrames as returned by data_preprocessing.load_processed().
    save : bool
        Persist feature matrices to parquet if True.
    out_dir : Path
        Output directory for parquet files.
    """
    logger.info("Building feature matrix …")

    # Application tables
    train = engineer_application(dfs["app_train"])
    test  = engineer_application(dfs["app_test"])

    # Side-table aggregations
    bureau_agg = agg_bureau(dfs["bureau"], dfs["bureau_bal"])
    prev_agg   = agg_previous(dfs["prev_app"])
    inst_agg   = agg_installments(dfs["installments"])
    pos_agg    = agg_pos_cash(dfs["pos_cash"])
    cc_agg     = agg_credit_card(dfs["cc_balance"])

    # Merge onto application
    for agg_df in [bureau_agg, prev_agg, inst_agg, pos_agg, cc_agg]:
        train = train.merge(agg_df, on="SK_ID_CURR", how="left")
        test  = test.merge(agg_df,  on="SK_ID_CURR", how="left")

    # Encode
    TARGET_COL = "TARGET"
    ID_COL     = "SK_ID_CURR"

    # Separate target before encoding
    y_col_df = train[[TARGET_COL]].copy()
    train_enc = train.drop(columns=[TARGET_COL])
    train_enc, test_enc = encode_categoricals(train_enc, test)
    train = pd.concat([train_enc, y_col_df], axis=1)

    feature_cols = [c for c in train.columns if c not in [TARGET_COL, ID_COL]]

    logger.info("Feature matrix: train=%s  test=%s  features=%d",
                train.shape, test.shape, len(feature_cols))

    if save:
        train.to_parquet(out_dir / "train_features.parquet", index=False)
        test.to_parquet(out_dir  / "test_features.parquet",  index=False)
        logger.info("Feature matrices saved to %s", out_dir)

    return train, test, feature_cols


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_preprocessing import load_processed
    dfs = load_processed()
    build_feature_matrix(dfs)
