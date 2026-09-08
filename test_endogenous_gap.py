"""
Tests whether including the endogenous, platform-assigned variables
(interest rate, grade, sub-grade, instalment) excluded in Section 3.3.1
changes the gap between Logistic Regression and XGBoost performance.

Run this LOCALLY on lendingclub_filtered.csv (the output of
prefilter_lendingclub.py, i.e. the full ~611,803-row dataset).

Usage:
    python test_endogenous_gap.py lendingclub_filtered.csv
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "lendingclub_filtered.csv"


def clean_data(df, keep_endogenous=False):
    CONSTANT_DROP = ["policy_code", "pymnt_plan"]
    df = df.drop(columns=[c for c in CONSTANT_DROP if c in df.columns])

    if not keep_endogenous:
        ENDOGENOUS_DROP = ["int_rate", "grade", "sub_grade", "installment"]
        df = df.drop(columns=[c for c in ENDOGENOUS_DROP if c in df.columns])

    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df = df.drop(columns=["fico_range_low", "fico_range_high", "funded_amnt", "funded_amnt_inv"])
    df = df.drop(columns=[c for c in ["addr_state", "zip_code", "emp_title", "title", "url"] if c in df.columns])

    df["issue_d_parsed"] = pd.to_datetime(df["issue_d"], format="%b-%y")
    df["earliest_cr_line_parsed"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%y")
    df["credit_hist_months"] = (
        (df["issue_d_parsed"].dt.year - df["earliest_cr_line_parsed"].dt.year) * 12
        + (df["issue_d_parsed"].dt.month - df["earliest_cr_line_parsed"].dt.month)
    )
    df = df.drop(columns=["issue_d", "earliest_cr_line", "issue_d_parsed", "earliest_cr_line_parsed"])

    ZERO_EVENT_COLS = [
        "mths_since_last_delinq", "mths_since_last_record",
        "mths_since_last_major_derog", "mths_since_recent_bc_dlq",
        "mths_since_recent_revol_delinq", "mths_since_recent_inq",
    ]
    for c in ZERO_EVENT_COLS:
        if c in df.columns:
            df[c + "_ever"] = df[c].notna().astype(int)
            df[c] = df[c].fillna(999)

    VINTAGE_BLOCK_COLS = [
        "il_util", "all_util", "open_il_24m", "open_acc_6m", "total_cu_tl",
        "inq_last_12m", "open_il_12m", "total_bal_il", "open_rv_24m",
        "open_rv_12m", "max_bal_bc", "inq_fi", "open_act_il", "mths_since_rcnt_il",
    ]
    for c in VINTAGE_BLOCK_COLS:
        if c in df.columns:
            df[c + "_available"] = df[c].notna().astype(int)
            df[c] = df[c].fillna(df[c].median())

    df["dti"] = df["dti"].replace(999, np.nan)
    df["dti"] = df["dti"].fillna(df["dti"].median())
    df["dti"] = df["dti"].clip(upper=df["dti"].quantile(0.99))

    EMP_LENGTH_MAP = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
        "10+ years": 10,
    }
    df["emp_length"] = df["emp_length"].map(EMP_LENGTH_MAP)
    df["emp_length"] = df["emp_length"].fillna(df["emp_length"].median())

    remaining_numeric_na = df.select_dtypes(include="number").columns[
        df.select_dtypes(include="number").isna().any()
    ]
    for c in remaining_numeric_na:
        df[c] = df[c].fillna(df[c].median())

    df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(int)
    df["initial_list_status"] = (df["initial_list_status"] == "w").astype(int)
    df["application_type"] = (df["application_type"] == "Joint App").astype(int)
    df["disbursement_method"] = (df["disbursement_method"] == "DirectPay").astype(int)

    ONEHOT_COLS = ["home_ownership", "verification_status", "purpose"]
    if keep_endogenous and "grade" in df.columns:
        ONEHOT_COLS += ["grade", "sub_grade"]
    df = pd.get_dummies(df, columns=[c for c in ONEHOT_COLS if c in df.columns], drop_first=True)

    df = df.drop(columns=[c for c in ["loan_status"] if c in df.columns])

    leftover_obj = df.select_dtypes(include="object").columns.tolist()
    if leftover_obj:
        print(f"WARNING: dropping unexpected non-numeric columns: {leftover_obj}")
        df = df.drop(columns=leftover_obj)

    return df


def run_comparison(df, label):
    y = df["default"]
    X = df.drop(columns=["default"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    logreg.fit(X_train_s, y_train)
    auc_lr = roc_auc_score(y_test, logreg.predict_proba(X_test_s)[:, 1])

    spw = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

    print(f"\n--- {label} ---")
    print(f"Shape: {X.shape}")
    print(f"Logistic Regression AUC: {auc_lr:.4f}")
    print(f"XGBoost AUC:             {auc_xgb:.4f}")
    print(f"Gap (XGB - LR):          {auc_xgb - auc_lr:.4f}")
    return auc_lr, auc_xgb


print("Loading data...")
df_base = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded: {df_base.shape}")

df_excluded = clean_data(df_base.copy(), keep_endogenous=False)
lr1, xgb1 = run_comparison(df_excluded, "Current approach: endogenous variables EXCLUDED")

df_included = clean_data(df_base.copy(), keep_endogenous=True)
lr2, xgb2 = run_comparison(df_included, "Test: endogenous variables (int_rate, grade, sub_grade, installment) INCLUDED")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Gap EXCLUDED: {xgb1 - lr1:.4f}")
print(f"Gap INCLUDED: {xgb2 - lr2:.4f}")
print(f"LR gain from endogenous variables:   {lr2 - lr1:.4f}")
print(f"XGBoost gain from endogenous variables: {xgb2 - xgb1:.4f}")
direction = "WIDENED" if (xgb2 - lr2) > (xgb1 - lr1) else "NARROWED"
print(f"\nGap {direction} when endogenous variables were included.")
