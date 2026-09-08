"""
Full verification_status investigation, addressing four questions:
  1. Raw default rate by verification status (unconditional)
  2. SHAP contribution by income quartile (already established previously)
  3. Model performance (AUC) with verification columns removed
  4. Real SHAP interaction values between annual_inc and verification status
     (not just binned quartile comparison)

Run this LOCALLY on your full filtered dataset.

Usage:
    python test_verification_status.py lendingclub_filtered.csv
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import shap

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "lendingclub_filtered.csv"


def clean_data(df):
    df = df.drop(columns=[c for c in ["policy_code", "pymnt_plan"] if c in df.columns])
    df = df.drop(columns=[c for c in ["int_rate", "grade", "sub_grade", "installment"] if c in df.columns])
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df = df.drop(columns=["fico_range_low", "fico_range_high", "funded_amnt", "funded_amnt_inv"])
    df = df.drop(columns=[c for c in ["addr_state", "zip_code", "emp_title", "title", "url"] if c in df.columns])
    df["issue_d_parsed"] = pd.to_datetime(df["issue_d"], format="%b-%y")
    df["earliest_cr_line_parsed"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%y")
    df["credit_hist_months"] = ((df["issue_d_parsed"].dt.year - df["earliest_cr_line_parsed"].dt.year) * 12
        + (df["issue_d_parsed"].dt.month - df["earliest_cr_line_parsed"].dt.month))
    df = df.drop(columns=["issue_d", "earliest_cr_line", "issue_d_parsed", "earliest_cr_line_parsed"])

    ZERO_EVENT_COLS = ["mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
        "mths_since_recent_bc_dlq", "mths_since_recent_revol_delinq", "mths_since_recent_inq"]
    for c in ZERO_EVENT_COLS:
        if c in df.columns:
            df[c + "_ever"] = df[c].notna().astype(int)
            df[c] = df[c].fillna(999)

    VINTAGE_BLOCK_COLS = ["il_util", "all_util", "open_il_24m", "open_acc_6m", "total_cu_tl",
        "inq_last_12m", "open_il_12m", "total_bal_il", "open_rv_24m", "open_rv_12m", "max_bal_bc",
        "inq_fi", "open_act_il", "mths_since_rcnt_il"]
    for c in VINTAGE_BLOCK_COLS:
        if c in df.columns:
            df[c + "_available"] = df[c].notna().astype(int)
            df[c] = df[c].fillna(df[c].median())

    df["dti"] = df["dti"].replace(999, np.nan)
    df["dti"] = df["dti"].fillna(df["dti"].median())
    df["dti"] = df["dti"].clip(upper=df["dti"].quantile(0.99))

    EMP_MAP = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5,
        "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10}
    df["emp_length"] = df["emp_length"].map(EMP_MAP)
    df["emp_length"] = df["emp_length"].fillna(df["emp_length"].median())

    for c in df.select_dtypes(include="number").columns[df.select_dtypes(include="number").isna().any()]:
        df[c] = df[c].fillna(df[c].median())

    df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(int)
    df["initial_list_status"] = (df["initial_list_status"] == "w").astype(int)
    df["application_type"] = (df["application_type"] == "Joint App").astype(int)
    df["disbursement_method"] = (df["disbursement_method"] == "DirectPay").astype(int)
    df = pd.get_dummies(df, columns=["home_ownership", "verification_status", "purpose"], drop_first=True)
    df = df.drop(columns=[c for c in ["loan_status"] if c in df.columns])

    leftover_obj = df.select_dtypes(include="object").columns.tolist()
    if leftover_obj:
        print(f"WARNING: dropping unexpected non-numeric columns: {leftover_obj}")
        df = df.drop(columns=leftover_obj)
    return df


print("Loading and cleaning...")
df_raw = pd.read_csv(INPUT_PATH, low_memory=False)

# --- Bullet 1: raw default rate by verification status (on RAW labels, before one-hot) ---
print("\n=== Bullet 1: Raw default rate by verification_status ===")
default_col = df_raw["loan_status"].isin(["Charged Off", "Default"]).astype(int)
raw_check = pd.DataFrame({"verification_status": df_raw["verification_status"], "default": default_col})
print(raw_check.groupby("verification_status")["default"].agg(["mean", "count"]))

df = clean_data(df_raw.copy())
y = df["default"]
X_full = df.drop(columns=["default"])
verif_cols = [c for c in X_full.columns if "verification_status" in c]
print("\nVerification columns:", verif_cols)

X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.25, random_state=42, stratify=y)
spw = (y_train == 0).sum() / (y_train == 1).sum()

# --- Full model (for bullet 2 and 4) ---
m_full = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=spw,
                            eval_metric="logloss", random_state=42, n_jobs=-1)
m_full.fit(X_train, y_train)
auc_full = roc_auc_score(y_test, m_full.predict_proba(X_test)[:, 1])

# --- Bullet 2: SHAP by income quartile (re-confirm) ---
explainer = shap.TreeExplainer(m_full)
shap_values_test = explainer.shap_values(X_test)
ver_idx = list(X_test.columns).index("verification_status_Verified")
inc_idx = list(X_test.columns).index("annual_inc")

check = pd.DataFrame({
    "verified": X_test["verification_status_Verified"].values,
    "annual_inc": X_test["annual_inc"].values,
    "shap": shap_values_test[:, ver_idx],
})
check["income_band"] = pd.qcut(check["annual_inc"], 4, duplicates="drop")
print("\n=== Bullet 2: SHAP contribution by income quartile ===")
print(check.groupby(["income_band", "verified"])["shap"].mean().unstack())

# --- Bullet 3: model performance with verification removed ---
X_train_nv = X_train.drop(columns=verif_cols)
X_test_nv = X_test.drop(columns=verif_cols)
m_nv = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=spw,
                          eval_metric="logloss", random_state=42, n_jobs=-1)
m_nv.fit(X_train_nv, y_train)
auc_nv = roc_auc_score(y_test, m_nv.predict_proba(X_test_nv)[:, 1])

print("\n=== Bullet 3: Model performance with verification removed ===")
print(f"Full model AUC:            {auc_full:.4f}")
print(f"Without verification cols: {auc_nv:.4f}")
print(f"Difference:                {auc_full - auc_nv:.4f}")

# --- Bullet 4: real SHAP interaction values ---
print("\n=== Bullet 4: SHAP interaction (annual_inc x verification_status_Verified) ===")
X_test_sub = X_test.sample(n=5000, random_state=42)
inter = explainer.shap_interaction_values(X_test_sub)
main_effect = np.abs(explainer.shap_values(X_test_sub)[:, ver_idx]).mean()
interaction_strength = np.abs(inter[:, inc_idx, ver_idx]).mean()
print(f"Mean |interaction value|:        {interaction_strength:.5f}")
print(f"Mean |main effect| of verified:  {main_effect:.5f}")
print(f"Interaction as % of main effect: {interaction_strength / main_effect * 100:.1f}%")
