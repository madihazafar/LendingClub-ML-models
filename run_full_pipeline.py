"""
LendingClub default-prediction pipeline — finalized version
=============================================================
Run this LOCALLY on the full filtered dataset (output of prefilter_lendingclub.py,
i.e. lendingclub_filtered.csv — ~611,803 rows, 94 columns).

This script performs every step built and validated on the 150k sample:
  3.3.1  Cleaning (drops, endogenous-variable exclusion, missingness handling)
  3.3.2  Model comparison (Logistic Regression / Random Forest / XGBoost)
  3.3.3  SHAP analysis (global importance, dependence plots)
         + cross-model robustness check (SHAP vs. logistic coefficients)
  3.4.1  Tier C ablation (FICO / public-record variables) — predictive
         ceiling comparison, referenced directly in the "three-tier
         translatability" discussion
         + verification_status / emp_length checks (credential argument)

Install requirements first:
    pip install pandas numpy scikit-learn xgboost shap matplotlib joblib --break-system-packages

Usage:
    python run_full_pipeline.py lendingclub_filtered.csv

Expected runtime: several minutes on ~612k rows / ~90 features, depending on machine.
All outputs are written to a folder called pipeline_outputs/.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import xgboost as xgb
import shap

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "lendingclub_filtered.csv"
OUT = "pipeline_outputs"
os.makedirs(OUT, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150)
    plt.close()

# =================================================================
# 3.3.1 — Cleaning
# =================================================================
print("=" * 70)
print("STEP 1: Loading and cleaning (Section 3.3.1)")
print("=" * 70)

df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Starting shape: {df.shape}")

# --- Constant / uninformative columns ---
CONSTANT_DROP = ["policy_code", "pymnt_plan"]
df = df.drop(columns=[c for c in CONSTANT_DROP if c in df.columns])

# --- Endogenous / platform-assigned variables ---
# int_rate, grade, sub_grade reflect LendingClub's own risk assessment;
# installment is a near-deterministic function of loan_amnt, int_rate, term.
ENDOGENOUS_DROP = ["int_rate", "grade", "sub_grade", "installment"]
df = df.drop(columns=[c for c in ENDOGENOUS_DROP if c in df.columns])

# --- Redundant / near-duplicate columns ---
if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df = df.drop(columns=["fico_range_low", "fico_range_high"])
df = df.drop(columns=[c for c in ["funded_amnt", "funded_amnt_inv"] if c in df.columns])

# --- Geography: dropped, not central to the research question ---
df = df.drop(columns=[c for c in ["addr_state", "zip_code"] if c in df.columns])
df = df.drop(columns=[c for c in ["emp_title", "title", "url"] if c in df.columns])

# --- Dates -> derive credit history length (the "wallet age" analogue) ---
if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
    df["issue_d_parsed"] = pd.to_datetime(df["issue_d"], format="%b-%y")
    df["earliest_cr_line_parsed"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%y")
    df["credit_hist_months"] = (
        (df["issue_d_parsed"].dt.year - df["earliest_cr_line_parsed"].dt.year) * 12
        + (df["issue_d_parsed"].dt.month - df["earliest_cr_line_parsed"].dt.month)
    )
    df = df.drop(columns=["issue_d", "earliest_cr_line", "issue_d_parsed", "earliest_cr_line_parsed"])

# --- Missingness: zero-event fields (flag + sentinel) ---
ZERO_EVENT_COLS = [
    "mths_since_last_delinq", "mths_since_last_record",
    "mths_since_last_major_derog", "mths_since_recent_bc_dlq",
    "mths_since_recent_revol_delinq", "mths_since_recent_inq",
]
SENTINEL = 999
for c in ZERO_EVENT_COLS:
    if c in df.columns:
        df[c + "_ever"] = df[c].notna().astype(int)
        df[c] = df[c].fillna(SENTINEL)

# --- Missingness: vintage-availability block (flag + median impute) ---
VINTAGE_BLOCK_COLS = [
    "il_util", "all_util", "open_il_24m", "open_acc_6m", "total_cu_tl",
    "inq_last_12m", "open_il_12m", "total_bal_il", "open_rv_24m",
    "open_rv_12m", "max_bal_bc", "inq_fi", "open_act_il", "mths_since_rcnt_il",
]
for c in VINTAGE_BLOCK_COLS:
    if c in df.columns:
        df[c + "_available"] = df[c].notna().astype(int)
        df[c] = df[c].fillna(df[c].median())

# --- dti: LendingClub's own 999 sentinel, plus winsorization of extreme values ---
if "dti" in df.columns:
    df["dti"] = df["dti"].replace(999, np.nan)
    df["dti"] = df["dti"].fillna(df["dti"].median())
    # Winsorization threshold: 99th percentile, chosen following the convention
    # described in Riffenburgh (2020) and confirmed empirically (comparison
    # across no-winsorization and multiple percentile levels showed AUC was
    # materially unaffected by threshold choice; see accompanying analysis).
    # Computed dynamically so it is reproducible on any comparable dataset,
    # rather than hardcoded to this dataset's specific value.
    DTI_CAP = df["dti"].quantile(0.99)
    n_capped = (df["dti"] > DTI_CAP).sum()
    df["dti"] = df["dti"].clip(upper=DTI_CAP)
    print(f"dti: recoded 999 sentinel to NaN and imputed; winsorized {n_capped} rows above {DTI_CAP:.2f} (99th percentile)")

# --- emp_length: ordinal encode ---
EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}
if "emp_length" in df.columns:
    df["emp_length"] = df["emp_length"].map(EMP_LENGTH_MAP)
    df["emp_length"] = df["emp_length"].fillna(df["emp_length"].median())

# --- Remaining low-missingness numeric columns: median impute ---
remaining_numeric_na = df.select_dtypes(include="number").columns[
    df.select_dtypes(include="number").isna().any()
]
for c in remaining_numeric_na:
    df[c] = df[c].fillna(df[c].median())

# --- Encode categoricals ---
if "term" in df.columns:
    df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(int)
if "initial_list_status" in df.columns:
    df["initial_list_status"] = (df["initial_list_status"] == "w").astype(int)
if "application_type" in df.columns:
    df["application_type"] = (df["application_type"] == "Joint App").astype(int)
if "disbursement_method" in df.columns:
    df["disbursement_method"] = (df["disbursement_method"] == "DirectPay").astype(int)

ONEHOT_COLS = [c for c in ["home_ownership", "verification_status", "purpose"] if c in df.columns]
df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=True)

# --- Drop redundant raw label (keep binary 'default' target) ---
df = df.drop(columns=[c for c in ["loan_status"] if c in df.columns])

# Any remaining object columns are unexpected — drop with a warning rather than fail
leftover_obj = df.select_dtypes(include="object").columns.tolist()
if leftover_obj:
    print(f"WARNING: dropping unexpected non-numeric columns: {leftover_obj}")
    df = df.drop(columns=leftover_obj)

print(f"Final cleaned shape: {df.shape}")
print(f"Remaining missing values: {df.isna().sum().sum()}")
print(f"Target balance:\n{df['default'].value_counts(normalize=True)}")

df.to_csv(os.path.join(OUT, "lendingclub_cleaned_full.csv"), index=False)

# =================================================================
# 3.3.2 — Model comparison
# =================================================================
print("\n" + "=" * 70)
print("STEP 2: Model comparison (Section 3.3.2)")
print("=" * 70)

y = df["default"]
X = df.drop(columns=["default"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate(name, model, Xtr, Xte, is_scaled=False):
    proba = model.predict_proba(Xte)[:, 1]
    pred = model.predict(Xte)
    return {
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "Accuracy": accuracy_score(y_test, pred),
    }

results = {}

logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
logreg.fit(X_train_scaled, y_train)
results["Logistic Regression"] = evaluate("Logistic Regression", logreg, X_train_scaled, X_test_scaled)

rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results["Random Forest"] = evaluate("Random Forest", rf, X_train, X_test)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train)
results["XGBoost"] = evaluate("XGBoost", xgb_model, X_train, X_test)

results_df = pd.DataFrame(results).T.round(4)
print("\n=== Model comparison (test set) ===")
print(results_df)
results_df.to_csv(os.path.join(OUT, "model_comparison_results.csv"))

joblib.dump(xgb_model, os.path.join(OUT, "xgb_model.joblib"))
joblib.dump(rf, os.path.join(OUT, "rf_model.joblib"))
joblib.dump(logreg, os.path.join(OUT, "logreg_model.joblib"))
joblib.dump(scaler, os.path.join(OUT, "scaler.joblib"))
X_test.to_csv(os.path.join(OUT, "X_test.csv"), index=False)
X_train.to_csv(os.path.join(OUT, "X_train.csv"), index=False)

# =================================================================
# 3.3.3 — SHAP analysis (best model: XGBoost)
# =================================================================
print("\n" + "=" * 70)
print("STEP 3: SHAP analysis (Section 3.3.3)")
print("=" * 70)

# SHAP values are computed on the TRAINING set, consistent with the other
# two selection measures (logistic regression coefficients and Random Forest
# impurity-based importance, both properties of the fitted model derived
# from training data). This keeps the test set completely uninformed by the
# variable-selection process (Section 3.3.4); the test set is used only once,
# for the final model evaluation already computed in Step 2 above.
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)
np.save(os.path.join(OUT, "shap_values.npy"), shap_values)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
ranking = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)
ranking.to_csv(os.path.join(OUT, "shap_ranking.csv"))
print("\nTop 20 variables by mean |SHAP value| (training set):")
print(ranking.head(20))

plt.figure()
shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=20, show=False)
savefig("shap_bar.png")

plt.figure()
shap.summary_plot(shap_values, X_train, max_display=20, show=False)
savefig("shap_beeswarm.png")

# Dependence plots for the cross-model-robust core variables identified on the sample
for feat in ["dti", "term", "loan_amnt", "acc_open_past_24mths"]:
    if feat in X_train.columns:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X_train, show=False)
        savefig(f"shap_dependence_{feat}.png")

# --- Supplementary: test-set SHAP, for interpretive comparison only ---
# NOT used for any selection decision. Included only to check that the
# shape/direction of relationships found on training data also holds
# out-of-sample.
print("\n(Supplementary, interpretive only) Computing test-set SHAP for comparison...")
shap_values_test_supplementary = explainer.shap_values(X_test)
ranking_test_supplementary = pd.Series(
    np.abs(shap_values_test_supplementary).mean(axis=0), index=X.columns
).sort_values(ascending=False)
ranking_test_supplementary.to_csv(os.path.join(OUT, "shap_ranking_TEST_SUPPLEMENTARY.csv"))
print("Top 10, test set (supplementary only, not used for selection):")
print(ranking_test_supplementary.head(10))

# --- Cross-model robustness check: SHAP vs. logistic regression coefficients ---
# Both computed on the training set.
coefs = pd.Series(np.abs(logreg.coef_[0]), index=X.columns).sort_values(ascending=False)
top_shap = set(ranking.head(15).index)
top_logreg = set(coefs.head(15).index)
overlap = top_shap & top_logreg
print(f"\nOverlap between top-15 SHAP and top-15 |logistic coefficient|: {len(overlap)}/15")
print("Robust core (in both):", sorted(overlap))

# =================================================================
# 3.4.1 — Tier C ablation (FICO / public-record variables)
# =================================================================
print("\n" + "=" * 70)
print("STEP 4: Tier C ablation (Section 3.4.1)")
print("=" * 70)

TIER_C = ["fico_score", "pub_rec", "pub_rec_bankruptcies", "tax_liens"]
tier_c_present = [c for c in TIER_C if c in X.columns]
X_no_tierc = X.drop(columns=tier_c_present)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_no_tierc, y, test_size=0.25, random_state=42, stratify=y
)
spw2 = (yc_train == 0).sum() / (yc_train == 1).sum()
model_no_tierc = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    scale_pos_weight=spw2, eval_metric="logloss", random_state=42, n_jobs=-1
)
model_no_tierc.fit(Xc_train, yc_train)
auc_full = results["XGBoost"]["AUC"]
auc_no_tierc = roc_auc_score(yc_test, model_no_tierc.predict_proba(Xc_test)[:, 1])
print(f"Full feature set AUC:        {auc_full:.4f}")
print(f"Excluding Tier C ({tier_c_present}) AUC: {auc_no_tierc:.4f}")
print(f"Difference: {auc_full - auc_no_tierc:.4f}")

explainer_c = shap.TreeExplainer(model_no_tierc)
shap_values_c = explainer_c.shap_values(Xc_train)
ranking_c = pd.Series(
    np.abs(shap_values_c).mean(axis=0), index=X_no_tierc.columns
).sort_values(ascending=False)
ranking_c.to_csv(os.path.join(OUT, "shap_ranking_no_tierc.csv"))
print("\nTop 15 variables, Tier C excluded (training set):")
print(ranking_c.head(15))

with open(os.path.join(OUT, "tier_c_ablation_summary.txt"), "w") as f:
    f.write(f"Full feature set AUC: {auc_full:.4f}\n")
    f.write(f"Excluding Tier C AUC: {auc_no_tierc:.4f}\n")
    f.write(f"Difference: {auc_full - auc_no_tierc:.4f}\n")
    f.write(f"Tier C variables present: {tier_c_present}\n")

# =================================================================
# Credential-argument checks: verification_status / emp_length
# =================================================================
print("\n" + "=" * 70)
print("STEP 5: Verification status / emp_length checks (training set, consistent with selection)")
print("=" * 70)

verif_cols = [c for c in X_train.columns if "verification_status" in c]
for vcol in verif_cols:
    if vcol in X_train.columns and "annual_inc" in X_train.columns:
        idx = X_train.columns.get_loc(vcol)
        check = pd.DataFrame({
            vcol: X_train[vcol].values,
            "annual_inc": X_train["annual_inc"].values,
            "shap": shap_values[:, idx],
        })
        check["income_band"] = pd.qcut(check["annual_inc"], 4, duplicates="drop")
        print(f"\n{vcol} — mean SHAP by income band:")
        print(check.groupby(["income_band", vcol])["shap"].mean().unstack())

if "emp_length" in X_train.columns:
    idx = X_train.columns.get_loc("emp_length")
    emp_check = pd.DataFrame({
        "emp_length": X_train["emp_length"].values,
        "shap": shap_values[:, idx],
    })
    print("\nemp_length — mean SHAP by value:")
    print(emp_check.groupby("emp_length")["shap"].mean())

print("\n" + "=" * 70)
print(f"DONE. All outputs saved to ./{OUT}/")
print("=" * 70)
