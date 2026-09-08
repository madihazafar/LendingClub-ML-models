"""
Generates SHAP dependence plots for the seven variables that enter the
reputation score in Chapter 5 (categories A and B, Section 5.5):

  mort_acc, home_ownership_MORTGAGE                (Category A - persistent)
  dti, acc_open_past_24mths, inq_last_6mths,
  credit_exposure_composite, revolving_util_composite  (Category B - dynamic)

Note: credit_exposure_composite and revolving_util_composite were built
AFTER the original SHAP analysis in Chapter 4 (they replace five separate
source variables), so no dependence plot for them exists yet. This script
refits XGBoost directly on the seven final variables (plus term/loan_amnt,
kept as context but not plotted) so that genuine, composite-aware SHAP
values can be computed, rather than inferring the composites' shape
indirectly from their construction.

SHAP values are computed on the TRAINING set, consistent with the
methodology correction in Section 3.3.3 (avoiding test-set contamination
of any variable-selection-adjacent analysis).

Run this LOCALLY, using the output of build_composite_variables.py
(lendingclub_with_composites.csv) as input.

Usage:
    python generate_final_variable_dependence_plots.py lendingclub_with_composites.csv
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "lendingclub_with_composites.csv"
OUT = "dependence_plots_final_vars"
os.makedirs(OUT, exist_ok=True)

# The final mechanism-design variable set (Chapter 5, Table 5.2)
FINAL_VARS = [
    "term", "loan_amnt",                     # Category C - context only, not plotted
    "dti", "acc_open_past_24mths", "inq_last_6mths",
    "credit_exposure_composite", "revolving_util_composite",  # Category B
    "mort_acc", "home_ownership_MORTGAGE",   # Category A
]

# The seven Category A/B variables that actually enter the reputation score
DEPENDENCE_PLOT_VARS = [
    "dti", "acc_open_past_24mths", "inq_last_6mths",
    "credit_exposure_composite", "revolving_util_composite",
    "mort_acc", "home_ownership_MORTGAGE",
]

print("Loading data...")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded: {df.shape}")

missing = [v for v in FINAL_VARS if v not in df.columns]
if missing:
    raise ValueError(f"Expected columns not found: {missing}. "
                      f"Make sure you're using the output of build_composite_variables.py.")

y = df["default"]
X = df[FINAL_VARS].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

spw = (y_train == 0).sum() / (y_train == 1).sum()
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    scale_pos_weight=spw, eval_metric="logloss", random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"\nModel AUC on held-out test set (9 final variables only): {auc:.4f}")
print("(Expect this to be somewhat lower than the full 114-variable model in Chapter 4 -- "
      "this model uses only the mechanism-design variable set, not all available predictors.)")

# SHAP on the TRAINING set, consistent with Section 3.3.3's corrected methodology
print("\nComputing SHAP values (training set)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
ranking = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)
print("\nMean |SHAP value| for all 9 variables (term/loan_amnt included for context):")
print(ranking)
ranking.to_csv(os.path.join(OUT, "shap_ranking_final_vars.csv"))

# Dependence plots for the 7 reputation-score variables
for feat in DEPENDENCE_PLOT_VARS:
    plt.figure()
    shap.dependence_plot(feat, shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"shap_dependence_{feat}.png"), dpi=150)
    plt.close()
    print(f"Saved: {OUT}/shap_dependence_{feat}.png")

print(f"\nDone. All outputs saved to ./{OUT}/")
