"""
Finalizes variable selection for mechanism design (Section 3.3.4).

Applies a majority-vote ensemble feature selection rule across three
independently-trained models (Logistic Regression, Random Forest, XGBoost):
a variable is selected if it appears in the top 15 by importance in AT LEAST
2 of the 3 models. This follows the ensemble/consensus feature selection
approach (Saeys, Abeel and Van de Peer, 2008), using majority voting rather
than requiring unanimous agreement across all selectors, consistent with
standard practice in the ensemble feature selection literature.

Run this LOCALLY, inside the same folder as your pipeline_outputs/ directory
(the one produced by run_full_pipeline.py).

Usage:
    python finalize_variable_selection.py
"""

import pandas as pd
import numpy as np
import joblib
import os

OUT = "pipeline_outputs"

# --- Load saved models, scaler, and training data ---
logreg = joblib.load(os.path.join(OUT, "logreg_model.joblib"))
rf = joblib.load(os.path.join(OUT, "rf_model.joblib"))
scaler = joblib.load(os.path.join(OUT, "scaler.joblib"))
X_train = pd.read_csv(os.path.join(OUT, "X_train.csv"))

# --- Logistic regression: absolute coefficient magnitude (scaled features) ---
coefs = pd.Series(np.abs(logreg.coef_[0]), index=X_train.columns).sort_values(ascending=False)
coefs.to_csv(os.path.join(OUT, "logreg_coef_ranking.csv"))

# --- Random Forest: built-in impurity-based feature importance ---
rf_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
rf_importance.to_csv(os.path.join(OUT, "rf_importance_ranking.csv"))

# --- SHAP ranking (XGBoost; already saved from the main pipeline run) ---
shap_ranking = pd.read_csv(os.path.join(OUT, "shap_ranking.csv"), index_col=0).iloc[:, 0]

TOP_N = 15
top_shap = set(shap_ranking.head(TOP_N).index)
top_logreg = set(coefs.head(TOP_N).index)
top_rf = set(rf_importance.head(TOP_N).index)

# --- Majority vote: appears in top-15 of at least 2 of 3 models ---
all_vars = top_shap | top_logreg | top_rf
vote_counts = pd.Series({
    v: sum([v in top_shap, v in top_logreg, v in top_rf]) for v in all_vars
}).sort_values(ascending=False)

selected = sorted(vote_counts[vote_counts >= 2].index)
unanimous = sorted(top_shap & top_logreg & top_rf)

print(f"Top {TOP_N} by SHAP (XGBoost):\n{sorted(top_shap)}\n")
print(f"Top {TOP_N} by |logistic coefficient|:\n{sorted(top_logreg)}\n")
print(f"Top {TOP_N} by Random Forest importance:\n{sorted(top_rf)}\n")

print(f"UNANIMOUS across all 3 models (n={len(unanimous)}): {unanimous}\n")

print(f"FINAL SELECTED VARIABLES — majority vote, >=2 of 3 models (n={len(selected)}):")
for v in selected:
    ranks = []
    if v in shap_ranking.index:
        ranks.append(f"SHAP={list(shap_ranking.index).index(v)+1}")
    if v in coefs.index:
        ranks.append(f"LogReg={list(coefs.index).index(v)+1}")
    if v in rf_importance.index:
        ranks.append(f"RF={list(rf_importance.index).index(v)+1}")
    print(f"  - {v}  (votes: {vote_counts[v]}/3; {', '.join(ranks)})")

only_one_vote = sorted(vote_counts[vote_counts == 1].index)
print(f"\nAppeared in only 1 of 3 models' top {TOP_N} (excluded): {only_one_vote}")

# Save final list with vote counts
result_df = vote_counts[vote_counts >= 2].sort_values(ascending=False).rename("votes_of_3")
result_df.to_csv(os.path.join(OUT, "final_selected_variables.csv"))
print(f"\nSaved to {OUT}/final_selected_variables.csv")

# --- Apply Tier C exclusion (Section 3.4.1) ---
# Variables that are empirically robust (pass the majority vote) but are
# centrally computed/held by an external institution (FICO) or sourced from
# public/court records (pub_rec, pub_rec_bankruptcies, tax_liens) are retained
# in Chapter 4 as a benchmark only, and excluded here from the variable set
# carried forward into the mechanism design in Chapter 5.
TIER_C = ["fico_score", "pub_rec", "pub_rec_bankruptcies", "tax_liens"]
mechanism_design_vars = sorted(set(selected) - set(TIER_C))
tier_c_removed = sorted(set(selected) & set(TIER_C))

print(f"\n{'='*70}")
print("TIER C FILTER (Section 3.4.1) — applied on top of the majority vote")
print(f"{'='*70}")
print(f"Removed as Tier C (empirically robust, but not on-chain translatable): {tier_c_removed}")
print(f"\nFINAL VARIABLE SET FOR MECHANISM DESIGN (n={len(mechanism_design_vars)}):")
for v in mechanism_design_vars:
    print(f"  - {v}")

pd.Series(mechanism_design_vars, name="mechanism_design_variable").to_csv(
    os.path.join(OUT, "final_mechanism_design_variables.csv"), index=False
)
print(f"\nSaved to {OUT}/final_mechanism_design_variables.csv")

