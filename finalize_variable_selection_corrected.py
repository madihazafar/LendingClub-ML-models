"""
Finalizes variable selection for mechanism design (Section 3.3.4) — CORRECTED VERSION.

Uses the same majority-vote ensemble approach as finalize_variable_selection.py,
but replaces the original logistic regression ranking (distorted by the
sentinel-value encoding, see check_logreg_without_sentinels.py) with the
corrected ranking computed with sentinel-filled columns removed.

Rule: a variable is selected if it appears in the top 15 by importance in
AT LEAST 2 of the 3 models:
  - SHAP (XGBoost)
  - |coefficient| (Logistic Regression, sentinel columns removed)
  - Random Forest importance

Then applies the Tier C filter (Section 3.4.1): FICO and public-record
variables are removed from the mechanism-design set even if selected,
since they are not on-chain translatable (retained only as a Chapter 4
benchmark).

Run this AFTER both finalize_variable_selection.py and
check_logreg_without_sentinels.py, in the same folder as pipeline_outputs/.

Usage:
    python finalize_variable_selection_corrected.py
"""

import pandas as pd
import os

OUT = "pipeline_outputs"

shap_ranking = pd.read_csv(os.path.join(OUT, "shap_ranking.csv"), index_col=0).iloc[:, 0]
rf_importance = pd.read_csv(os.path.join(OUT, "rf_importance_ranking.csv"), index_col=0).iloc[:, 0]
coefs_corrected = pd.read_csv(
    os.path.join(OUT, "logreg_coef_ranking_no_sentinel.csv"), index_col=0
).iloc[:, 0]

TOP_N = 15
top_shap = set(shap_ranking.head(TOP_N).index)
top_rf = set(rf_importance.head(TOP_N).index)
top_logreg = set(coefs_corrected.head(TOP_N).index)

all_vars = top_shap | top_logreg | top_rf
vote_counts = pd.Series({
    v: sum([v in top_shap, v in top_logreg, v in top_rf]) for v in all_vars
}).sort_values(ascending=False)

selected = sorted(vote_counts[vote_counts >= 2].index)
unanimous = sorted(top_shap & top_logreg & top_rf)

print(f"Top {TOP_N} by SHAP (XGBoost):\n{sorted(top_shap)}\n")
print(f"Top {TOP_N} by |logistic coefficient|, CORRECTED (sentinel cols removed):\n{sorted(top_logreg)}\n")
print(f"Top {TOP_N} by Random Forest importance:\n{sorted(top_rf)}\n")

print(f"UNANIMOUS across all 3 models (n={len(unanimous)}): {unanimous}\n")

print(f"MAJORITY-VOTE SELECTED (>=2 of 3, corrected LogReg) (n={len(selected)}):")
for v in selected:
    votes_detail = []
    if v in shap_ranking.index and v in top_shap:
        votes_detail.append(f"SHAP={list(shap_ranking.index).index(v)+1}")
    if v in coefs_corrected.index and v in top_logreg:
        votes_detail.append(f"LogReg={list(coefs_corrected.index).index(v)+1}")
    if v in rf_importance.index and v in top_rf:
        votes_detail.append(f"RF={list(rf_importance.index).index(v)+1}")
    print(f"  - {v}  (votes: {vote_counts[v]}/3; {', '.join(votes_detail)})")

# --- Tier C filter ---
TIER_C = ["fico_score", "pub_rec", "pub_rec_bankruptcies", "tax_liens"]
mechanism_design_vars = sorted(set(selected) - set(TIER_C))
tier_c_removed = sorted(set(selected) & set(TIER_C))

print(f"\n{'='*70}")
print("TIER C FILTER (Section 3.4.1)")
print(f"{'='*70}")
print(f"Removed as Tier C: {tier_c_removed}")
print(f"\nFINAL VARIABLE SET FOR MECHANISM DESIGN, CORRECTED (n={len(mechanism_design_vars)}):")
for v in mechanism_design_vars:
    print(f"  - {v}")

# --- Comparison against the original (uncorrected) result, for transparency ---
try:
    original = pd.read_csv(os.path.join(OUT, "final_mechanism_design_variables.csv"))
    original_set = set(original.iloc[:, 0])
    new_set = set(mechanism_design_vars)
    print(f"\n{'='*70}")
    print("CHANGE LOG vs. original (uncorrected LogReg) mechanism design set")
    print(f"{'='*70}")
    print(f"Removed (no longer selected): {sorted(original_set - new_set)}")
    print(f"Added (newly selected): {sorted(new_set - original_set)}")
except FileNotFoundError:
    pass

pd.Series(mechanism_design_vars, name="mechanism_design_variable").to_csv(
    os.path.join(OUT, "final_mechanism_design_variables_corrected.csv"), index=False
)
print(f"\nSaved to {OUT}/final_mechanism_design_variables_corrected.csv")
