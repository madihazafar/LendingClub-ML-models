"""
Builds two composite variables to resolve multicollinearity found among the
majority-vote selected variables (Section 3.3.4), following standard
composite-index construction practice (z-score averaging of correlated
components, e.g. Wood et al. driving-performance composites; general
practice per composite-index methodology).

  credit_exposure_composite   = average(z(avg_cur_bal), z(tot_hi_cred_lim))
  revolving_util_composite    = average(z(total_bc_limit), -z(bc_open_to_buy), z(total_rev_hi_lim))
                                 (bc_open_to_buy is *available* credit, so its
                                 sign is flipped to align direction with the
                                 other two "more exposure" variables)

Run this LOCALLY, in the same folder as pipeline_outputs/.

Usage:
    python build_composite_variables.py
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import pointbiserialr

OUT = "pipeline_outputs"

df = pd.read_csv(os.path.join(OUT, "lendingclub_cleaned_full.csv"))
print(f"Loaded: {df.shape}")

def zscore(s):
    return (s - s.mean()) / s.std()

# --- Build composites ---
df["credit_exposure_composite"] = (zscore(df["avg_cur_bal"]) + zscore(df["tot_hi_cred_lim"])) / 2
df["revolving_util_composite"] = (
    zscore(df["total_bc_limit"]) - zscore(df["bc_open_to_buy"]) + zscore(df["total_rev_hi_lim"])
) / 3

# --- Sanity checks ---
print("\nCorrelation between the two composites (should be well below the ~0.8 seen among originals):")
print(df[["credit_exposure_composite", "revolving_util_composite"]].corr().round(3))

print("\nPoint-biserial correlation with default (sign should stay negative, as with source variables):")
for c in ["credit_exposure_composite", "revolving_util_composite"]:
    r, p = pointbiserialr(df["default"], df[c])
    print(f"  {c}: r={r:.4f}, p={p:.6f}")

# --- Build the final consolidated variable list ---
final_corrected = pd.read_csv(os.path.join(OUT, "final_mechanism_design_variables_corrected.csv"))
final_set = set(final_corrected.iloc[:, 0])

CLUSTER_1 = {"avg_cur_bal", "tot_hi_cred_lim"}
CLUSTER_2 = {"total_bc_limit", "bc_open_to_buy", "total_rev_hi_lim"}

consolidated = sorted((final_set - CLUSTER_1 - CLUSTER_2) | {"credit_exposure_composite", "revolving_util_composite"})

print(f"\n{'='*70}")
print("FINAL CONSOLIDATED VARIABLE SET FOR CHAPTER 5 (n={})".format(len(consolidated)))
print(f"{'='*70}")
for v in consolidated:
    print(f"  - {v}")

df.to_csv(os.path.join(OUT, "lendingclub_with_composites.csv"), index=False)
pd.Series(consolidated, name="final_variable").to_csv(
    os.path.join(OUT, "final_consolidated_variables.csv"), index=False
)
print(f"\nSaved dataset with composites to {OUT}/lendingclub_with_composites.csv")
print(f"Saved final consolidated variable list to {OUT}/final_consolidated_variables.csv")
