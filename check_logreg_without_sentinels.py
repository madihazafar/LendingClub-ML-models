"""
OPTIONAL diagnostic: re-fits logistic regression using the binary _ever /
_available flags in place of the raw sentinel-filled (999) numeric columns,
to check whether this resolves the distortion identified in Section 3.3.4
(logistic regression's coefficient ranking being dominated by the sentinel
encoding rather than genuine predictive signal).

This does NOT replace the main pipeline's logistic regression model — it is
a supplementary check you can report alongside the main result to show the
distortion is specifically attributable to the sentinel encoding, and that
requiring majority-vote agreement with SHAP/Random Forest already handles it.

Run this LOCALLY, inside the same folder as your pipeline_outputs/ directory.

Usage:
    python check_logreg_without_sentinels.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

OUT = "pipeline_outputs"

X_train = pd.read_csv(os.path.join(OUT, "X_train.csv"))
X_test = pd.read_csv(os.path.join(OUT, "X_test.csv"))
y_train = pd.read_csv(os.path.join(OUT, "..", "y_test.csv")) if False else None

# We need y_train; if not saved separately, reload from cleaned file using the
# same split logic as the main pipeline (random_state=42, test_size=0.25).
df = pd.read_csv(os.path.join(OUT, "lendingclub_cleaned_full.csv"))
y = df["default"]
X_full = df.drop(columns=["default"])
_, _, y_train, y_test = train_test_split(
    X_full, y, test_size=0.25, random_state=42, stratify=y
)

# Identify the sentinel-affected raw columns (those with a matching _ever flag)
SENTINEL_RAW_COLS = [
    c.replace("_ever", "") for c in X_train.columns if c.endswith("_ever")
]
SENTINEL_RAW_COLS = [c for c in SENTINEL_RAW_COLS if c in X_train.columns]

print(f"Dropping raw sentinel-filled columns, keeping only their _ever flags: {SENTINEL_RAW_COLS}")

X_train_fixed = X_train.drop(columns=SENTINEL_RAW_COLS)
X_test_fixed = X_test.drop(columns=SENTINEL_RAW_COLS)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fixed)
X_test_scaled = scaler.transform(X_test_fixed)

logreg_fixed = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
logreg_fixed.fit(X_train_scaled, y_train)

auc = roc_auc_score(y_test, logreg_fixed.predict_proba(X_test_scaled)[:, 1])
print(f"\nLogistic Regression AUC (sentinel columns removed): {auc:.4f}")

coefs_fixed = pd.Series(
    np.abs(logreg_fixed.coef_[0]), index=X_train_fixed.columns
).sort_values(ascending=False)

print("\nTop 15 by |coefficient|, sentinel raw columns removed:")
print(coefs_fixed.head(15))

coefs_fixed.to_csv(os.path.join(OUT, "logreg_coef_ranking_no_sentinel.csv"))
print(f"\nSaved to {OUT}/logreg_coef_ranking_no_sentinel.csv")

print("\nCompare this ranking to logreg_coef_ranking.csv (the original, with sentinels)")
print("to see how much of the original top-15 was driven by the sentinel encoding.")
