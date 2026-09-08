"""
Run this LOCALLY on lendingclub_filtered.csv to shrink it further for upload.
Does two things only:
  1. Drops a few more columns unlikely to be useful for modeling
     (free-text, high-cardinality identifiers) — does NOT drop anything
     that could plausibly be a predictor.
  2. Takes a stratified random sample, preserving the 79/21 default split
     exactly, so the smaller file is still representative.

Usage:
    python shrink_for_upload.py lendingclub_filtered.csv
"""

import sys
import pandas as pd

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "lendingclub_filtered.csv"
OUTPUT_PATH = "lendingclub_sample.csv"

# Adjust this if you still want it smaller/larger after seeing the result
SAMPLE_SIZE = 150_000

df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded: {df.shape}")

# Columns very unlikely to be predictive and often high-cardinality/free-text
# (only drop if present — safe no-op if a name doesn't exist in your file)
EXTRA_DROP = [
    "emp_title",       # free-text job title, thousands of unique values
    "title",           # free-text loan title, mostly redundant with purpose
    "desc",            # free-text borrower description
    "zip_code",        # high-cardinality; state (addr_state) is usually kept instead
    "url",
]
existing_extra_drop = [c for c in EXTRA_DROP if c in df.columns]
df = df.drop(columns=existing_extra_drop)
print(f"Dropped: {existing_extra_drop}")

# Stratified sample preserving the exact default/non-default ratio.
# Using groupby(...).sample() directly (rather than groupby(...).apply(...))
# avoids a pandas version issue where the grouping column gets dropped.
sample = df.groupby("default", group_keys=False).sample(
    frac=SAMPLE_SIZE / len(df), random_state=42
)

print(f"\nSample shape: {sample.shape}")
print(f"Sample default balance:\n{sample['default'].value_counts(normalize=True)}")

sample.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")
