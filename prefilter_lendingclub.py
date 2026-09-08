"""
Run this LOCALLY (not here) on your full LendingClub CSV before uploading.
It only does eligibility filtering and removes obviously unusable columns —
no modeling or cleaning decisions are made here, so it stays faithful to
Section 3.3.1 of the methodology (that step happens transparently later).

Usage:
    python prefilter_lendingclub.py path/to/your_full_file.csv
"""

import sys
import pandas as pd

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "lending_club_full.csv"
OUTPUT_PATH = "lendingclub_filtered.csv"

# Read in chunks to keep memory manageable for a 1.5GB+ file
chunks = []
chunk_iter = pd.read_csv(INPUT_PATH, low_memory=False, chunksize=200_000)

# --- Step 1: keep only loans with a resolved outcome ---
# "Default" is merged into the Charged Off / default class below, since it is
# a small, transitional status that precedes Charged Off in LendingClub's data.
# Current, In Grace Period, and both Late categories are excluded as unresolved.
RESOLVED_STATUSES = ["Fully Paid", "Charged Off", "Default"]

# --- Step 2: drop obvious ID/metadata and post-origination fields ---
# This is a conservative first pass — anything clearly an identifier, URL,
# or a field that is only populated/updated AFTER the loan outcome is known.
DROP_KEYWORDS = [
    "id", "url", "member_id",
    "recoveries", "collection_recovery_fee", "last_pymnt", "next_pymnt",
    "total_pymnt", "total_rec_", "out_prncp", "last_credit_pull",
    "hardship", "settlement", "debt_settlement", "payment_plan",
    "last_fico", "collections_12_mths_ex_med"  # example post-origination/updated field
]

for i, chunk in enumerate(chunk_iter):
    chunk = chunk[chunk["loan_status"].isin(RESOLVED_STATUSES)]

    # Merge "Default" into "Charged Off" as a single default-outcome label.
    # A separate binary "default" column is added here for convenience;
    # loan_status itself is left untouched so the raw label is still visible.
    chunk["default"] = chunk["loan_status"].isin(["Charged Off", "Default"]).astype(int)

    cols_to_drop = [
        c for c in chunk.columns
        if any(kw in c.lower() for kw in DROP_KEYWORDS)
    ]
    chunk = chunk.drop(columns=cols_to_drop, errors="ignore")

    chunks.append(chunk)
    print(f"Processed chunk {i+1}, kept {len(chunk)} rows")

df = pd.concat(chunks, ignore_index=True)

# Drop columns that are >90% missing after filtering (varies by vintage)
missing_frac = df.isna().mean()
mostly_missing = missing_frac[missing_frac > 0.9].index.tolist()
df = df.drop(columns=mostly_missing)

print(f"\nFinal shape: {df.shape}")
print(f"Dropped {len(mostly_missing)} columns for >90% missingness")
print(f"Resolved loan_status counts:\n{df['loan_status'].value_counts()}")
print(f"\nBinary default label counts:\n{df['default'].value_counts()}")

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved filtered file to {OUTPUT_PATH}")
