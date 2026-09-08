# On-Chain Reputation Mechanism Design — Empirical Pipeline

This repository contains the data-processing, modelling, and simulation code supporting the
dissertation "Does social collateral, the trust-based substitute for traditional creditworthiness established in microfinance theory, generalise to decentralised digital lending contexts, and what design conditions allow it to remain robust against strategic identity manipulation?". It accompanies Chapters 3–6 (Methodology, Findings, Mechanism Design and Simulation) and is provided for transparency and reproducibility.

## Data source

This project uses the LendingClub loan dataset published on Kaggle:

> LendingClub (2020) *All Lending Club loan data*, uploaded by wordsforthewise, available at:
> https://www.kaggle.com/datasets/wordsforthewise/lending-club

The raw dataset is **not included in this repository** (file sizes exceed GitHub's limits, and
redistribution is restricted under Kaggle's dataset terms). To reproduce the pipeline, download
the dataset from the link above and place the raw CSV in a local `data/` folder before running
the scripts below.

## Pipeline run order

The scripts must be run in this order — several depend on files produced by earlier steps.

| Step | Script | Input | Key output |
|---|---|---|---|
| 1 | `prefilter_lendingclub.py` | raw Kaggle CSV | `lendingclub_filtered.csv` (resolved-outcome loans only) |
| 2 *(optional)* | `shrink_for_upload.py` | `lendingclub_filtered.csv` | stratified sample, for quick local iteration |
| 3 | `run_full_pipeline.py` | `lendingclub_filtered.csv` | `pipeline_outputs/` — cleaned data, model comparison (3.3.2), SHAP rankings (3.3.3, computed on the training set), Tier C ablation (3.4.1), verification/emp_length checks |
| 4 | `check_logreg_without_sentinels.py` | `pipeline_outputs/` | corrected logistic-regression coefficient ranking, with sentinel-encoded columns removed |
| 5 | `finalize_variable_selection.py` | `pipeline_outputs/` | original (uncorrected) 3-model majority-vote selection — kept for provenance; superseded by step 6 |
| 6 | `finalize_variable_selection_corrected.py` | outputs of steps 3–5 | corrected majority-vote selection using the sentinel-free logistic regression ranking (3.3.4) |
| 7 | `build_composite_variables.py` | outputs of step 6 | `credit_exposure_composite`, `revolving_util_composite`; final 10-variable mechanism-design set |
| 8 | `generate_final_variable_dependence_plots.py` | `lendingclub_with_composites.csv` | SHAP dependence plots for the 7 variables entering the reputation score (Section 5.7) |
| 9 *(robustness checks, either order)* | `test_endogenous_gap.py` | `lendingclub_filtered.csv` | tests whether reinstating platform-assigned variables (interest rate, grade) changes the logistic regression–XGBoost performance gap (Section 4.4) |
| 9 | `test_verification_status.py` | `lendingclub_filtered.csv` | tests the raw default rate, predictive contribution, and income interaction of verification status (Section 4.7) |

Each script prints its key results to the console and writes outputs to `pipeline_outputs/`
(created automatically). Running the full sequence on the complete dataset (~612,000 rows) takes
several minutes; the sample from step 2 is recommended for testing changes before a full run.

## Simulation (Chapter 6)

`dissertation_simulation.py` is self-contained and does not require the LendingClub
data or the outputs of steps 1–9 above. It implements the agent-based model described in Section
6.3, testing the manipulation-resistance property established analytically in Section 5.11
(`K(h) ≥ B(R)`) across the minimum-history parameter `h`.

## Requirements

See `requirements.txt`. Install with:

```
pip install -r requirements.txt
```

## Notes on reproducibility

- All scripts use `random_state=42` (or equivalent seeded RNGs) for train/test splits and model
  fitting, so results should be exactly reproducible given the same input data.
- SHAP values used for variable selection (steps 3, 6, 7) are computed on the **training set**
  only, to avoid the held-out test set informing the variable-selection process (see Section
  3.3.3 for the methodological rationale). A supplementary, purely interpretive test-set SHAP
  computation is also produced by step 3 for comparison, but is not used in any selection
  decision.
- The exact row/column counts and headline results (AUC, SHAP rankings, final variable set)
  reported in Chapter 4 were produced by running steps 1–9 on the full dataset; sample-based runs
  are for development purposes only and should not be cited as the dissertation's reported
  findings.
