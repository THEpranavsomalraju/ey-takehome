# Model card: ledger classifier

## Intended use

Suggest one of six classes (`Category_1` to `Category_6`) for each line of an anonymized accounts payable ledger, with a calibrated confidence, a review flag, and an exact explanation. It is meant as a first pass that a reviewer checks: rows marked `review_required` go to a person, and every other row is still open to review. The model is not meant for ledgers whose vendors, accounts, and descriptions use a different anonymization scheme. Tokens such as `Word563` only mean something inside this dataset.

## Data

- Source: `data/raw/Challenge_Data.xls`, 5,899 rows, inputs `Col1` to `Col7`, label `ClassificationLabel`. Eleven raw label spellings map to six classes (DECISIONS.md D01).
- Holdout: 590 rows (10%), chosen by a seed rule written down before any modeling (D03). The two sides share no group. A group is a set of rows identical on every model input (vendor, amount, description, month, account, document type), so near-copies never sit on both sides (D02, D22). Holdout labels are read only by `evaluate-holdout`.
- Training portion: 5,309 rows. Class counts: Category_1 4,694 (88.4%), Category_2 568 (10.7%), Category_3 20, Category_4 14, Category_6 11, Category_5 2.
- Exact-duplicate structure: the label is almost fixed by vendor, description, account, amount, and document type. The deterministic ceiling is 0.9996 (`reports/eda/accuracy_ceiling.json`), so most of the remaining error comes from new or rare combinations, not from label noise.

## Model

The model is a log-odds blend of two members. Both use the same scikit-learn feature pipeline (`ledgerclf/features.py`):

- **Features:** per-column TF-IDF on word tokens (1- and 2-grams), whole-value and vendor-account pair one-hots, document type, signed log amount, amount flags (credit, round), and a period index clipped to the training range. Missing amounts and dates are imputed with training medians inside the pipeline.
- **Logistic regression:** C = 3.0, with class weights (n_max / n_c)^0.25.
- **LightGBM:** 800 trees, learning rate 0.1, 31 leaves, min_child_samples 1, subsample 0.7, colsample 0.8, reg_lambda 1. It runs with `deterministic=True` and `n_jobs=1`.
- **Blend:** logits = 0.7 x logistic + 0.3 x LightGBM, divided by temperature 1.392. Contributions stay exactly additive: logistic coefficient x feature value, plus LightGBM `pred_contrib`, add up to the logit. The maximum gap is 7e-12, which is what makes the per-row explanations exact.
- **Selection:** the champion was chosen by a rule fixed before tuning (D11). It takes the best grouped-CV accuracy among candidates that pass a macro-F1 floor and an out-of-time floor. The highest-accuracy candidate without class weights (blend p0) failed the macro-F1 floor.

## Metrics (training portion; holdout below)

Every CV uses `StratifiedGroupKFold` on the group key, with 5 folds and 3 seeds.

| Measure | Champion (blend p0.25) | Fallback (logistic p0) | Majority baseline |
|---|---|---|---|
| Grouped CV accuracy | 0.9539 +- 0.0068 | 0.9513 +- 0.0082 | 0.8842 |
| Grouped CV macro F1 (5 classes, Category_5 excluded) | 0.7475 | 0.6887 | 0.1877 |
| Out-of-time accuracy (train to 2018-09, test after) | 0.9403 | 0.9442 | - |
| Out-of-time macro F1 (5 classes) | 0.8743 | 0.6784 | - |
| Nested CV accuracy (tuning inside each outer fold) | 0.9548 +- 0.0050 | - | - |

- **Validation schemes** (logistic, `reports/benchmark/validation_schemes.csv`): a random split gives 0.9636, while grouped CV gives 0.9493. The 1.4-point gap is the leakage a random split would hide. Unseen-vendor CV gives 0.9134.
- **Out-of-fold per class** (champion, `reports/selection/oof_classification_report.csv`):

| Class | Precision | Recall | Support |
|---|---|---|---|
| Category_1 | 0.971 | 0.979 | 4,694 |
| Category_2 | 0.830 | 0.773 | 568 |
| Category_3 | 0.826 | 0.950 | 20 |
| Category_4 | 0.438 | 0.500 | 14 |
| Category_5 | 0.000 | 0.000 | 2 |
| Category_6 | 1.000 | 0.364 | 11 |

## Calibration and review routing

- **Temperature scaling:** fitted on out-of-fold logits. Expected calibration error drops from 0.019 to 0.0066 and Brier score from 0.0690 to 0.0667 (`reports/selection/calibration_summary.json`). Scaling divides every logit by the same number, so contributions stay additive.
- **Review threshold:** 0.8, chosen by rule D17 (`reports/selection/threshold_table.csv`). In out-of-fold data, 92.7% of rows fall above it with 97.9% accuracy, and the 7.3% below it have 63.4% accuracy.
- **Flags:** each flag's severity comes from out-of-fold accuracy (D18). A critical flag sends the row to review: unseen vendor 0.896, unseen account 0.893, low confidence 0.634. Rows with no flag reach 0.982. Flags never withhold a prediction.

## Holdout evaluation

Pending. It runs once, after the project owner approves, with `python -m ledgerclf evaluate-holdout --confirm`. Results go to `reports/holdout/` together with the CV estimate beside each number.

## Limitations

- **Rare classes:** Category_3 to Category_6 have 2 to 20 training rows each, and Category_5 has only 2. Recall on them is unstable across folds, and Category_5 is almost never predicted. Macro F1 therefore excludes Category_5 and is reported alongside accuracy.
- **Unseen vendors and accounts:** accuracy falls to about 0.89 on rows whose vendor or account was not seen in training. Such rows carry a critical flag.
- **Anonymized tokens:** the model learns token identity, not meaning. A real vendor name or a new anonymization run would need retraining.
- **Time drift:** the data covers one period range. Dates outside it are clipped and flagged.
- **In-sample uploads:** labeled rows that were part of training score near 100%. The service warns when an upload contains them (D35).

## Fallback behavior

If LightGBM cannot load (for example macOS without `libomp`), or `LEDGERCLF_FORCE_FALLBACK=1` is set, the service loads `model_fallback.joblib`. This is a temperature-calibrated logistic regression with the same features and no LightGBM objects. It gives a grouped CV accuracy of 0.9513. Every prediction and explanation reports which model produced it and why the fallback is active.

## Reproducibility

`python -m ledgerclf train` refits the bundle on the training portion and writes `artifacts/golden_holdout_predictions.csv`. `python tools/compare_golden.py` checks a fresh run against that file. On the same machine the maximum probability difference is 5e-11. Library versions are pinned exactly (`requirements.txt`) and recorded in `artifacts/metadata.json`.
