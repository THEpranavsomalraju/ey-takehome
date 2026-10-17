# Model card: ledger classifier

## Summary

| | |
|---|---|
| Task | Suggest one of six categories (`Category_1` to `Category_6`) for each accounts payable invoice line |
| Model | Blend of logistic regression and LightGBM in log-odds space, temperature-calibrated |
| Holdout accuracy (590 unseen rows, scored once) | **0.9712** (fallback model: 0.9525) |
| Expected accuracy on new data (grouped cross-validation) | **0.954 +- 0.007** |
| Baseline (always the most common category) | 0.884 |
| Review routing (holdout) | 8.8% of rows sent to a person; the rest 98.9% correct |

## Intended use

A first pass for a reviewer. Each invoice line gets a suggested category, a calibrated confidence, a review flag, and an exact explanation. Rows marked `review_required` go to a person, and every other row stays open to review. The model only applies to ledgers anonymized the same way as this one: tokens such as `Word563` mean something only inside this dataset.

## Data

- **Source:** `data/raw/Challenge_Data.xls`, 5,899 invoice lines. The inputs are Col1 vendor, Col2 invoice number, Col3 amount, Col4 line description, Col5 posting date, Col6 GL account, and Col7 document flag; the label is `ClassificationLabel`. Eleven raw label spellings (typos, spacing) map to six categories by one fixed rule.
- **Holdout:** 590 rows (10%), set aside before any analysis by a selection rule written down in advance. Rows that are identical on every model input (with the date reduced to its month) form a group, and no group is split between training and holdout, so the model is never tested on a copy of a training row. The holdout labels were read once, by `evaluate-holdout`.
- **Training rows:** 5,309. Category_1 4,694 (88.4%), Category_2 568 (10.7%), Category_3 20, Category_4 14, Category_6 11, Category_5 2.
- **Label consistency:** rows that agree on vendor, description, account, amount, and document flag almost always share a label; the best possible accuracy from those inputs is 0.9996. Remaining errors therefore come from new or rare combinations, not from contradictory labels.

## Model

Both members read the same scikit-learn feature pipeline (`ledgerclf/features.py`):

- **Features:** a separate TF-IDF vocabulary per text column (vendor, description, account; single words and word pairs); one-hot indicators for recurring whole vendors, accounts, and vendor-account pairs; the document flag; the amount as a signed log plus credit and round-amount flags; the posting month as an index clipped to the training range. Missing amounts and dates are filled with training medians inside the pipeline. Col2 is an identifier and is not used.
- **Logistic regression:** C = 3.0, class weights (n_max / n_c)^0.25.
- **LightGBM:** 800 trees, learning rate 0.1, 31 leaves, minimum 1 row per leaf, subsample 0.7, column sample 0.8, L2 penalty 1, deterministic and single-threaded.
- **Blend:** score = (0.7 x logistic + 0.3 x LightGBM) / 1.392, in log-odds. Because both parts are additive (logistic coefficient x feature value, plus LightGBM's per-feature contributions), every prediction splits exactly into one contribution per input. The largest numerical gap is 7e-12, which is what makes the explanations exact rather than approximate.

## Results

**Cross-validation** (5 grouped folds x 3 seeds on the training rows; "macro F1" averages the F1 score of the five classes with 11 or more rows, so rare classes count as much as common ones):

| Measure | Final model | Fallback (logistic) | Majority baseline |
|---|---|---|---|
| Accuracy | 0.9539 +- 0.0068 | 0.9513 +- 0.0082 | 0.8842 |
| Macro F1 | 0.7475 | 0.6887 | 0.1877 |
| Out-of-time accuracy (train to 2018-09, test later months) | 0.9403 | 0.9442 | - |
| Out-of-time macro F1 | 0.8743 | 0.6784 | - |
| Nested CV accuracy (tuning repeated inside each outer fold) | 0.9548 +- 0.0050 | - | - |

A random split would report 0.9636 for the logistic model against 0.9493 with grouped folds. The 1.4-point gap is the inflation that grouping removes. On vendors never seen in training, accuracy is 0.913 (unseen-vendor CV).

**Holdout** (590 rows, scored once on 2026-10-17; `reports/holdout/`):

| Measure | Final model | Fallback | Cross-validation (final model) |
|---|---|---|---|
| Accuracy | **0.9712** | 0.9525 | 0.9539 |
| Macro F1 | 0.7068 | 0.6880 | 0.7475 |
| Expected calibration error | 0.011 | 0.023 | 0.0066 |
| Rows sent to review | 8.8% | 8.8% | 7.3% |
| Accuracy on rows not sent to review | 0.9888 | 0.9888 | 0.9791 |

| Class (final model) | Precision | Recall | Holdout rows | Cross-validation recall | Training rows |
|---|---|---|---|---|---|
| Category_1 | 0.977 | 0.990 | 521 | 0.979 | 4,694 |
| Category_2 | 0.930 | 0.841 | 63 | 0.773 | 568 |
| Category_3 | 1.000 | 1.000 | 3 | 0.950 | 20 |
| Category_4 | 0.000 | 0.000 | 2 | 0.500 | 14 |
| Category_5 | - | - | 0 | 0.000 | 2 |
| Category_6 | 0.500 | 1.000 | 1 | 0.364 | 11 |

The holdout accuracy is about 2 standard errors above the cross-validation estimate. Two effects explain most of it. First, the final model learned from all 5,309 training rows, while each cross-validation model saw about 4,250. Second, the holdout has slightly fewer unfamiliar rows: 2.0% new vendors against 2.9% in cross-validation, and 5.8% new vendor-account pairs against 7.1%. Training and holdout share no group and no identical row. **The cross-validation figure (about 95%) is the honest expectation for new data.** With 1 to 3 holdout rows per rare class, the holdout macro F1 is too noisy to compare closely.

## Calibration and review routing

- **Calibration:** one temperature (1.392) is fitted on out-of-fold scores, so that a stated confidence matches the observed accuracy. Expected calibration error falls from 0.019 to 0.0066. Dividing every score by one number changes neither the predicted category nor the exact explanations.
- **Review threshold: 0.8.** This is the lowest confidence at which automatically decided rows reach 97.5% accuracy out of fold. Out of fold, 92.7% of rows fall above it at 97.9% accuracy, and the 7.3% below it are only 63.4% accurate.
- **Flags:** a flag becomes critical (review required) when its rows are at least 5 points less accurate than average out of fold. Out-of-fold accuracy: new vendor 0.896, new GL account 0.893, low confidence 0.634; rows with no flag 0.982. Flags mark risk; they never withhold a prediction.

## Key design choices

| Choice | Why | Evidence |
|---|---|---|
| Grouped splits and grouped cross-validation everywhere | 26% of training rows duplicate another row except for the invoice number; random splits test the model on copies it memorized | A random split inflates accuracy by 1.4 points (notebook 03, section 1) |
| Final model chosen by a rule fixed before tuning | Prevents picking whatever looks best after the fact | The decision trail is in `reports/selection/champion_decision.json` |
| Log-odds blend instead of averaging probabilities | Exact additive explanations, and slightly more accurate | +0.0008 accuracy over a probability average with identical members |
| Mild class weights (power 0.25) | Rare-class recall without false rare alarms on common rows | Macro F1 0.70 to 0.75 for the blend; false rare alarms stay at 1 (notebook 03, section 4) |
| Every transformer fitted inside each training fold | No statistic leaks from test rows into training | A unit test checks that the fitted vocabulary comes only from the rows it was fitted on |
| Col2 (invoice number) dropped | An identifier that never recurs, partly corrupted (`4.80P+11`) | Format features add less than one fold's standard deviation |
| Per-column TF-IDF on anonymized tokens, no pretrained language model | Tokens have no meaning a pretrained model could use; the same token means different things in different columns | Description words alone carry 1.7 accuracy points (ablation) |
| Outliers kept, amounts as a signed log | Large invoices are real; credits must stay negative | 227 of 235 outliers are ordinary Category_1 rows |
| Calibrated confidence plus a review threshold | Lets a reviewer trust "auto" rows and focus on risky ones | Holdout: auto rows 98.9% correct, reviewed rows 78.9% |
| Holdout labels guarded in the dashboard | A copy of the holdout (even re-saved or with Col2 removed) must not be scored before the official evaluation | Row fingerprints on model inputs; tests cover re-saved and edited copies |

## Limitations

- **Rare classes:** Category_3 to Category_6 have 2 to 20 training rows each. Their recall is unstable across folds, and Category_5 (2 rows) is essentially never predicted. Results are therefore reported as accuracy and macro F1 side by side.
- **New vendors and accounts:** accuracy falls to about 0.89 on rows whose vendor or account never appeared in training. Such rows are flagged for review.
- **Anonymized tokens:** the model learns token identities, not meanings. Real vendor names, or the same ledger anonymized differently, would need retraining.
- **Time range:** training covers 2016-10 to 2018-12, mostly 2018. Dates outside that range are clipped to it and flagged.
- **Labeled uploads that overlap training:** accuracy on rows the model was trained on is close to 100%. The dashboard warns when an upload contains such rows.

## Fallback behavior

If LightGBM cannot load (for example on macOS without `libomp`), or `LEDGERCLF_FORCE_FALLBACK=1` is set, the service loads `artifacts/model_fallback.joblib`. It is a temperature-calibrated logistic regression with the same features and no LightGBM parts: 0.9513 cross-validation accuracy and 0.9525 on the holdout. Every prediction and explanation states which model produced it, and why the fallback is active.

## Reproducibility

Library versions are pinned exactly (`requirements.txt`) and recorded in `artifacts/metadata.json`. Seeds are fixed and LightGBM runs deterministically. `python -m ledgerclf train` refits both models and rewrites `artifacts/golden_holdout_predictions.csv`; `python tools/compare_golden.py` compares a fresh run against it. On the same machine the largest probability difference is 5e-11.
