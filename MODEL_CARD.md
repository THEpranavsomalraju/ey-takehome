# Model card: ledger classifier

| | |
|---|---|
| Task | Suggest one of six categories (`Category_1` to `Category_6`) for each accounts payable invoice line |
| Model | Log-odds blend of logistic regression (0.7) and LightGBM (0.3), temperature-calibrated |
| Holdout accuracy | **0.9712** on 590 unseen lines, scored once (fallback model: 0.9525) |
| Expected accuracy on new data | **0.954 ± 0.007** (grouped cross-validation) |
| Baseline | 0.884 (always the most common category) |
| Review | 8.8% of holdout lines sent to a person; the rest 98.9% correct |

## Intended use

A first pass for a reviewer. Each line gets a category, a calibrated confidence, a review flag, and an explanation. It applies only to ledgers anonymized the same way: codes such as `Word563` mean nothing outside this dataset.

## Data

- `data/raw/Challenge_Data.xls`: 5,899 lines. Inputs: vendor (Col1), invoice number (Col2), amount (Col3), description (Col4), posting date (Col5), GL account (Col6), document flag (Col7). Label: `ClassificationLabel`, 11 raw spellings mapped to 6 categories.
- **Holdout:** 590 lines (10%), chosen before any analysis by a rule written down in advance. Lines identical on every model input form a group, and no group is split between training and holdout.
- **Training:** 5,309 lines. Category_1 88.4%, Category_2 10.7%, then 20, 14, 2, and 11 lines for Categories 3, 4, 5, and 6.

## Model

- **Features** (one scikit-learn pipeline, fitted on training rows only): TF-IDF per text column (single words and word pairs), one-hot indicators for recurring vendors, accounts, and vendor-account pairs, the document flag, signed-log amount with credit and round-amount flags, and the posting month clipped to the training range. Col2 is not used.
- **Logistic regression:** C = 3, class weights (n_max / n_c)^0.25.
- **LightGBM:** 800 trees, learning rate 0.1, 31 leaves, minimum 1 row per leaf, subsample 0.7, column sample 0.8, L2 penalty 1; deterministic.
- **Blend:** (0.7 × logistic + 0.3 × LightGBM) / 1.392 in log-odds. Both parts are additive, so each prediction splits exactly into per-input contributions; the explanations use these.

## Results

| | Final model | Fallback (logistic) | Baseline |
|---|---|---|---|
| Grouped CV accuracy (3 seeds × 5 folds) | 0.9539 ± 0.0068 | 0.9513 ± 0.0082 | 0.8842 |
| Grouped CV macro F1 (5 categories) | 0.7475 | 0.6887 | 0.1877 |
| Out-of-time accuracy (train to 2018-09, test later) | 0.9403 | 0.9442 | |
| Nested CV accuracy (tuning repeated per fold) | 0.9548 ± 0.0050 | | |
| **Holdout accuracy** | **0.9712** | 0.9525 | |
| Holdout macro F1 | 0.7068 | 0.6880 | |
| Holdout calibration error | 0.011 | 0.023 | |

A random split would report 0.964 for the logistic model against 0.949 with grouped folds: the 1.4 points are copies of test rows leaking into training. On vendors never seen in training, accuracy is 0.913.

| Category | Training lines | Holdout lines | Holdout recall | Holdout precision | CV recall |
|---|---|---|---|---|---|
| Category_1 | 4,694 | 521 | 0.990 | 0.977 | 0.979 |
| Category_2 | 568 | 63 | 0.841 | 0.930 | 0.773 |
| Category_3 | 20 | 3 | 1.000 | 1.000 | 0.950 |
| Category_4 | 14 | 2 | 0.000 | 0.000 | 0.500 |
| Category_5 | 2 | 0 | | | 0.000 |
| Category_6 | 11 | 1 | 1.000 | 0.500 | 0.364 |

The holdout is about 2 standard errors above cross-validation. The final model trained on all 5,309 lines (each CV model on about 4,250), and the holdout has fewer unfamiliar lines (2.0% new vendors against 2.9%). Training and holdout share no group. Cross-validation (about 95%) is the honest expectation for new data.

## Confidence and review

- **Calibration:** one temperature (1.392), fitted out of fold, cuts the calibration error from 0.019 to 0.0066 without changing any prediction.
- **Threshold 0.8:** the lowest confidence at which automatic lines reach 97.5% accuracy out of fold. 92.7% of lines pass it at 97.9% accuracy; the 7.3% below it are 63.4% accurate.
- **Warnings:** a warning sends the line to review when its lines are at least 5 points less accurate than average out of fold: new vendor (0.896), new GL account (0.893), low confidence (0.634). Lines with no warning are 0.982 accurate.

## Key design choices

| Choice | Reason | Evidence |
|---|---|---|
| Grouped splits everywhere | 26% of lines repeat another line apart from the invoice number | Random split inflates accuracy by 1.4 points |
| Model chosen by a rule fixed before tuning | No picking the best-looking result after the fact | `reports/selection/champion_decision.json` |
| Log-odds blend | Exact explanations, slightly more accurate than averaging probabilities | +0.0008 accuracy |
| Mild class weights (power 0.25) | Rare categories found without false alarms on common lines | Macro F1 0.70 → 0.75; false alarms unchanged |
| No pretrained language model | Text is anonymized codes with no meaning to transfer | Description words alone carry 1.7 accuracy points |
| Outliers kept | Large invoices are real | 227 of 235 outliers are ordinary Category_1 lines |
| Holdout guarded in the dashboard | Holdout labels must not be scored before the one official run, even in a re-saved copy | Row fingerprints on model inputs; tested |

## Limitations

- **Rare categories:** 2 to 20 training lines each. Their scores swing with single lines, and Category_5 (2 lines) is essentially never predicted.
- **New vendors and accounts:** accuracy drops to about 0.89; these lines are flagged for review.
- **Anonymized text:** the model learns codes, not meanings. A differently anonymized ledger needs retraining.
- **Time range:** 2016-10 to 2018-12, mostly 2018. Later dates are clipped and flagged.
- **Category meaning:** the data does not say what each category represents.

## Fallback and reproducibility

If LightGBM cannot load, the app uses `artifacts/model_fallback.joblib`, a calibrated logistic regression on the same features (0.9525 on the holdout), and says so on every prediction. Versions are pinned in `requirements.txt` and seeds are fixed; `python -m ledgerclf train` followed by `python tools/compare_golden.py` reproduces the shipped predictions (largest difference 5e-11 on the same machine).
