# 01. Data Findings

Verified facts about `data/raw/Challenge_Data.xls` and the results of exploratory experiments. Treat every number here as a reference to reproduce, not as a value to hardcode. Section 15 lists tolerances.

Provenance note: all exploration below ran on the full labeled file before any holdout existed. The build re-runs every analysis on the 90% training portion. DECISIONS.md entry D00 records this.

## 1. Source file

| Property | Value |
|---|---|
| File | `Challenge_Data.xls` (legacy Excel 97-2003, OLE2 container) |
| Reader | `pandas.read_excel(path, engine="xlrd")`, requires `xlrd>=2.0.1`. openpyxl rejects `.xls`. |
| Sheets | One sheet: `Challenge_Data` |
| Shape | 5,899 rows by 8 columns |
| Header row | Row 1: `Col1, Col2, Col3, Col4, Col5, Col6, Col7, ClassificationLabel` |

## 2. Schema and inferred roles

Column roles come from value patterns. The file names only `Col1` to `Col7`. Treat the roles as working hypotheses and say so in reports.

| Column | Parsed dtype | Unique | Missing | Inferred role | Notes |
|---|---|---|---|---|---|
| Col1 | text | 313 | 0 | Vendor name | Anonymized tokens such as `Word16 Word17 Word18 Word7`. Last token often looks like a legal suffix (`Word5`, `Word7`, `Word4`, `Word39`). |
| Col2 | mixed | 4,818 | 0 | Invoice number | 3,526 ints, 2,370 strings, 3 floats. Corrupted values. Section 6. |
| Col3 | float | 2,397 | 0 | Amount | Heavy right skew, 69 negatives, 2 zeros. Section 7. |
| Col4 | text | 2,032 | 153 | Line description | Most varied text column. Section 9. |
| Col5 | datetime | 17 | 0 | Posting period | Month-end or month-start dates only. Section 8. |
| Col6 | text | 196 | 0 | GL account name | Section 9. |
| Col7 | text | 4 | 0 | Document flag | `NoDoc`, `Doc1`, `Doc2`, `Doc3`. Section 10. |
| ClassificationLabel | text | 11 raw | 0 | Target | Collapses to 6 classes. Section 3. |

All text tokens follow the pattern `Word<N>`. Tokens carry no meaning outside this dataset. Pretrained embeddings and LLM semantics add nothing.

## 3. Label cleanup

Normalization rule: lowercase, remove spaces and underscores, fix the typo `categry` to `category`, extract the digits, map to `Category_<digit>`.

| Raw value | Rows | Canonical |
|---|---|---|
| `Category_1` | 5,204 | Category_1 |
| `category_1` | 11 | Category_1 |
| `Category_2` | 627 | Category_2 |
| `Category2` | 4 | Category_2 |
| `Category_3` | 12 | Category_3 |
| `Category 3` | 10 | Category_3 |
| `Category _3` | 1 | Category_3 |
| `Category4` | 16 | Category_4 |
| `Category 5` | 2 | Category_5 |
| `Categry_6` | 10 | Category_6 |
| `Category_6` | 2 | Category_6 |

Canonical class counts:

| Class | Rows | Share |
|---|---|---|
| Category_1 | 5,215 | 88.40% |
| Category_2 | 631 | 10.70% |
| Category_3 | 23 | 0.39% |
| Category_4 | 16 | 0.27% |
| Category_6 | 12 | 0.20% |
| Category_5 | 2 | 0.03% |

Short names used in this document: C1 to C6.

## 4. Class balance consequences

- Majority baseline (always C1): accuracy 0.8840, six-class macro F1 0.1564, five-class macro F1 0.1877.
- One C5 row moves six-class macro F1 by up to 0.167. Report macro F1 over the five classes with 12 or more rows (C1, C2, C3, C4, C6), called `macro_f1_5` below, and report C5 separately.
- With 12 to 23 rows per rare class, one row moves rare-class recall by 4 to 8 points in CV.

## 5. Missing values

- Only Col4 has missing values: 153 rows (2.6%). Classes of those rows: C1 140, C2 13.
- No other column has missing values in the training file. Uploaded files will. See `docs/05`.

## 6. Col2 (invoice number)

- 4,818 unique values in 5,899 rows. 1,652 rows share a Col2 value with at least one other row.
- Python types after parsing: 3,526 int, 2,370 str, 3 float (`108542.1`, `111692.2`, `108543.1`).
- 326 values show Excel scientific notation with a letter swapped in for `E`: `4.80P+11` (22), `4.80W+11` (19), `4.80J+11` (19), `4.80F+11` (17), and others. Regex: `^\d\.\d+[A-Za-z][+-]\d+$`.
- Some prefixes look synthetic: vendor `Word137 Word138 Word7` carries `ARBG5026`, `ICMH5026`, `PCBP5026` on otherwise identical rows.
- Top format patterns (digit to 9, letter to A): `9999` (1,189), `99999` (1,149), `999999` (676), `9.99A+99` (326), `AA9999` (326), `99999999` (242), `AA999` (220), `AAA99999` (146), `999999-99999` (116).
- Decision: drop Col2 from the model. Format features (length, letters, hyphen, digit share, scientific flag) did not help (Section 14, Exp 4). Keep Col2 as a string for display only.

## 7. Col3 (amount)

| Statistic | Value |
|---|---|
| Min | -269,111.60 |
| Median | 1,093.20 |
| Mean | 135,878.70 |
| Max | 8,799,523.60 |
| Negative rows | 69 (C1: 67, C5: 2) |
| Zero rows | 2 |

Amount by class:

| Class | Count | Median | Mean | Min | Max |
|---|---|---|---|---|---|
| C1 | 5,215 | 1,200.0 | 152,951.7 | -269,111.6 | 8,799,523.6 |
| C2 | 631 | 617.6 | 5,666.8 | 7.5 | 275,188.1 |
| C3 | 23 | 1,680.0 | 5,880.9 | 703.6 | 51,073.0 |
| C4 | 16 | 8,018.9 | 11,043.4 | 1,500.0 | 44,543.8 |
| C5 | 2 | -2,718.3 | -2,718.3 | -4,645.9 | -790.7 |
| C6 | 12 | 1,440.0 | 1,912.3 | 119.6 | 9,820.0 |

- Round amounts recur in rare classes: 1,680 and 1,500 appear in C3 and C4.
- Extremes look like real ledger values. Do not clip or drop outliers. Use a signed log transform.

## 8. Col5 (posting period)

- 17 unique dates, from 2016-10-01 to 2018-12-31.
- Year counts: 2016: 36, 2017: 232, 2018: 5,631 (95.5%).
- Day of month: 31st 2,365, 1st 1,890, 30th 1,644. Treat Col5 as a monthly period, not a calendar date.

Month by class (full file):

| Period | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| 2016-10 | 36 | 0 | 0 | 0 | 0 | 0 |
| 2017-04 | 2 | 0 | 0 | 0 | 0 | 0 |
| 2017-10 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2017-12 | 176 | 51 | 0 | 2 | 0 | 0 |
| 2018-01 | 283 | 27 | 0 | 2 | 0 | 0 |
| 2018-02 | 484 | 48 | 0 | 1 | 0 | 0 |
| 2018-03 | 440 | 61 | 0 | 1 | 0 | 0 |
| 2018-04 | 481 | 43 | 0 | 1 | 0 | 3 |
| 2018-05 | 474 | 45 | 6 | 0 | 0 | 1 |
| 2018-06 | 468 | 61 | 3 | 1 | 0 | 2 |
| 2018-07 | 425 | 64 | 3 | 4 | 0 | 0 |
| 2018-08 | 498 | 77 | 4 | 0 | 0 | 1 |
| 2018-09 | 405 | 33 | 2 | 2 | 0 | 2 |
| 2018-10 | 571 | 74 | 1 | 1 | 2 | 2 |
| 2018-11 | 363 | 44 | 4 | 0 | 0 | 1 |
| 2018-12 | 108 | 3 | 0 | 1 | 0 | 0 |

## 9. Text columns (Col1, Col4, Col6)

| Column | Unique values | Token vocabulary | Mean tokens per row | Max tokens | Values seen once | In-sample majority purity |
|---|---|---|---|---|---|---|
| Col1 | 313 | 559 | 3.97 | 10 | 108 | 0.9339 |
| Col4 | 2,032 (+153 missing) | 1,569 | 2.75 | 9 | 1,335 | 0.9858 (inflated by singletons) |
| Col6 | 196 | 275 | 3.75 | 7 | 52 | 0.9239 |

Purity means: predict the majority class per unique value, scored in-sample. Optimistic by design.

Token overlap across columns: Col1 and Col4 share 116 tokens, Col1 and Col6 share 64, Col4 and Col6 share 144. Example: `Word562` appears in Col4 and in Col6. Use one vectorizer per column, so a token in Col4 and the same token in Col6 become different features.

Top Col1 values:

| Col1 | C1 | C2 | Total |
|---|---|---|---|
| Word16 Word17 Word18 Word7 | 1,240 | 5 | 1,245 |
| Word8 Word9 Word10 Word11 Word5 | 323 | 0 | 323 |
| Word30 Word31 | 264 | 0 | 264 |
| Word19 Word20 Word21 Word22 Word23 Word5 | 13 | 190 | 203 |
| Word16 Word17 Word3 Word7 | 111 | 87 | 198 |
| Word85 Word86 Word87 Word5 Word88 Word16 Word17 ... | 100 | 96 | 196 |
| Word33 Word34 Word35 Word7 | 173 | 0 | 173 |
| Word42 Word43 Word5 | 149 | 14 | 163 |

Top Col6 values:

| Col6 | C1 | C2 | Total |
|---|---|---|---|
| Word563 Word564 Word9 Word26 | 2,160 | 5 | 2,165 |
| Word563 Word580 Word581 Word232 | 369 | 137 | 506 |
| Word563 Word366 Word562 | 302 | 136 | 438 |
| Word563 Word575 | 31 | 195 | 226 |
| Word567 Word568 Word569 Word570 | 187 | 0 | 187 |
| Word563 Word121 Word579 | 183 | 0 | 183 |
| Word563 Word366 Word56 Word4 | 155 | 0 | 155 |
| Word563 Word132 Word610 | 129 | 10 | 139 |

Top Col4 values:

| Col4 | C1 | C2 | Total |
|---|---|---|---|
| Word61 | 220 | 0 | 220 |
| Word562 | 158 | 33 | 191 |
| (missing) | 140 | 13 | 153 |
| Word783 Word784 Word785 | 148 | 0 | 148 |
| Word826 Word780 | 95 | 0 | 95 |

Some vendors split across classes (`Word16 Word17 Word3 Word7`: 111 C1, 87 C2). Col4 and Col6 resolve most of those cases.

## 10. Col7 (document flag)

| Col7 | C1 | C2 | C3 | C4 | C5 | C6 | Total |
|---|---|---|---|---|---|---|---|
| Doc1 | 1,067 | 40 | 0 | 0 | 1 | 0 | 1,108 |
| Doc2 | 33 | 1 | 0 | 1 | 0 | 0 | 35 |
| Doc3 | 95 | 15 | 0 | 11 | 0 | 0 | 121 |
| NoDoc | 4,020 | 575 | 23 | 4 | 1 | 12 | 4,635 |

Doc3 holds 11 of the 16 C4 rows. Ablation showed Col7 adds nothing measurable once text is present.

## 11. Duplicates and label conflicts

- Exact duplicates on Col1 to Col7: 680 extra rows (11.5%) in 535 groups.
- Group size distribution: size 1: 4,684 groups, size 2: 419, size 3: 92, size 4: 19, size 5: 5.
- Rows inside duplicate groups by class: C1 1,002, C2 213.
- Ignoring Col2: 1,508 duplicate rows (25.6%) in 913 groups. Many copies differ only by invoice number.
- One conflicting group: 5 identical rows (Col1 `Word42 Word43 Word5`, Col2 `8229`, Col3 1,510.0, Col4 `Word562`, Col5 2018-07-31, Col6 `Word563 Word580 Word581 Word232`, Col7 `NoDoc`) carry 3 C1 and 2 C2 labels.
- Group key for all splits: every input column except Col2 (`Col1, Col3, Col4, Col5, Col6, Col7`, with missing Col4 filled by a fixed token before keying).

## 12. Rare-class patterns

- **C3 (23 rows):** 18 rows from vendor `Word137 Word138 Word7`, amount 1,680, Col6 `Word563 Word359`, NoDoc. The other 5 come from accounts starting `Word571 Word52 Word457`.
- **C4 (16 rows):** 11 rows from vendor `Word101 Word102 Word103 Word7`, Col7 Doc3, Col6 starting `Word576`. 3 rows from vendor `Word137 Word138 Word7` in 2017-12 and 2018-01.
- **C5 (2 rows):** vendor `Word324 Word4`, negative amounts (-4,645.86 and -790.66), period 2018-10.
- **C6 (12 rows):** 3 rows from vendor `Word137 Word138 Word7` in 2018-04. Most others have Col6 starting `Word571 Word52`.
- Vendor `Word137 Word138 Word7` has 24 rows, all amount 1,680: C3 18, C4 3, C6 3. Col4 tokens and the period separate them. Removing the date cut C4 recall from 0.65 to 0.48 (Exp 4).

## 13. Accuracy ceiling

Rows with identical Col1, Col4, and Col6 text but different labels: 84 rows in 15 groups. The resulting ceiling: accuracy 0.9961. Adding Col3 and Col7 to the key leaves 1 mixed group (5 rows), ceiling 0.9997. Remaining model errors come from generalization gaps, not identical-input conflicts.

## 14. Exploratory experiment log

Common setup unless stated: grouped CV with `StratifiedGroupKFold(n_splits=5, shuffle=True)` on the full file, group key from Section 11, seeds 0, 1, 2, metrics averaged across seeds. Run on one CPU core.

Feature sets:

- **base:** per-column TF-IDF (Col1, Col4, Col6), token pattern `\S+`, unigrams and bigrams, `sublinear_tf=True`. One-hot Col7. Scaled numeric: signed log amount, negative flag, period index (`year*12 + month`).
- **combined:** base with unigrams only, plus round-amount flag (`Col3 % 10 == 0`), whole-value one-hot for Col1 and Col6 (`min_frequency=2`), and vendor-by-account one-hot (`min_frequency=2`).

Class weight power `p`: weight of class c = `(n_max / n_c) ** p`. `p=0` means unweighted. `p=1` matches sklearn `balanced` up to scale.

### Exp A. Leakage probe (one seed, base features, LR C=10 balanced)

| Split | Accuracy | Six-class macro F1 |
|---|---|---|
| Random StratifiedKFold | 0.9527 | 0.7514 |
| Grouped StratifiedGroupKFold | 0.9461 | 0.7132 |

### Exp 1. Baselines and linear family (base features, 3 seeds)

| Model | Accuracy | macro_f1_5 | Recall C2 | C3 | C4 | C6 |
|---|---|---|---|---|---|---|
| Dummy (always C1) | 0.8840 | 0.1877 | 0.00 | 0.00 | 0.00 | 0.00 |
| Lookup: majority per (Col1, Col6), fallback Col1, fallback global | 0.9296 | 0.5976 | 0.60 | 0.87 | 0.08 | 0.33 |
| LogisticRegression C=10, balanced | 0.9448 | 0.7295 | 0.88 | 0.96 | 0.65 | 0.50 |
| LogisticRegression C=10, unweighted | 0.9538 | 0.7082 | 0.79 | 0.87 | 0.35 | 0.39 |
| LinearSVC C=1, balanced | 0.9515 | 0.6948 | 0.86 | 0.93 | 0.33 | 0.44 |
| ComplementNB alpha=0.1 (text and Col7 only) | 0.9121 | 0.6589 | 0.94 | 0.96 | 0.56 | 0.50 |

### Exp 2. LightGBM leaf size (base features, balanced, one seed)

| Config | Accuracy | macro_f1_5 |
|---|---|---|
| 300 trees, lr 0.05, 31 leaves, min_child_samples=20 | 0.9393 | 0.6324 |
| 400 trees, 15 leaves, depth 6, min_child_samples=5, colsample 0.5 | 0.9393 | 0.7029 |
| 500 trees, 31 leaves, depth 8, min_child_samples=2, reg_lambda 2, colsample 0.5, subsample 0.8 | 0.9456 | 0.7225 |

With the default 20-row leaf minimum, a leaf of only C4 rows (about 13 per training fold) never forms.

### Exp 3. CatBoost (fold 0 only)

CatBoost with native categorical (Col1, Col6, Col7) and text features (Col1, Col4, Col6), 300 iterations, depth 6, balanced weights: 193 seconds per fold, accuracy 0.924, macro_f1_5 0.635. Logistic regression on the same fold: under 1 second, accuracy 0.942, macro_f1_5 0.742. Excluded from production. Keep as an optional research benchmark.

### Exp 4. Ablations (LR C=10 balanced, 3 seeds)

| Variant | Accuracy | macro_f1_5 | Recall C4 |
|---|---|---|---|
| base | 0.9448 | 0.7295 | 0.65 |
| + Col2 format features | 0.9451 | 0.7248 | 0.67 |
| + round flag | 0.9465 | 0.7376 | 0.65 |
| + whole-value Col1/Col6 one-hot | 0.9442 | 0.7366 | 0.71 |
| + vendor-by-account one-hot | 0.9443 | 0.7418 | 0.73 |
| unigrams only | 0.9449 | 0.7286 | 0.67 |
| date as year + month | 0.9451 | 0.7076 | 0.50 |
| date as period one-hot | 0.9473 | 0.7140 | 0.50 |
| no date | 0.9447 | 0.7099 | 0.48 |
| minus Col1 text | 0.9268 | 0.6860 | 0.50 |
| minus Col4 text | 0.9059 | 0.5911 | 0.88 |
| minus Col6 text | 0.9430 | 0.6863 | 0.58 |
| minus Col7 | 0.9458 | 0.7413 | 0.67 |
| minus amount features | 0.9452 | 0.7426 | 0.69 (C5 recall falls from 1.00 to 0.00) |

### Exp 5. Regularization (combined features, LR balanced, 3 seeds)

| C | Accuracy | macro_f1_5 |
|---|---|---|
| 1 | 0.9251 | 0.6961 |
| 3 | 0.9365 | 0.7299 |
| 10 | 0.9425 | 0.7398 |
| 30 | 0.9447 | 0.7328 |
| 100 | 0.9465 | 0.7041 |

### Exp 6. LightGBM and probability-average blend (combined features, balanced)

| Model | Seeds | Accuracy | macro_f1_5 |
|---|---|---|---|
| LightGBM (Exp 2 best config) | 3 | 0.9450 | 0.7268 |
| Blend: mean of LR and LightGBM probabilities | 2 | 0.9478 | 0.7581 |

### Exp 7 and 8. Robustness views (combined features, LR C=10 balanced)

| View | Result |
|---|---|
| CV grouped by vendor (every test vendor unseen) | accuracy 0.8911, macro_f1_5 0.3956 |
| Out-of-time: train on Col5 up to 2018-09-30 (4,724 rows), test on later periods (1,175 rows) | accuracy 0.877, below the 0.887 majority rate for those periods |
| Out-of-time test rows with unseen vendor / account / vendor-account pair | 8.8% / 1.7% / 19.7% |
| Out-of-time accuracy, balanced LR: seen pair (943 rows) / unseen pair (232 rows) | 0.914 / 0.728 |
| Out-of-time accuracy, unweighted LR: overall / seen pair / unseen pair | 0.932 / 0.945 / 0.879 |
| Grouped CV accuracy: seen pair / unseen pair (608 unseen rows) | 0.948 / 0.888 |

Balanced weights labeled 67 C1 rows as rare classes out of time. Each C3 row counts as 227 C1 rows under balanced weights.

### Exp 9. Class weight power (combined features, LR C=10, 2 seeds, plus out-of-time)

| Power | CV accuracy | CV macro_f1_5 | Out-of-time accuracy | C1 rows predicted as C3 to C6 out of time |
|---|---|---|---|---|
| 0 | 0.9509 | 0.7204 | 0.932 | 6 |
| 0.25 | 0.9496 | 0.7369 | 0.923 | 14 |
| 0.5 | 0.9490 | 0.7444 | 0.918 | 20 |
| 0.75 | 0.9475 | 0.7401 | 0.893 | 47 |
| 1.0 | 0.9443 | 0.7238 | 0.874 | 67 |

### Exp 10. Confidence, label noise, deduplication (LR power 0.5, seed 0)

| Threshold on max probability | Coverage | Accuracy kept | Accuracy flagged | Rows flagged |
|---|---|---|---|---|
| 0.6 | 0.987 | 0.9541 | 0.646 | 79 |
| 0.7 | 0.974 | 0.9588 | 0.621 | 153 |
| 0.8 | 0.956 | 0.9645 | 0.636 | 261 |
| 0.9 | 0.922 | 0.9757 | 0.644 | 458 |

- 132 wrong predictions with probability at or above 0.9: label C1 predicted C2 61, label C2 predicted C1 58, others 13. Candidates for client label review.
- Training on deduplicated folds: accuracy 0.9500, macro_f1_5 0.7478, unchanged from training with duplicates. Keep duplicates in training. Group them in evaluation.

### Exp 11. Blend by class weight power (combined features, seed 0, probability averaging)

| Model | CV accuracy | CV macro_f1_5 | Out-of-time accuracy |
|---|---|---|---|
| Blend, power 0 | 0.9559 | 0.7345 | 0.942 |
| Blend, power 0.25 | 0.9541 | 0.7391 | 0.938 |
| Blend, power 0.5 | 0.9532 | 0.7359 | 0.935 |
| LR alone, power 0.5 | 0.9500 | 0.7467 | 0.918 |

The unweighted blend leads on accuracy in both views. This build confirms the result across 3 seeds with log-odds blending (docs/04 section 8).

## 15. Reference ranges for reproduction

The build runs on the 90% training portion, so numbers shift. Use these ranges as sanity checks. Investigate any result outside its range before moving on. A large unexplained gain signals leakage. A large loss signals a bug.

| Quantity | Reference | Tolerance |
|---|---|---|
| Majority baseline accuracy | 0.884 | plus or minus 0.005 |
| Lookup baseline accuracy | 0.930 | plus or minus 0.01 |
| LR (combined, power 0) grouped CV accuracy | 0.951 | plus or minus 0.01 |
| Blend (power 0) grouped CV accuracy | 0.956 | plus or minus 0.01 |
| Blend (power 0) out-of-time accuracy | 0.942 | plus or minus 0.015 |
| Any model macro_f1_5 | per tables above | plus or minus 0.05 |
| Random-split accuracy minus grouped accuracy | positive | must stay positive |
| Unseen-vendor CV accuracy | 0.891 | plus or minus 0.02 |
| LightGBM min_child_samples=20 versus 2 | 2 wins on macro_f1_5 | direction must hold |

Store these ranges in `tools/reference_ranges.json` and check them in the gate.

## 16. Known gotchas

- **pandas 3 strings.** `Series.values` on a string column returns a `StringArray`. CatBoost and some sklearn paths reject a `StringArray`. Use `.to_numpy(dtype=object)`.
- **HistGradientBoosting categorical cap.** Native categorical features accept at most 255 categories. Col1 has 313.
- **VotingClassifier and class weights.** `VotingClassifier` label-encodes `y` before fitting members, so a `class_weight` dict keyed by label strings fails. Implement the blend as a custom estimator.
- **StratifiedGroupKFold warning.** C5 has 2 rows, fewer than `n_splits`. sklearn warns. Suppress only this warning, locally, with a comment.
- **LightGBM on macOS.** PyPI wheels need the OpenMP runtime (`brew install libomp`). Without the runtime, `import lightgbm` fails. The fallback model handles this. See docs/05 section 5.
- **LightGBM on Linux slim images.** Needs `libgomp1` (`apt-get install -y libgomp1`).
- **CatBoost runtime.** 193 seconds per fold on one core. Research only.
- **Col2 parsing.** Read Col2 as string to stop numeric coercion and preserve original values for display.
