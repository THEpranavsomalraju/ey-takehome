# 04. Modeling Specification

Exact specifications for splits, features, models, tuning, selection, calibration, flags, and explanations. Values marked "default" live in `config.py`. Values marked "selected" come from `artifacts/champion.json` after the `select` command.

## 1. Labels

- Canonical classes, in this order everywhere: `Category_1`, `Category_2`, `Category_3`, `Category_4`, `Category_5`, `Category_6`.
- Normalization (docs/01 section 3): lowercase, strip, remove spaces and underscores, replace `categry` with `category`, require the form `category<digits>`, map to `Category_<digits>`. Anything else returns `None` (unrecognized).
- Training: every raw label must map. An unmapped training label raises `SchemaError`.
- Uploads: unrecognized labels exclude the row from scoring (warning). Prediction still happens.
- Every probability matrix uses all six canonical columns. When a fitted model lacks a class (a training fold without C5), fill the missing column with probability 0 and logit `-1e9`.

## 2. Group key and splits

### 2.1 Group key

Build from `Col1, Col3, Col4, Col5, Col6, Col7` after schema coercion (display values, missing Col4 as `MISSING_TOKEN`). Col2 stays out because many duplicates differ only in Col2. Factorize the joined key into integer group ids.

### 2.2 Holdout seed rule (finishing item E)

Write this rule into DECISIONS.md (D03) before running the split. The rule depends only on label counts, never on model performance.

```
for seed in 0..99:
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    train_idx, holdout_idx = first split from splitter.split(X, y, groups)
    accept the seed when all hold:
        580 <= len(holdout_idx) <= 600
        holdout has at least 1 row of each of Category_3, Category_4, Category_6
        holdout has 0 rows of Category_5
        abs(share_C1_holdout - share_C1_all) <= 0.01
        abs(share_C2_holdout - share_C2_all) <= 0.01
first accepted seed wins
```

Exploration found seed 0 acceptable on the full file (590 rows: C1 522, C2 63, C3 1, C4 1, C5 0, C6 3). Run the rule anyway and record the result.

Persist the split. Never recompute the split at train time:

- `data/splits/holdout_row_ids.csv`: one column `row_id` (0-based row index in the raw file)
- `data/splits/split_manifest.json`: seed, rule, class counts per side, group counts, sha256 of raw file, sha256 of the row id file, sklearn version used
- `data/holdout/holdout_features.xlsx` and `.csv`: Col1 to Col7 only, original raw values, Col2 written as text
- `data/holdout/holdout_labels.csv`: `row_number`, `ClassificationLabel` (raw value)
- `data/holdout/holdout_with_labels.xlsx`: features plus raw label, for scoring demos in the dashboard

Holdout files keep the raw (uncoerced) values so the upload path exercises the full validation layer.

### 2.3 CV schemes (all on training rows only)

| Scheme | Splitter | Groups | Purpose |
|---|---|---|---|
| Primary | `StratifiedGroupKFold(5, shuffle=True, random_state=s)` for s in (0, 1, 2) | group key | All model comparison and selection |
| Leakage demo | `StratifiedKFold(5, shuffle=True, random_state=0)` versus Primary seed 0 | none / group key | Shows inflation from duplicates |
| Out-of-time | train: period <= 2018-09, test: period > 2018-09 | n/a | Generalization to later periods (guardrail) |
| Unseen vendor | `StratifiedGroupKFold(5, shuffle=True, random_state=0)` | normalized Col1 | Floor for new vendors |
| Nested | outer Primary seed 0, inner `StratifiedGroupKFold(3, shuffle=True, random_state=0)` | group key | Unbiased estimate of the tuned pipeline |

No group crosses the out-of-time boundary, because Col5 sits inside the group key.

## 3. Features

All transformers live in `features.py`, fit inside the pipeline, and expose `get_feature_names_out`. Schema coercion (docs/05) handles generic cleaning and keeps display case. The feature pipeline applies model-specific normalization.

| Group | Source | Chain | Parameters (default) |
|---|---|---|---|
| Col1 tokens | Col1 | `TextNormalizer` then `TfidfVectorizer` | `token_pattern=r"(?u)\S+"`, `lowercase=False` (normalizer lowercases), `ngram_range=(1,1)`, `sublinear_tf=True`, `min_df=1`, `dtype=np.float64` |
| Col1 whole value | Col1 | `TextNormalizer` then `OneHotEncoder` | `handle_unknown="infrequent_if_exist"`, `min_frequency=2`, `sparse_output=True` |
| Col4 tokens | Col4 | as Col1 tokens | as Col1 tokens |
| Col6 tokens | Col6 | as Col1 tokens | as Col1 tokens |
| Col6 whole value | Col6 | as Col1 whole value | as Col1 whole value |
| Vendor-account pair | Col1, Col6 | `TextNormalizer`, `ColumnConcat(sep=" || ")`, `OneHotEncoder` | `handle_unknown="infrequent_if_exist"`, `min_frequency=2` |
| Col7 | Col7 | `TextNormalizer` then `OneHotEncoder` | `handle_unknown="ignore"` |
| Amount | Col3 | `TrainingMedianImputer`, `SignedLogAmount`, `StandardScaler` | |
| Amount flags | Col3 | `TrainingMedianImputer`, `AmountFlags` | `is_negative = x < 0`, `is_round = (x % 10 == 0)` |
| Period | Col5 | `TrainingMedianImputer`, `PeriodEncoder(clip=True)`, `StandardScaler` | period index `year*12 + month`, clipped to the training min and max |
| Col2 | n/a | dropped | display only |

`TextNormalizer`: lowercase, missing or empty to `__missing__`. (Unicode normalization, whitespace collapsing, and truncation happen earlier in schema coercion.)

Period clipping matters: logistic regression extrapolates linearly, so an unclipped 2030 date would push predictions without evidence. Clipping keeps inference inside the training range. The flag `DATE_OUT_OF_RANGE` tells the user.

Output: one sparse CSR matrix shared by both blend members. Never densify the full matrix.

## 4. Model zoo (benchmark stage)

Every model below runs on Primary CV, 3 seeds, combined features unless noted. Record accuracy, macro_f1_5, macro_f1_6, balanced accuracy, per-class recall, log loss (when probabilities exist), and fit seconds.

| Name | Definition |
|---|---|
| `dummy_majority` | `DummyClassifier(strategy="most_frequent")` |
| `lookup_vendor_account` | Majority label per (Col1, Col6), fallback per Col1, fallback global majority |
| `complement_nb` | `ComplementNB(alpha=0.1)` on text and Col7 features only |
| `linear_svc` | `LinearSVC(C=1.0)`, power 0.5, predictions from `decision_function` argmax, no probability calibration (an inner random calibration split would cross duplicate groups), log loss reported as n/a |
| `logistic_p0` | `LogisticRegression(C=10, max_iter=5000, solver="lbfgs")`, power 0 |
| `logistic_p05` | same, power 0.5 |
| `lightgbm_leaf20` | LightGBM, 300 trees, learning_rate 0.05, num_leaves 31, min_child_samples 20, power 0 |
| `lightgbm_p0` | LightGBM, 500 trees, learning_rate 0.05, num_leaves 31, max_depth 8, min_child_samples 2, reg_lambda 2.0, colsample_bytree 0.5, subsample 0.8, subsample_freq 1, power 0 |
| `probability_blend_p0` | mean of `logistic_p0` and `lightgbm_p0` probabilities (comparison only) |
| `logodds_blend_p0` | `LogOddsBlendClassifier(logistic_p0, lightgbm_p0, linear_weight=0.5)` |
| `logodds_blend_p025` | same, power 0.25 |
| `catboost` | research only, from `reports/research/catboost.json`, shown with a "1 fold, research" note |

LightGBM always runs with `objective="multiclass"`, `deterministic=True`, `force_col_wise=True`, `n_jobs=1`, `random_state` from config, `verbose=-1`.

Also run:

- **Ablations** on `logistic_p0`: drop each feature group in turn (Col1 tokens, Col1 whole, Col4, Col6 tokens, Col6 whole, pair, Col7, amount, amount flags, period), plus additions (Col2 format features, bigrams). Output `reports/benchmark/ablation.csv` with deltas versus the full set.
- **Class weight dial**: `logistic` and `logodds_blend` at power 0, 0.25, 0.5, 0.75, 1.0: CV accuracy, macro_f1_5, out-of-time accuracy, count of C1 rows predicted as C3 to C6 out of time.
- **Leaf size effect**: LightGBM with min_child_samples in (1, 2, 5, 10, 20), other params as `lightgbm_p0`.
- **Validation schemes**: leakage demo, out-of-time, unseen vendor, each for `logistic_p0` and `logodds_blend_p0`, plus accuracy split by seen versus unseen vendor-account pair.

## 5. Class weight power

`PoweredClassWeight(estimator, power)` computes, at every fit, `weight_c = (n_max / n_c) ** power` from the labels of the current fit, and sets `class_weight` on a clone of the wrapped estimator. Works for `LogisticRegression` and `LGBMClassifier`. Exposes nested params (`estimator__C`) for searches.

## 6. LogOddsBlendClassifier (finishing item C)

### 6.1 Math

Inputs: one shared feature matrix `X` (sparse). Classes in canonical order.

```
z_lin(x)  = W x + b                       # LogisticRegression.decision_function (multinomial)
z_tree(x) = LightGBM raw score            # predict(X, raw_score=True)
z(x)      = (w * z_lin(x) + (1 - w) * z_tree(x)) / T
p(x)      = softmax(z(x))
```

`w` = `linear_weight` (default 0.5, selected by search). `T` = temperature (default 1.0, set by calibration).

### 6.2 Exact contributions

```
linear:  c_lin[k, j]  = W[k, j] * x_j         bias_lin[k]  = b[k]
tree:    c_tree[k, j] = pred_contrib value    bias_tree[k] = expected value column
blend:   c[k, j]      = (w * c_lin[k, j] + (1 - w) * c_tree[k, j]) / T
         bias[k]      = (w * bias_lin[k] + (1 - w) * bias_tree[k]) / T
identity: sum_j c[k, j] + bias[k] == z_k(x)
```

LightGBM `predict(X, pred_contrib=True)` returns, for multiclass sparse input, a list of per-class sparse matrices with `n_features + 1` columns, and for dense input one array of width `K * (n_features + 1)`. Handle both shapes. Map LightGBM class order to canonical order explicitly.

Explanation target: predicted class `k*` versus runner-up `r`. Margin `m = z_k* - z_r`. Contribution of feature j to the margin: `c[k*, j] - c[r, j]`. Bias term: `bias[k*] - bias[r]`.

### 6.3 Acceptance tests

1. `predict_proba` equals `softmax(decision_function)` within 1e-9.
2. For 200 random training rows and every holdout-features row: contributions plus bias equal the blended logit for every class within 1e-6.
3. Group-aggregated contributions sum to the same totals within 1e-6.
4. Class order assertion: `LogisticRegression.classes_` equals `LGBMClassifier.classes_` equals the canonical subset.
5. Log-odds blend versus probability blend on Primary CV, 3 seeds: record both in `reports/selection/blend_comparison.csv`. The log-odds blend stays the production form for exact explanations unless its accuracy trails by more than 0.003 (then ask the user).

## 7. Hyperparameter search (finishing item B)

Use sklearn `GridSearchCV` and `RandomizedSearchCV` with `cv=StratifiedGroupKFold(5, shuffle=True, random_state=0)`, `groups` passed to `fit`, `scoring={"accuracy": "accuracy", "macro_f1_5": make_scorer(f1_score, average="macro", labels=FIVE_CLASSES, zero_division=0)}`, `refit=False`, `return_train_score=False`, `error_score="raise"`.

| Search | Space | Budget |
|---|---|---|
| Linear | `C` in (1, 3, 10, 30, 100), power in (0, 0.25, 0.5), ngram in ((1,1), (1,2)) | full grid, 30 configs |
| Tree | n_estimators (300, 500, 800), learning_rate (0.03, 0.05, 0.1), num_leaves (15, 31, 63), max_depth (-1, 6, 8, 12), min_child_samples (1, 2, 3, 5, 10, 20), reg_lambda (0, 1, 2, 5), colsample_bytree (0.3, 0.5, 0.8), subsample (0.7, 0.8, 1.0), power (0, 0.25) | 40 random configs, `random_state=0` |
| Blend weight | linear_weight in (0.3, 0.4, 0.5, 0.6, 0.7), power in (0, 0.25), best linear and tree params | full grid, 10 configs |

Search metric: accuracy (rubric). Always log macro_f1_5 beside accuracy.

Logging: `reports/tuning/search_space.json` and one CSV per search (`linear_trials.csv`, `tree_trials.csv`, `blend_trials.csv`) with columns `trial_id`, one column per parameter, `mean_accuracy`, `std_accuracy`, `mean_macro_f1_5`, `std_macro_f1_5`, `mean_fit_seconds`, `rank_accuracy`.

Nested CV (`reports/tuning/nested_cv.json`): outer Primary seed 0, inner 3-fold grouped search with a reduced space (linear: C x power, 15 configs. tree: 15 random configs. blend weight: 0.4, 0.5, 0.6). Report outer accuracy and macro_f1_5 mean and std, the non-nested best score, and the gap. `--quick` cuts every budget by about 75% for development runs. Final reports always come from a full run.

## 8. Scorecard and decision rule (finishing item A)

Record this rule in DECISIONS.md (D11) before running `select`.

Candidates, each with its best tuned parameters:

1. `logistic` at power 0
2. `logistic` at power 0.25
3. `logodds_blend` at power 0
4. `logodds_blend` at power 0.25

Scores per candidate: Primary CV accuracy (mean and std over 3 seeds, 15 folds), Primary CV macro_f1_5 (mean), out-of-time accuracy.

```
best_f1  = max macro_f1_5 across candidates
best_oot = max out-of-time accuracy across candidates
eligible = candidates with macro_f1_5 >= best_f1 - 0.03 and oot >= best_oot - 0.015
ranked   = eligible sorted by CV accuracy, descending
if ranked[0].cv_acc - ranked[1].cv_acc < 0.002:
    pick the higher out-of-time accuracy among the two
    if their out-of-time accuracies differ by < 0.002:
        pick the simpler: logistic over blend, then lower power over higher power
else:
    pick ranked[0]
```

Write `artifacts/champion.json` (config, all candidate scores, the decision trail step by step) and `reports/selection/champion_candidates.csv`. The fallback model: the best `logistic` candidate.

## 9. Calibration (finishing item D)

1. Collect out-of-fold blended logits before temperature for the champion on Primary CV seed 0 (pre-temperature logits, all six canonical classes).
2. Fit temperature: `T* = argmin_T NLL(softmax(z / T), y)` with `scipy.optimize.minimize_scalar(bounds=(0.05, 20.0), method="bounded")`.
3. Report before and after: ECE (10 equal-width bins on max probability), multiclass Brier score, log loss, reliability table (bin, rows, mean confidence, accuracy).
4. Temperature scaling preserves argmax (accuracy unchanged) and keeps logits additive (contributions scale by `1/T`). Isotonic or sigmoid calibration breaks additivity: do not use them on the champion.
5. Target ECE after scaling: 0.03 or lower. A miss does not block, but log the value and reason in DECISIONS.md (D14).
6. Repeat for the fallback logistic model (its own temperature).

Threshold table on calibrated out-of-fold probabilities, thresholds (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99): coverage, accuracy kept, accuracy flagged, rows flagged. Default selection rule: the lowest threshold whose kept accuracy reaches 0.975. Record in DECISIONS.md (D17).

## 10. Flags and review policy

Row flags (computed against reference statistics from the training data of the current fit, so out-of-fold analysis uses fold-specific references):

| Code | Condition | Default severity |
|---|---|---|
| `UNSEEN_VENDOR` | normalized Col1 absent from training | critical |
| `UNSEEN_ACCOUNT` | normalized Col6 absent | warning |
| `UNSEEN_VENDOR_ACCOUNT_PAIR` | pair absent, both parts known | warning |
| `UNSEEN_DOC_TYPE` | Col7 absent | warning |
| `ALL_TOKENS_UNSEEN` | no Col1, Col4, or Col6 token known | critical |
| `TEXT_MISSING` | Col1 or Col6 missing | critical |
| `AMOUNT_IMPUTED` | Col3 missing or unparseable | critical |
| `AMOUNT_OUT_OF_RANGE` | Col3 outside training min and max | warning |
| `DATE_IMPUTED` | Col5 missing or unparseable | critical |
| `DATE_OUT_OF_RANGE` | period outside training range (clipped for features) | warning |
| `TEXT_TRUNCATED` | a text cell exceeded the length limit | info |
| `LOW_CONFIDENCE` | calibrated confidence below threshold | critical |

`review_required = any critical flag`. Measure accuracy per flag on out-of-fold predictions (`reports/selection/accuracy_by_flag.csv`) and adjust severities by evidence before `train`. Record the final policy in DECISIONS.md (D18).

## 11. Final training and holdout evaluation

`train`:

1. Load the persisted split and training rows only.
2. Read `artifacts/champion.json`.
3. Fit the champion pipeline on all training rows. Set temperature from `champion.json`.
4. Fit the fallback logistic pipeline. Set its temperature.
5. Build reference statistics from training rows.
6. Save both bundles, `metadata.json`, `reference.json`.
7. Predict on `holdout_features.xlsx` through `predict_file` (features only, no labels) with both models. Save `golden_holdout_predictions.csv`.
8. Predict `ledgerclf/resources/sample_rows.csv`. Save `sample_predictions.csv`.

`evaluate-holdout` (ask the user first):

1. Refuse without `--confirm`. Refuse when `reports/holdout/holdout_evaluation.json` exists, unless `--rerun --reason "..."`.
2. Append to `reports/holdout/access_log.csv`: UTC timestamp, git commit (or "unknown"), reason, artifact sha256 values.
3. Score both models: accuracy, macro_f1_5 (with per-class support), macro_f1_6, balanced accuracy, log loss, Brier, ECE, per-class report, confusion matrix, accuracy by flag, CV estimate beside each holdout number.
4. Report rare-class metrics with row counts beside them (expect 1 to 3 rows per rare class).

## 12. Reference statistics

Built from training rows after schema coercion and model normalization (lowercase). Contents in docs/02 section 4 (`ReferenceStats`). Also store amount quantiles at 101 points (0 to 100) for percentile lookup in explanations. Serialize to `artifacts/reference.json` with sorted keys.

## 13. Explanations

`explain_rows` returns `ExplanationPayload` dicts (schema in docs/02 section 5):

- `contributions.by_group`: sum of margin contributions per group (Col1, Col1+Col6 pair, Col3, Col4, Col5, Col6, Col7), sorted by absolute value.
- `contributions.top_features`: top 8 features by absolute margin contribution, with readable names (`token:word575`, `value:word563 word575`, `pair:...`, `amount_signed_log`, `is_round`, `period`, `doc:nodoc`). Features absent from the row (x = 0) carry `present_in_row: false`. Tree contributions for absent tokens are real signal ("the row lacks token X").
- `evidence`: Col1 value, Col6 value, pair, Col7 value, plus the three highest-impact present tokens, each with training row counts and class counts from reference statistics.
- `flags`: codes with messages.
- `context`: training rows, class prior, period range, amount percentile.
- `fallback_text`: deterministic template, used when no LLM is available:

```
Predicted {label} with {p:.0%} confidence (runner-up {r}, {pr:.0%}).
Strongest drivers: {group_1} ({toward_1}), {group_2} ({toward_2}), {group_3} ({toward_3}).
Evidence: in training, {n_class} of {n_total} rows with {strongest seen value description} were {label}.
{flag sentence, or "No risk flags."}
```

Performance: one explanation under 300 ms on a laptop.

## 14. Metric definitions

| Metric | Definition |
|---|---|
| accuracy | share of correct predictions |
| macro_f1_5 | `f1_score(average="macro", labels=[C1, C2, C3, C4, C6], zero_division=0)` |
| macro_f1_6 | same over all six classes |
| balanced_accuracy | sklearn `balanced_accuracy_score` |
| per-class | precision, recall, F1, support, canonical order |
| log_loss | sklearn `log_loss(labels=all six)` on calibrated probabilities |
| brier | mean over rows of the sum over classes of `(p_k - y_k)^2` |
| ece | 10 equal-width bins on max probability: sum of `bin_share * abs(accuracy - confidence)` |

## 15. Required reports

`reports/eda/`: `schema_summary.json`, `raw_label_mapping.csv`, `class_distribution.csv`, `missing_values.csv`, `col2_type_mix.csv`, `col2_patterns.csv`, `amount_summary.csv`, `amount_by_class.csv`, `amount_histogram.csv` (signed-log bins by class), `amount_outliers.csv` (IQR on signed log, with class), `period_by_class.csv`, `day_of_month.csv`, `col7_by_class.csv`, `text_cardinality.csv`, `top_values_col1.csv`, `top_values_col4.csv`, `top_values_col6.csv`, `token_overlap.csv`, `cramers_v.csv` (bias-corrected, among Col1, Col6, pair, Col7, period, amount decile, label), `mutual_information.csv` (each column versus label, raw and normalized), `duplicates_summary.json`, `duplicate_group_sizes.csv`, `conflict_groups.csv`, `rare_class_rows.csv`, `accuracy_ceiling.json`.

`reports/preprocessing/`: `stage_snapshots.json` (rows, columns, missing counts, dtypes at each stage: raw, resolved, coerced, engineered, matrix), `before_after_samples.csv` (5 rows: raw, coerced, engineered summary), `feature_space_summary.csv` (group, features, non-zero share), `transformer_catalog.json` (each transformer, parameters, reason, decision id).

`reports/benchmark/`: `leaderboard.csv`, `leaderboard_per_seed.csv`, `ablation.csv`, `validation_schemes.csv`, `oot_detail.json`, `unseen_pair_accuracy.csv`, `class_weight_dial.csv`, `leaf_size_effect.csv`.

`reports/tuning/`: `search_space.json`, `linear_trials.csv`, `tree_trials.csv`, `blend_trials.csv`, `nested_cv.json`.

`reports/selection/`: `champion_candidates.csv`, `champion_decision.json`, `blend_comparison.csv`, `oof_confusion_matrix.csv`, `oof_classification_report.csv`, `calibration_reliability.csv`, `calibration_summary.json`, `threshold_table.csv`, `accuracy_by_flag.csv`, `high_confidence_errors.csv`.

`reports/holdout/` (written only by `evaluate-holdout`): `holdout_evaluation.json`, `holdout_classification_report.csv`, `holdout_confusion_matrix.csv`, `holdout_predictions.csv`, `holdout_accuracy_by_flag.csv`, `access_log.csv`.

`reports/robustness/`: `edge_case_matrix.csv` (catalog id, description, test name, status), `performance.json` (timings against budgets).

`reports/research/catboost.json` (optional).

`reports/manifest.json` lists every file above with title, producer command, description, and columns.
