# DECISIONS

Every modeling, data, and engineering decision, with evidence. Statuses: `Accepted`, `Pending (Phase N)`, `Superseded by Dxx`. Numbers marked "exploration" come from docs/01 (full file, before the holdout existed). Replace them with build numbers as each phase completes.

Template for new entries:

```
## Dxx. Title
Status:
Context: the problem or question
Decision: what we chose
Alternatives: what else we considered and why we rejected it
Evidence: numbers, report paths, test names
Consequences: what this changes, risks, follow-ups
```

---

## D00. Exploration provenance

Status: Accepted

Context: exploratory experiments (docs/01 section 14) ran on the full labeled file before any holdout existed, so holdout rows influenced plan-level choices (model family, feature set, weighting direction).

Decision: disclose this openly. Re-run every analysis and the full selection on the 90% training portion. Evaluate the holdout exactly once at the end.

Alternatives: discard exploration and start blind (wastes verified insight). Pick a different holdout to "hide" the overlap (no real benefit, since every row was seen during exploration).

Evidence: docs/01 provenance note.

Consequences: holdout accuracy carries mild optimism at the level of broad design choices, not at the level of fitted parameters. State this in MODEL_CARD.md and in the interview.

## D01. Normalize labels to six canonical classes

Status: Accepted

Context: the label column holds 11 spellings of 6 classes (`Category 3`, `Category _3`, `Categry_6`, `category_1`, and others).

Decision: deterministic normalization (lowercase, remove spaces and underscores, fix `categry`, extract digits) to `Category_1` through `Category_6`. Unmapped training labels raise an error. Unmapped upload labels exclude the row from scoring only.

Alternatives: manual mapping dict (brittle for new spellings). Leaving labels raw (splits one class into several).

Evidence: docs/01 section 3. Test: label mapping counts.

Consequences: the same function serves training and upload scoring.

## D02. Group key excludes Col2, grouped splits everywhere

Status: Accepted

Context: 680 rows duplicate another row on all inputs. Ignoring Col2, 1,508 rows (25.6%) duplicate another. Random splits put copies on both sides.

Decision: group key from Col1, Col3, Col4, Col5, Col6, Col7. Use `StratifiedGroupKFold` for the holdout split and every CV.

Alternatives: random stratified split (inflated scores). Deduplicate before splitting (loses real frequency information and still leaves near-duplicates differing in Col2).

Evidence (exploration): random CV accuracy 0.9527 and six-class macro F1 0.7514 versus grouped 0.9461 and 0.7132 (Exp A).

Consequences: honest estimates. Holdout size lands near 10% instead of exactly 10%.

## D03. Holdout seed rule (finishing item E)

Status: Accepted. Rule fixed before the split (commit c732a06); result recorded below.

Decision: iterate seeds 0 to 99 of `StratifiedGroupKFold(n_splits=10, shuffle=True)`, take the first split as holdout, accept the first seed where: holdout size 580 to 600, at least one row of each of Category_3, Category_4, Category_6, zero rows of Category_5, Category_1 and Category_2 shares within 0.01 of the full file. The rule uses label counts only, never model results.

Rationale: C5 has 2 rows, so both stay in training. Rare classes need at least one holdout row to show any holdout behavior at all.

Evidence (exploration): seed 0 qualifies (590 rows: C1 522, C2 63, C3 1, C4 1, C5 0, C6 3).

Result (2026-10-17, build group key from D22): seed 0 accepted on the first try. Holdout 590 rows: C1 521, C2 63, C3 3, C4 2, C5 0, C6 1 (C1 share 0.8831 vs 0.8840 overall, C2 0.1068 vs 0.1070). Training 5,309 rows: C1 4,694, C2 568, C3 20, C4 14, C5 2, C6 11. Groups: 3,945 training, 435 holdout, none shared. `holdout_row_ids.csv` sha256 a524a97f751dadf3cd9518eebcd23fbd087c42293511e32a014f947e6dc31b6f. The composition differs from the exploration counts above because the build key groups by monthly period (D22); with the date-level key, seed 0 reproduces the exploration counts exactly (verified by the leakage audit). `split` refuses to change a persisted split unless run with `--force`, and `load_split` verifies the stored hashes.

## D04. Drop Col2 from the model

Status: Accepted

Context: Col2 looks like an invoice number. 4,818 unique values, mixed types, 326 corrupted scientific-notation values with swapped letters, synthetic-looking prefixes.

Decision: exclude Col2 from features. Keep Col2 as a display string.

Alternatives: raw Col2 as a category (memorizes IDs, fails on new invoices). Format features (length, letters, hyphen, digit share, scientific flag).

Evidence (exploration): format features moved macro_f1_5 from 0.7295 to 0.7248 (Exp 4).

Consequences: simpler pipeline, no ID memorization. Col2 missing in an upload triggers a warning only.

## D05. Text representation: per-column TF-IDF unigrams plus whole-value and pair one-hots

Status: Accepted (confirm in Phase 5 tuning)

Context: tokens are anonymized (`Word563`). The same token appears in several columns with different roles. Vendor and account identity drive most signal.

Decision: separate `TfidfVectorizer` per text column (token pattern `\S+`, unigrams, sublinear TF), whole-value one-hot for Col1 and Col6 (`min_frequency=2`), vendor-account pair one-hot (`min_frequency=2`).

Alternatives: one shared vectorizer (mixes roles). Bigrams (no gain). Pretrained embeddings or LLM features (no semantics in anonymized tokens). Character n-grams (not needed, tokens are atomic).

Evidence (exploration): unigrams 0.7286 versus bigrams 0.7295 macro_f1_5. Whole-value plus 0.007, pair plus 0.012 (within noise individually, consistent direction). Dropping Col4 text costs 4 points of accuracy (Exp 4).

## D06. Keep duplicates in training

Status: Accepted

Decision: train on all training rows including duplicates. Group them in evaluation.

Evidence (exploration): deduplicated training gave accuracy 0.9500 and macro_f1_5 0.7478, unchanged (Exp 10).

Consequences: duplicates act as natural frequency weights. No extra preprocessing step.

## D07. Amount features: signed log, negative flag, round flag, no outlier removal

Status: Accepted

Context: median 1,093, max 8.8 million, 69 negatives (credits), round amounts recur in rare classes.

Decision: `sign(x) * log1p(abs(x))` scaled, `is_negative`, `is_round`. No clipping, no row removal. Inference never drops rows.

Evidence (exploration): amount features carry the only C5 signal (C5 recall 1.00 with, 0.00 without, Exp 4). Round flag plus 0.008 macro_f1_5.

## D08. Period feature: numeric period index, clipped to the training range

Status: Accepted

Decision: period index `year*12 + month`, clipped at the training minimum and maximum, scaled. Flag out-of-range dates.

Alternatives: year plus month, period one-hot (both worse), no date (C4 recall falls from 0.65 to 0.48).

Consequences: clipping stops linear extrapolation for dates outside 2016-10 to 2018-12.

## D09. Primary metric: accuracy, with macro F1 guardrail

Status: Accepted

Context: the rubric scores "overall prediction performance on test and hold-out data". Rare classes still matter to a tax client.

Decision: select by grouped CV accuracy. Guardrails: macro_f1_5 within 0.03 of the best candidate and out-of-time accuracy within 0.015 of the best. Report C5 separately (one C5 row moves six-class macro F1 by up to 0.17).

Alternatives: macro F1 as the primary metric (pushed toward balanced weights, which failed out of time).

## D10. Class weight power

Status: Accepted (Phase 5). Chosen: 0.25 (through the champion rule, D11).

Context: balanced weights count one C3 row as 227 C1 rows and produced 67 false rare-class alarms out of time.

Evidence (exploration): blend power 0: CV 0.9559, out-of-time 0.942. Power 0.25: 0.9541, 0.938. Power 0.5: 0.9532, 0.935. LR power 1.0: out-of-time 0.874, below the 0.887 majority rate (Exp 9, Exp 11).

Result: the class weight dial (reports/benchmark/class_weight_dial.csv, 3 seeds) reproduces the exploration direction: logistic out-of-time false rare alarms 1, 2, 5, 9, 15 for powers 0 to 1. At power 0 the log-odds blend loses rare-class recall (Category_3 0.78) because confident LightGBM logits outvote the linear member; power 0.25 restores it (Category_3 0.95, macro_f1_5 0.75). The champion rule picked power 0.25.

## D11. Champion decision rule and champion model

Status: Accepted. Champion: logodds_blend_p0.25.

Rule: docs/04 section 8, recorded here before running `select`. Candidates: logistic power 0, logistic power 0.25, log-odds blend power 0, log-odds blend power 0.25.

Expectation (exploration): log-odds blend of logistic regression and LightGBM leads on accuracy in grouped CV and out of time.

Result (reports/selection/champion_decision.json, tuned parameters from reports/tuning/best_params.json):

- logistic_p0: CV accuracy 0.9513 (std 0.0082), macro_f1_5 0.6887, out-of-time 0.9442
- logistic_p0.25: CV accuracy 0.9511 (std 0.0077), macro_f1_5 0.7389, out-of-time 0.9336
- logodds_blend_p0: CV accuracy 0.9544 (std 0.0064), macro_f1_5 0.7159, out-of-time 0.9480
- logodds_blend_p0.25: CV accuracy 0.9539 (std 0.0068), macro_f1_5 0.7475, out-of-time 0.9403

Trail: best macro_f1_5 0.7475 and best out-of-time 0.9480. logistic_p0 (macro_f1_5 0.6887) and logodds_blend_p0 (0.7159, below the 0.7175 floor) are excluded by the macro F1 guardrail. Of the eligible candidates, logodds_blend_p0.25 leads logistic_p0.25 on CV accuracy by 0.0028 (above the 0.002 tie margin): chosen. Fallback: logistic_p0, the most accurate logistic candidate. Champion configuration: logistic C=3 with unigrams and bigrams, LightGBM 800 trees (learning rate 0.1, 31 leaves, min_child_samples 1, reg_lambda 1, colsample 0.8, subsample 0.7), linear weight 0.7, class weight power 0.25 for both members. Nested CV of the tuning procedure: outer accuracy 0.9548 +- 0.0050 versus 0.9540 non-nested (gap -0.0008).

Metric aggregation (recorded after the leakage audit, cycle 2; unchanged since c1c54e3, before selection ran): accuracy is the mean over 15 folds. Macro F1 and per-class recall are computed on each seed's pooled out-of-fold predictions and then averaged over the 3 seeds, because the rare classes have 0 to 4 rows in a single fold, so fold-level F1 would be mostly undefined. Tuning trials report fold means, which serve only for ranking inside one search. logodds_blend_p0 missed the floor by 0.0016 under this aggregation. The rule was applied as pre-registered and was not revisited after seeing that margin.

## D12. LightGBM minimum leaf size of 2

Status: Accepted (confirm in Phase 5 tuning)

Context: about 13 C4 rows per training fold. The default 20-row leaf minimum never forms a C4-only leaf.

Evidence (exploration): min_child_samples 20: macro_f1_5 0.6324. 2: 0.7225 (Exp 2).

## D13. CatBoost excluded from production

Status: Accepted

Evidence (exploration): 193 s per fold versus under 1 s for logistic regression, fold-0 macro_f1_5 0.635 versus 0.742 (Exp 3).

Consequences: CatBoost appears on the leaderboard as a research row only, with an optional requirements file.

## D14. Calibration by temperature scaling

Status: Accepted

Decision: one temperature on out-of-fold blended logits. Preserves accuracy and the additivity of log-odds contributions.

Alternatives: isotonic or sigmoid calibration (breaks exact explanations).

Result (reports/selection/calibration_summary.json, grouped CV seed 0): champion T = 1.392; ECE 0.0189 -> 0.0066 (target 0.03 met); Brier 0.0690 -> 0.0667; log loss 0.1297 -> 0.1189. Fallback T = 1.076; ECE 0.0085 -> 0.0056.

## D15. Fallback model when LightGBM fails to load

Status: Accepted

Context: LightGBM wheels on macOS need the OpenMP runtime (`brew install libomp`). Linux slim images need `libgomp1`. A grader machine might lack these.

Decision: ship a second artifact, a calibrated logistic regression with zero LightGBM objects. Import LightGBM lazily. Load the blend when possible, else the fallback, and surface the state in every result and in `doctor`.

Evidence (exploration): logistic regression alone sits about 0.5 points of CV accuracy below the blend (Exp 9, Exp 11).

Consequences: the application never breaks on a missing native library. Accuracy degrades slightly and visibly.

## D16. Demo model trains on the 90% only

Status: Accepted

Decision: artifacts used in the dashboard come from the training portion. Refitting on 100% belongs to a production deployment, described but not done here.

Rationale: a model trained on the holdout would score its own training rows in the Tab 2 demo.

## D17. Confidence threshold for review routing

Status: Accepted

Rule: lowest threshold whose kept accuracy on calibrated out-of-fold probabilities reaches 0.975.

Evidence (exploration, uncalibrated LR): threshold 0.9 kept 92.2% of rows at 97.6% accuracy, flagged rows at 64% accuracy (Exp 10).

Result: threshold 0.8 (reports/selection/threshold_table.csv): 92.7% of rows kept at 97.9% out-of-fold accuracy; the 7.3% flagged are 63.4% accurate. 0.7 keeps 97.5% exactly (0.9748, just under the target).

## D18. Row flags and review policy

Status: Accepted

Context: unseen vendors drop accuracy to near the majority baseline (grouped-by-vendor CV 0.891). Unseen vendor-account pairs: 0.888 in grouped CV, 0.728 out of time with balanced weights.

Decision draft: critical flags (review required) for unseen vendor, all tokens unseen, missing Col1 or Col6, imputed amount or date, low confidence. Warnings for unseen account, unseen pair, unseen document type, out-of-range values.

Result (reports/selection/accuracy_by_flag.csv, out-of-fold with fold-specific reference statistics; rule in config.FlagPolicyParams: critical when a flag's accuracy is more than 0.05 below the overall 0.954 on at least 10 rows, otherwise warning; flags with fewer rows keep their default): UNSEEN_VENDOR 154 rows, 0.896 -> critical; UNSEEN_ACCOUNT 112 rows, 0.893 -> critical (upgraded from warning by evidence); UNSEEN_VENDOR_ACCOUNT_PAIR 377 rows, 0.936 -> warning; LOW_CONFIDENCE 385 rows, 0.634 -> critical. Flags that never occur in training (imputed values, missing text, out-of-range values) keep their defaults. A new flag ROW_STRUCTURE_SUSPECT (critical) marks rows with extra fields or an unclosed quote in the uploaded file. Rows with no flag: 0.982 accuracy.

## D19. Category_5 handling

Status: Accepted

Context: 2 rows, both from vendor `Word324 Word4` with negative amounts.

Decision: keep C5 as a class, keep both rows in training (holdout rule), report C5 metrics separately, and state plainly in the model card: no model learns a class from 2 rows.

Alternatives: merge rare classes into "Other" (removes categories the client needs).

## D20. Rejected approaches

Status: Accepted

- LLM zero-shot classification and pretrained embeddings: anonymized tokens carry no meaning for a language model.
- SMOTE: synthetic TF-IDF vectors mix unrelated tokens into rows no invoice produces. Class weights cover the need.
- Deep learning: 5,309 training rows with 12 to 23 rows in rare classes.
- Stacking beyond the two-model blend: complexity without evidence of gain.
- `VotingClassifier` for the blend: label-encodes targets, breaks class-weight dicts, and averages probabilities instead of logits.

## D21. High-confidence errors as label-review candidates

Status: Accepted

Context: 132 wrong predictions with probability 0.9 or higher in exploration, mostly C1 versus C2.

Decision: publish `reports/selection/high_confidence_errors.csv` as a client review list instead of altering labels.

Result: 53 out-of-fold errors with calibrated confidence of 0.9 or more (training portion only).

## D22. Group key uses the monthly period, not the full date

Status: Accepted (2026-10-17, before the split ran)

Context: docs/04 section 2.1 builds the group key from coerced display values, and the coerced display value of Col5 is the month (`YYYY-MM`). docs/01 section 11 counted duplicates on full dates. The two disagree: 1,508 duplicate rows ignoring Col2 with full dates, 1,519 with months.

Decision: key on the monthly period. The model sees only the period index, so two rows that differ only by day of month are identical feature vectors and must share a group.

Alternatives: full-date key (reproduces the docs/01 count, but lets 11 feature-identical rows straddle a split).

Evidence: tests/unit/test_splitting.py checks both counts (1,508 at date level from the raw file, 1,519 with the build key).

Consequences: slightly stricter grouping. The holdout seed rule (D03) runs unchanged on these groups; seed 0 still qualifies, with a different rare-class composition than the date-level exploration (C3 3, C4 2, C6 1 instead of C3 1, C4 1, C6 3).

## D23. Pin pandas 2.3.3 instead of pandas 3.0.4

Status: Accepted

Context: pandas 3.0.4 crashes natively (bus error) on macOS arm64 for basic datetime operations (`pd.date_range`, `pd.to_datetime` on a datetime column), with numpy 2.3.5 and 2.4.6 alike. pandas 2.3.3 and 3.0.6 do not crash. 3.0.6 is newer than the 3-month pin rule (docs/05 section 5.2) allows.

Decision: pin pandas 2.3.3 (released 2025-09-29, wheels for CPython 3.11 to 3.14 on all target platforms). Code stays compatible with pandas 3 (CLAUDE.md rule 12); the CI canary job runs the latest pandas.

Evidence: reproduction in a fresh venv: `pd.Series(pd.date_range("2018-01-01", periods=5))` exits with SIGBUS under pandas 3.0.4.

## D24. Compressed phase schedule

Status: Accepted (user approval, 2026-10-17)

Context: the submission deadline is 2026-10-18 23:59 ET, and the dashboard is the brief's primary deliverable.

Decision: run phases in four blocks (0+1, 2+3, 4+5, 6+7). Each block gets the full gate, a review round with the reviewer subagents, a second round when the first returns any Critical or Warning finding, and a commit at green. Every acceptance criterion and every rule in CLAUDE.md section 3 still applies.

Consequences: fewer review cycles per phase; spare capacity goes to extra audits at the end.

## D25. Network use limited to the optional LLM call

Status: Accepted (user approval)

Context: CLAUDE.md rule 7 forbids network access at runtime. The brief requires an LLM-powered explanation agent.

Decision: the pipeline (reading, validation, prediction, reports, offline explanations) never touches the network. The single exception: when a user clicks Explain, the agent may call Claude (direct key, or the hosted relay). Any failure falls back to the deterministic offline explanation built from the same payload.

## D26. Labeled files in tests before the holdout evaluation

Status: Accepted (user approval)

Context: edge case F13 scores the full raw file, which contains holdout rows with labels, and `holdout_with_labels.xlsx` exists for dashboard demos. CLAUDE.md rule 1 reserves holdout labels for `evaluate-holdout`.

Decision: before the holdout evaluation, tests that touch labeled files containing holdout rows assert structure only (row counts, presence of metric keys), never metric values. No report or decision uses them. `holdout_with_labels.xlsx` joins `holdout_labels.csv` in the gate's holdout guard grep, together with `load_labeled_raw`, direct reads of `paths.raw_data`, and the token internals; only `splitting.py`, `cli.py`, the holdout evaluation module, and `eda.py` (label-free schema audit and raw label mapping on the full file, sanctioned by docs/03 Phase 2) may use them. Two documents show holdout class counts outside `reports/holdout/` by design: `split_manifest.json` (docs/04 section 2.2) and the full-file raw label mapping. Both hold counts only, never metrics or predictions.

## D27. European number format only when unambiguous

Status: Accepted

Context: docs/05 section 2.5 treats `^\d{1,3}(\.\d{3})+(,\d+)?$` as European notation, so `12.500` and `1.234` read as 12,500 and 1,234, which is 1000 times too large for a dot-decimal file with three decimals (code review finding).

Decision: read European notation only with a decimal comma (`1.234,56`) or two or more dot groups (`1.234.567`). `1.234` and `12.500` stay dot-decimal numbers.

Evidence: tests/unit/test_values.py (`1.234`, `12.500`, `1.234.567`, `1.234,56`), tests/adversarial/test_value_parsing.py::test_three_decimal_amount_in_dot_decimal_column_keeps_magnitude.

## D28. Day-first date inference counts only valid dates

Status: Accepted

Context: docs/05 section 2.5 reads a column day-first when any value has a first part above 12. One typo (`45/08/2018`) would then silently flip every other date in the column (edge-case breaker finding).

Decision: a value counts as day-first evidence only when it is a valid day-first date (`31/08/2018`), and as month-first evidence only when it is a valid month-first date (`08/31/2018`). The majority wins; mixed evidence raises the WARNING `DATE_ORDER_MIXED`. Without evidence the column reads month-first with the INFO issue `DATE_ORDER_ASSUMED_MONTH_FIRST`, as before.

Evidence: tests/adversarial/test_value_parsing.py::test_invalid_date_does_not_flip_other_rows_to_day_first, tests/edge_cases/test_value_level.py::test_v12_*.

## D29. LightGBM search space excludes reg_lambda = 0

Status: Accepted (2026-10-17, before the full tuning run)

Context: docs/04 section 7 lists `reg_lambda` in (0, 1, 2, 5). The quick tuning run picked a configuration with `reg_lambda=0` (15 leaves, min_child_samples 3) because its mean CV accuracy ranked first, yet on one fold it predicted Category_1 for every row (accuracy 0.8842, the majority rate).

Decision: search `reg_lambda` in (1, 2, 5) only. Without L2 regularization, a class with one or two training rows has a near-zero hessian, so its leaf values explode and dominate the logits.

Evidence: same configuration on the 5 primary folds (seed 0). reg_lambda 0: fold accuracy 0.951, 0.919, 0.884, 0.957, 0.951; largest raw score 51,830. reg_lambda 1: 0.949, 0.944, 0.940, 0.953, 0.943; largest raw score 11.8. reg_lambda 2: largest raw score 10.9.

Consequences: a numerically stable tree member, and a mean CV accuracy that no longer hides a collapsed fold. Test: tests/unit/test_reference_flags_selection.py::test_tree_space_keeps_l2_regularization.

## D30. Fallback model as a tree-less blend; artifact file names

Status: Accepted

Decision: the fallback is `LogOddsBlendClassifier(tree=None)`: a temperature-scaled logistic regression with the same explanation code path and zero LightGBM objects in its pickle (test e01 unpickles it with LightGBM blocked). Artifacts are named `model_champion.joblib` and `model_fallback.joblib` (docs/02 used `model_blend.joblib`), because the champion rule could have chosen a logistic model.

## D31. Explanation payload details beyond docs/02 and docs/04

Status: Accepted

- `explain_row(bundle, result, row_number)` explains one row of a `PredictionResult` (docs/02 sketched `explain_rows(bundle, frame, row_numbers)`); the dashboard explains one row per click, so a batch API added nothing. `PredictionResult.frame` and `row_flags` are public fields because the agent's tools read them.
- The deterministic fallback text adds one sentence to the docs/04 section 13 template when the class base rate carries more of the margin than the row's own values ("Most of the margin comes from the base rate ..."). Without it, rows such as holdout row 1 read as contradictory: every input column pushes toward the runner-up, yet the prediction holds.
- One-hot features for values seen fewer than twice are shown as "(rare or unseen value)" instead of sklearn's internal name.

## D32. Explanation agent: tools, engines, grounding, relay

Status: Accepted (user approved Claude Haiku 5.5 and the relay on 2026-10-17)

Decision: the agent runs only when a user clicks Explain on one row. Engines in order: the user's own `ANTHROPIC_API_KEY` (dashboard field or environment, optional `ANTHROPIC_WORKSPACE_ID`), then the hosted relay (`relay/`, deployed on Vercel; the project owner's key lives only in Vercel's encrypted environment), then a deterministic offline engine. Claude (`claude-haiku-5-5`, effort low, at most 6 turns, 25 s timeout) calls four read-only tools: the exact explanation payload, training evidence for any value or token, the most similar training rows (cosine on the model's own feature space), and a what-if re-score. Its answer is accepted only when every number it cites appears in a tool result (percentages within 1 point); otherwise, and on refusal, truncation, timeout, or any client error, the offline engine answers from the same tools and the reason is shown.

Relay safeguards: shared token header, fixed model, output cap 4,000 tokens, the four tool names only, request size and message count limits, a per-instance rate limit, and the Anthropic workspace spend limit.

Alternatives: shipping a key in the ZIP (exposes the key); requiring the grader's key (most graders would see only the offline engine); a local model (large download, slow on CPU, unreliable tool use); LangChain (adds dependencies without changing the key or network problem).

Evidence: tests/integration/test_agent_and_server.py (scripted fake client: tool loop, grounding rejection, refusal, truncation, turn limit, client failure, connection resolution order); relay checked live: health OK, 401 without token, unknown tool rejected, request forwarded to Anthropic. Live end to end (2026-10-17, after the owner supplied a key that needs no workspace header): three holdout-feature rows explained by claude-haiku-5-5, two through the hardened relay and one direct, 3 to 5 s each, every cited number grounded, no fallback. Hardening after the code review: the relay rejects requests unless they carry exactly the four agent tools and the agent's system prompt (sha256 checked; a test keeps the hash in sync), forwards only whitelisted fields, and refuses to run without RELAY_TOKEN.

## D33. Layering of the report writer

Status: Accepted (code review, block B to D)

Context: docs/06 section 7.1 places `reports` in the top layer, but every analysis command (eda, benchmark, tune, select) writes its own report files through it, and predictions used its JSON helper.

Decision: `to_jsonable` moves to a bottom-layer module `jsonsafe.py`, so prediction, explanation, and the agent never import the report layer. `reports.py` is a file-writing utility (tables, JSON, manifest) that the producer commands call through one helper, `write_reports`; it imports only config, exceptions, and jsonsafe, so no cycle is possible.

## D34. Open question for the project owner: champion rule when no candidate is eligible

Status: Open (does not affect the current result)

docs/04 section 8 does not define the case where no candidate passes both guardrails (possible when the best-F1 candidate fails the out-of-time floor and vice versa). `select` now stops with a clear `NO_ELIGIBLE_CANDIDATE` error and the decision trail instead of crashing. Defining a fallback rule changes the champion rule, which needs the owner's decision (CLAUDE.md section 8). Two related notes for the model card: the fallback model is "the best logistic candidate" by CV accuracy without guardrails, as docs/04 states (logistic_p0, whose macro F1 is below the champion guardrail); and candidate scores reuse seed 0's folds, which tuning also used, so they are mildly optimistic; nested CV measures that optimism at -0.0008.

## D35. Holdout labels are recognized by row, not by file bytes

Status: Accepted (edge-case review, round 2; extends D26)

Context: the service withheld metrics only when an upload was byte-identical to `holdout_with_labels.xlsx`. A CSV re-save, a copy pasted into another workbook, or the raw `Challenge_Data.xls` scored holdout labels in the dashboard, which CLAUDE.md rule 1 forbids before `evaluate-holdout`. The raw file also showed in-sample accuracy for training rows with no caveat.

Decision: `ledgerclf/overlap.py` fingerprints rows in model form (normalized text of the five text columns, amount in cents, period index). Before any label is scored, the service matches the upload against the fingerprints of the unreleased holdout rows (read from the label-free `holdout_features.xlsx`; the labeled file is never opened) and of the training rows. If the features file is missing while the holdout is unreleased, the guard fails closed: every label in the upload stays unread (code review, cycle 2). Matching holdout rows are predicted, but their labels are blanked before scoring (INFO `HOLDOUT_METRICS_WITHHELD`); matching training rows are scored with WARNING `TRAINING_ROWS_IN_UPLOAD`. Both checks run only for uploads that carry a label column. After the holdout evaluation, only the training check remains.

Consequences: label-handling edge tests attach synthetic labels to holdout feature rows with a new Col2, so the rows match neither set (`unseen_rows` fixture). F13 now asserts the stricter count: the raw file scores 5,309 training rows and withholds 590.

Evidence: tests/adversarial/test_holdout_guard.py, tests/adversarial/test_report_messages.py, tests/edge_cases/test_prediction_level.py::test_f13...; 531 distinct holdout fingerprints, 4,688 training fingerprints, overlap 0.

## D36. Feature set kept after the ablation

Status: Accepted

Context: the ablation on `logistic_p0` (`reports/benchmark/ablation.csv`) shows that dropping the Col6 whole-value one-hot raises macro F1 by 0.021 and accuracy by 0.0012. The gain comes from about one more Category_6 row per seed (recall 0.212 to 0.303, 11 rows in total).

Decision: keep the specified feature set (docs/04 section 3). A check of the champion configuration (log-odds blend p0.25, the same 3 seeds x 5 grouped folds, training portion only) points the other way: without `col6_value`, accuracy falls from 0.9539 to 0.9532 and macro F1 from 0.7475 to 0.7326. The full run reproduced the recorded champion scores exactly.

Evidence: the logistic baseline result is in ablation.csv. The champion check is a one-off script run during the Phase 8 notebook review; its result is recorded here and in notebooks/03_modeling.ipynb section 3.

## D37. Runtime dependencies for the API and the agent

Status: Accepted (project owner approval, 2026-10-17, when the custom dashboard and the Claude agent were agreed)

Context: docs/02 section 12 lists the modeling stack only. Part 2 needs an HTTP server for the dashboard and a Claude client for the agent. CLAUDE.md section 8 requires the owner's approval before adding a dependency.

Decision: add exact pins for fastapi, starlette, pydantic, uvicorn, python-multipart (the API), anthropic and httpx (the agent), and truststore (operating-system certificates behind corporate TLS proxies). Their two compiled transitive dependencies, pydantic-core and jiter, are pinned too, so every platform installs the same binary wheels. The modeling package imports none of these: the CLI, the pipeline and the offline explanations work without them, and `agent.py` imports `anthropic` only inside the client factory. Rejected: LangChain (adds dependencies without solving the key or network problem, see D32); Streamlit (no custom UI).

Evidence: `python tools/check_wheels.py` covers 29 pins x Python 3.11 to 3.14 x Windows x64, macOS arm64, macOS x86_64 and Linux x86_64, with no gaps (2026-10-17).
