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

Status: Pending (Phase 1). Rule fixed now, result recorded after running.

Decision: iterate seeds 0 to 99 of `StratifiedGroupKFold(n_splits=10, shuffle=True)`, take the first split as holdout, accept the first seed where: holdout size 580 to 600, at least one row of each of Category_3, Category_4, Category_6, zero rows of Category_5, Category_1 and Category_2 shares within 0.01 of the full file. The rule uses label counts only, never model results.

Rationale: C5 has 2 rows, so both stay in training. Rare classes need at least one holdout row to show any holdout behavior at all.

Evidence (exploration): seed 0 qualifies (590 rows: C1 522, C2 63, C3 1, C4 1, C5 0, C6 3).

Result: (fill in after Phase 1: seed, counts, manifest hash)

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

Status: Pending (Phase 5). Candidates: 0 and 0.25.

Context: balanced weights count one C3 row as 227 C1 rows and produced 67 false rare-class alarms out of time.

Evidence (exploration): blend power 0: CV 0.9559, out-of-time 0.942. Power 0.25: 0.9541, 0.938. Power 0.5: 0.9532, 0.935. LR power 1.0: out-of-time 0.874, below the 0.887 majority rate (Exp 9, Exp 11).

Result: (fill in after Phase 5)

## D11. Champion decision rule and champion model

Status: Rule accepted now. Champion pending (Phase 5).

Rule: docs/04 section 8, recorded here before running `select`. Candidates: logistic power 0, logistic power 0.25, log-odds blend power 0, log-odds blend power 0.25.

Expectation (exploration): log-odds blend of logistic regression and LightGBM leads on accuracy in grouped CV and out of time.

Result: (fill in after Phase 5: chosen config, scores of all candidates, trail)

## D12. LightGBM minimum leaf size of 2

Status: Accepted (confirm in Phase 5 tuning)

Context: about 13 C4 rows per training fold. The default 20-row leaf minimum never forms a C4-only leaf.

Evidence (exploration): min_child_samples 20: macro_f1_5 0.6324. 2: 0.7225 (Exp 2).

## D13. CatBoost excluded from production

Status: Accepted

Evidence (exploration): 193 s per fold versus under 1 s for logistic regression, fold-0 macro_f1_5 0.635 versus 0.742 (Exp 3).

Consequences: CatBoost appears on the leaderboard as a research row only, with an optional requirements file.

## D14. Calibration by temperature scaling

Status: Pending (Phase 5)

Decision: one temperature on out-of-fold blended logits. Preserves accuracy and the additivity of log-odds contributions.

Alternatives: isotonic or sigmoid calibration (breaks exact explanations).

Result: (fill in: temperature, ECE before and after, Brier, log loss)

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

Status: Pending (Phase 5)

Rule: lowest threshold whose kept accuracy on calibrated out-of-fold probabilities reaches 0.975.

Evidence (exploration, uncalibrated LR): threshold 0.9 kept 92.2% of rows at 97.6% accuracy, flagged rows at 64% accuracy (Exp 10).

Result: (fill in)

## D18. Row flags and review policy

Status: Pending (Phase 5)

Context: unseen vendors drop accuracy to near the majority baseline (grouped-by-vendor CV 0.891). Unseen vendor-account pairs: 0.888 in grouped CV, 0.728 out of time with balanced weights.

Decision draft: critical flags (review required) for unseen vendor, all tokens unseen, missing Col1 or Col6, imputed amount or date, low confidence. Warnings for unseen account, unseen pair, unseen document type, out-of-range values.

Result: (fill in with accuracy-by-flag evidence)

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

Status: Pending (Phase 5)

Context: 132 wrong predictions with probability 0.9 or higher in exploration, mostly C1 versus C2.

Decision: publish `reports/selection/high_confidence_errors.csv` as a client review list instead of altering labels.

## D22. Group key uses the monthly period, not the full date

Status: Accepted (2026-10-17, before the split ran)

Context: docs/04 section 2.1 builds the group key from coerced display values, and the coerced display value of Col5 is the month (`YYYY-MM`). docs/01 section 11 counted duplicates on full dates. The two disagree: 1,508 duplicate rows ignoring Col2 with full dates, 1,519 with months.

Decision: key on the monthly period. The model sees only the period index, so two rows that differ only by day of month are identical feature vectors and must share a group.

Alternatives: full-date key (reproduces the docs/01 count, but lets 11 feature-identical rows straddle a split).

Evidence: tests/unit/test_splitting.py checks both counts (1,508 at date level from the raw file, 1,519 with the build key).

Consequences: slightly stricter grouping. The holdout seed rule (D03) runs unchanged on these groups.

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

Decision: before the holdout evaluation, tests that touch labeled files containing holdout rows assert structure only (row counts, presence of metric keys), never metric values. No report or decision uses them. `holdout_with_labels.xlsx` joins `holdout_labels.csv` in the gate's holdout guard grep.
