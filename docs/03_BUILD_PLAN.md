# 03. Build Plan

Ten phases. Run them in order. Each phase ends with the loop from `docs/06_QUALITY_LOOP.md` and a stop for the user. Never start the next phase on a red gate.

Format of every phase: goal, documents to read first, tasks, deliverables, acceptance criteria, commit message. Copy the acceptance criteria into `PROGRESS.md` and tick each one with evidence.

## The six finishing items

These six items complete the plan. Each has a home phase and a hard acceptance test.

| Item | What | Phase | Spec |
|---|---|---|---|
| A | Confirm the champion across 3 seeds: unweighted blend versus the 0.25 weight setting (plus logistic candidates), decided by accuracy under the pre-registered rule | 5 | docs/04 section 8 |
| B | Logged hyperparameter search for both models, with results shown in reports, plus nested CV | 5 | docs/04 section 7 |
| C | Blend in log-odds space, with per-feature contributions summing to the blended score | 5 | docs/04 section 6 |
| D | Probability calibration check and temperature scaling, since confidence routing and the agent both show probabilities | 5 | docs/04 section 9 |
| E | Holdout seed fixed by a written rule before any modeling | 1 | docs/04 section 2.2 |
| F | Full EDA coverage: distributions, correlations (Cramer's V, mutual information), outliers, missing values | 2 | docs/04 section 15 |

Scope guard: skip new model families, deep learning, stacking beyond the two-model blend, and further feature engineering. Spend effort on correctness, robustness, and clarity.

---

## Phase 0. Scaffold and environment

**Goal:** a clean, pinned, cross-platform skeleton with working tooling before any data work.

**Read first:** CLAUDE.md, docs/02 (sections 2, 9 to 12), docs/05 (sections 5, 8, 10), docs/06 (sections 2, 7).

**Tasks:**

1. Create the layout from docs/02 section 2 with empty modules carrying module docstrings.
2. `pyproject.toml`: project metadata, `requires-python = ">=3.11,<3.15"`, ruff config (line length 100, rules `E F W I B UP SIM N D RUF`, Google docstring convention, tests exempt from `D`), pytest config (markers `slow edge portability golden`, `--strict-markers`, timeout default 120 s), mypy config (`ignore_missing_imports = true`, `disallow_untyped_defs = true` for `ledgerclf`), coverage config (`fail_under = 90`, source `ledgerclf`).
3. Choose exact pins per docs/05 section 5.2. Write `requirements.txt`, `requirements-dev.txt`, `requirements-research.txt`.
4. `tools/check_wheels.py`. Run the check. Adjust pins until the matrix shows no gaps.
5. `config.py` with frozen dataclasses and `get_settings()`. `exceptions.py`. `logging_setup.py` with the ASCII-safe formatter.
6. `cli.py` with argparse subcommands from CLAUDE.md section 5 (stubs raising a clear "not implemented in this phase" message), `__main__.py`.
7. `doctor.py` implementing checks 1 to 4 and 8 (the rest arrive later).
8. `tools/gate.py` per docs/06 section 2 (steps for later phases skip with a note until their inputs exist).
9. `.gitignore` (venv, caches, `dist/`, `catboost_info/`, `.DS_Store`, `*.pyc`, `.ipynb_checkpoints/`). Keep `artifacts/`, `reports/`, `data/` tracked.
10. `.github/workflows/ci.yml` per docs/05 section 8. `Dockerfile` per docs/05 section 5.9.
11. Portability static tests (docs/05 section 6) from day one.
12. `README.md` skeleton with install steps for Windows, macOS, Linux. `PROGRESS.md` updated.

**Deliverables:** skeleton repo, pins, tooling, CI file, Dockerfile, README skeleton.

**Acceptance criteria:**

- [ ] `python tools/check_wheels.py` prints a full matrix with no gaps
- [ ] `python -m ledgerclf doctor` runs from the repo root and from another directory, ASCII output only
- [ ] `python tools/gate.py` passes (format, lint, types, tests, portability checks)
- [ ] Importing every `ledgerclf` module works with LightGBM blocked (subprocess test)
- [ ] No absolute paths, no text `open()` without encoding, no non-ASCII in `.py` files (static tests green)

**Commit:** `chore: scaffold project, pinned environment, tooling, CI`

---

## Phase 1. Data contract, labels, and split (item E)

**Goal:** a validated reading path shared by training and inference, clean labels, and a persisted leakage-safe holdout.

**Read first:** docs/01 (sections 1 to 6, 11), docs/04 (sections 1, 2), docs/05 (sections 2, 3, 4).

**Tasks:**

1. `io.read_table` with format sniffing, sheet selection, encoding and delimiter sniffing, limits.
2. `schema.resolve_columns` and `schema.coerce_values` per docs/05 section 2. `ValidationIssue`, `ValidationReport`.
3. `labels.normalize_label` and `normalize_labels` with the mapping table.
4. `splitting.build_group_key`. Verify: 680 duplicate rows on Col1 to Col7, 1,508 duplicates ignoring Col2, one conflicting group.
5. Write the holdout seed rule into DECISIONS.md (D03) before running anything. Then implement `select_holdout_seed` and `make_holdout_split`.
6. `python -m ledgerclf split`: writes `data/splits/*`, `data/holdout/*` (raw values, Col2 as text).
7. `splitting.load_training_frame`: the single entry point for modeling code. Training rows only.
8. Holdout guard: a function in `splitting.py` which raises `HoldoutAccessError` unless called from `evaluate-holdout` (pass an explicit token object created only by the `evaluate-holdout` command). Test the guard.
9. Edge-case tests for F, S, V, L, R families as far as reading and coercion go (prediction-dependent assertions arrive in Phase 7, mark them `xfail(strict=True)` with a reason until then).

**Acceptance criteria:**

- [ ] Label mapping reproduces docs/01 section 3 exactly (11 raw values to 6 classes, counts match)
- [ ] Holdout seed rule recorded in DECISIONS.md before the split commit, rule result in `split_manifest.json`
- [ ] Holdout: 580 to 600 rows, at least 1 row of C3, C4, C6, zero C5, C1 and C2 shares within 0.01
- [ ] No group appears on both sides (test)
- [ ] `split` run twice produces byte-identical split files
- [ ] Holdout files read back through `read_table` and `coerce_values` without ERROR issues
- [ ] Reading the raw `.xls` and the holdout `.xlsx` and `.csv` produces identical coerced values for the same rows

**Commit:** `feat: data contract, label normalization, grouped holdout split`

---

## Phase 2. Exploratory analysis (item F)

**Goal:** every EDA table the brief asks for, computed by package functions on the training portion, rendered in notebook 01.

**Read first:** docs/01 (all), docs/04 section 15 (eda list).

**Tasks:**

1. `eda.py`: one pure function per report in docs/04 section 15 (`reports/eda/`). Schema audit and raw label mapping run on the full file (structure only). Everything else runs on training rows.
2. Correlations: bias-corrected Cramer's V matrix (Col1, Col6, pair, Col7, period, amount decile, label). Mutual information of each column with the label (raw and normalized).
3. Outliers: IQR on signed-log amount, with class mix of outliers and a note on why the build keeps them.
4. Missing values: per column, per class.
5. Duplicates: summary, group sizes, conflict groups. Accuracy ceiling.
6. `python -m ledgerclf eda` writes every file and updates `reports/manifest.json`.
7. `notebooks/01_eda.ipynb`: markdown narrative plus charts built from report files. No analysis logic in the notebook.

**Acceptance criteria:**

- [ ] Every `reports/eda/` file from docs/04 section 15 exists and appears in the manifest
- [ ] Full-file structural facts match docs/01 exactly (rows, columns, raw labels, Col2 type mix, 326 scientific-style Col2 values)
- [ ] Training-portion statistics fall close to docs/01 (shares within 1 point)
- [ ] Notebook executes top to bottom with `nbclient` in under 2 minutes
- [ ] Cramer's V and mutual information present with a short interpretation in the notebook

**Commit:** `feat: exploratory analysis reports and EDA notebook`

---

## Phase 3. Preprocessing and features

**Goal:** the feature pipeline from docs/04 section 3, with before and after views.

**Read first:** docs/04 section 3, docs/02 (features contract).

**Tasks:**

1. Custom transformers: `TextNormalizer`, `SignedLogAmount`, `AmountFlags`, `PeriodEncoder`, `ColumnConcat`, `TrainingMedianImputer`. Each with `fit`, `transform`, `get_feature_names_out`, docstring, unit tests.
2. `build_feature_pipeline(params)` returning the `ColumnTransformer`. `feature_groups()` mapping output features to source groups.
3. Snapshot reports in `reports/preprocessing/` (docs/04 section 15).
4. `notebooks/02_preprocessing.ipynb`: each transformation with a before and after table, feature space summary, and the decision id behind each step.

**Acceptance criteria:**

- [ ] Pipeline fits on training rows only and transforms holdout rows without error (unseen categories, missing values)
- [ ] Output stays sparse, finite, no NaN (test)
- [ ] Period clipping works (2030 maps to the training maximum) and amount flags compute correctly (tests)
- [ ] Every output feature maps to exactly one group (test)
- [ ] Notebook executes cleanly

**Commit:** `feat: feature pipeline, transformers, preprocessing reports`

---

## Phase 4. Models, CV runner, benchmark

**Goal:** the model zoo, the grouped CV runner with out-of-fold capture, the leaderboard, ablations, and the validation schemes.

**Read first:** docs/04 sections 2.3, 4, 5, 6.1, 14. docs/01 section 14 and 15.

**Tasks:**

1. `models.py`: factories, `PoweredClassWeight`, `VendorAccountLookup`, `LogOddsBlendClassifier` (fit, decision_function, predict_proba, predict. Contributions arrive in Phase 5).
2. `evaluation.py`: metrics, `run_grouped_cv` (3 seeds, canonical six-column probabilities, out-of-fold logits and probabilities), random CV, out-of-time, unseen-vendor CV, accuracy by seen versus unseen pair.
3. `python -m ledgerclf benchmark [--quick]`: leaderboard, per-seed table, ablation, class weight dial, leaf size effect, validation schemes.
4. `research/catboost_benchmark.py` (optional, needs `requirements-research.txt`): fold 0 only, writes `reports/research/catboost.json`. The leaderboard reads this file when present.
5. `tools/reference_ranges.json` and `tools/check_reference_ranges.py`, wired into the gate.

**Acceptance criteria:**

- [ ] Leaderboard holds every model from docs/04 section 4 with mean and std over 3 seeds and fit seconds
- [ ] All reference ranges from docs/01 section 15 pass, or each deviation has an investigation note in PROGRESS.md and user sign-off
- [ ] Random-split accuracy exceeds grouped accuracy (leakage demonstration reproduced)
- [ ] Class weight dial reproduces the direction from docs/01 Exp 9 (higher power, more out-of-time false rare alarms)
- [ ] `benchmark --quick` finishes in under 3 minutes on one core. Full run time recorded

**Commit:** `feat: model zoo, grouped CV runner, benchmark reports`

---

## Phase 5. Tuning, blend, selection, calibration (items A, B, C, D)

**Goal:** complete the four modeling finishing items with logged evidence and a champion chosen by the pre-registered rule.

**Read first:** docs/04 sections 6 to 10.

**Tasks:**

1. **Item C first:** implement `LogOddsBlendClassifier.contributions()` and the margin view. Write the additivity tests from docs/04 section 6.3 before tuning, so every later model passes them. Run the log-odds versus probability blend comparison.
2. **Item B:** `tuning.py`: linear grid, tree random search, blend weight grid, nested CV. `python -m ledgerclf tune [--quick]` writes `reports/tuning/*`.
3. Write the decision rule into DECISIONS.md (D11) before running `select`.
4. **Item A:** `selection.py`: score the four candidates (docs/04 section 8) with tuned parameters on Primary CV, 3 seeds, plus out-of-time. Apply the rule. Write `artifacts/champion.json` and `reports/selection/champion_candidates.csv`, `champion_decision.json`.
5. **Item D:** `calibration.py`: temperature on out-of-fold logits for the champion and the fallback, reliability tables, ECE, Brier, log loss before and after, threshold table, default threshold by rule.
6. Accuracy by flag on out-of-fold predictions with fold-specific reference statistics. Set flag severities by evidence. Record D18.
7. `high_confidence_errors.csv` for client label review.
8. `python -m ledgerclf select` produces everything in `reports/selection/` and updates `champion.json` with temperature and thresholds.

**Acceptance criteria:**

- [ ] Additivity holds within 1e-6 for 200 training rows and all holdout-feature rows, for every class (item C)
- [ ] `blend_comparison.csv` exists, log-odds blend within 0.003 accuracy of the probability blend or user consulted (item C)
- [ ] Trial logs for all three searches and `nested_cv.json` exist, with search space recorded (item B)
- [ ] Champion chosen by the rule, decision trail in `champion_decision.json`, DECISIONS.md D10 and D11 updated with numbers (item A)
- [ ] Calibration summary shows ECE before and after, temperature recorded, threshold chosen by rule, D14 and D17 updated (item D)
- [ ] Accuracy-by-flag report exists and D18 records the review policy

**Commit:** `feat: logged tuning, log-odds blend with exact contributions, champion selection, calibration`

---

## Phase 6. Final training, artifacts, explanations

**Goal:** production artifacts, reference statistics, golden files, and the explanation payload.

**Read first:** docs/02 sections 5 and 6, docs/04 sections 11 to 13.

**Tasks:**

1. `reference.py`, `flags.py` (final policy), `artifacts.py` (save, load, version checks, `load_best_available`).
2. `python -m ledgerclf train` per docs/04 section 11.
3. `explain.py`: payload builder, group aggregation, evidence, fallback text.
4. `python -m ledgerclf explain FILE --row N` prints the fallback text and writes the payload JSON to a path given by `--out`.
5. Parity, golden, reproducibility, and additivity tests.

**Acceptance criteria:**

- [ ] `model_fallback.joblib` unpickles in a subprocess with LightGBM blocked
- [ ] Golden file matches `predict_file` on holdout `.xlsx` and `.csv` (identical labels, probabilities within 1e-6)
- [ ] Two quick trainings produce identical predictions (reproducibility test)
- [ ] Every explanation payload passes `json.dumps`, schema keys match docs/02, additivity within 1e-6
- [ ] `metadata.json` holds every field from docs/02 section 6

**Commit:** `feat: final training, artifacts, reference stats, explanation payloads`

---

## Phase 7. Prediction service and robustness

**Goal:** the upload path and the service facade, hardened against every catalog case.

**Read first:** docs/05 (all), docs/02 section 8.

**Tasks:**

1. `predict.py`: `predict_frame`, `predict_file`, chunking, flags, review policy, metrics when labels exist, timings.
2. `service.py`: `PredictionService` per docs/02 section 8, including `auto_train` and `health()`.
3. Finish `doctor.py` (all checks from docs/05 section 10).
4. Remove every `xfail` from Phase 1. Implement the full edge-case catalog (docs/05 section 4). Property-based fuzz tests.
5. Performance tests P01 to P04. Write `reports/robustness/performance.json`.
6. Generate `reports/robustness/edge_case_matrix.csv` from the test run (catalog id, description, test name, status).
7. Run the `edge-case-breaker` subagent. Turn every finding into a test in `tests/adversarial/`, then fix the code.

**Acceptance criteria:**

- [ ] Every catalog ID has a passing test, matrix generated, no `xfail` left
- [ ] Fuzz tests pass with 200 examples per strategy
- [ ] Performance budgets met (or measured values recorded with user sign-off)
- [ ] Fallback path works end to end with `LEDGERCLF_FORCE_FALLBACK=1` and with LightGBM blocked
- [ ] `doctor` shows all checks, exit codes correct
- [ ] Edge-case-breaker run with zero open findings

**Commit:** `feat: prediction service, validation hardening, full edge-case suite`

---

## Phase 8. Modeling notebook, documentation, holdout evaluation

**Goal:** the story and the single holdout evaluation.

**Read first:** docs/04 section 11, docs/06 section 8.

**Tasks:**

1. `notebooks/03_modeling.ipynb`: split strategy, leaderboard, ablations, validation schemes, tuning results, champion decision, calibration, flags. Charts from report files.
2. Ask the user for approval. Then run `python -m ledgerclf evaluate-holdout --confirm`. Add a holdout section to notebook 03.
3. `MODEL_CARD.md`: intended use, data, model, metrics (CV, out-of-time, holdout with support counts), calibration, limitations (unseen vendors, rare classes, anonymized tokens), fallback behavior.
4. Complete `README.md`: install per OS, quick start, commands, project structure, troubleshooting (LightGBM on macOS, Python versions, Docker), how to rerun everything.
5. Complete DECISIONS.md: every entry has evidence and a final status.
6. `reports/manifest.json` complete and validated by a test (every listed file exists, every report file listed).

**Acceptance criteria:**

- [ ] All three notebooks execute top to bottom in a fresh kernel, outputs saved
- [ ] Holdout evaluated exactly once, access log has one entry (or reruns with reasons)
- [ ] README commands verified by running each one
- [ ] Manifest test passes

**Commit:** `docs: modeling notebook, holdout evaluation, model card, README`

---

## Phase 9. Clean room, cross-platform, packaging, final review

**Goal:** proof the submission runs on a machine which is not this one.

**Read first:** docs/05 sections 5, 7, 8. docs/06 sections 5 and 11.

**Tasks:**

1. `tools/package_submission.py --last LAST --first FIRST`: builds `dist/LAST, FIRST_AI Technical Assessment.zip` from an explicit include list (README, MODEL_CARD, DECISIONS, pyproject, requirements files, Dockerfile, `ledgerclf/`, `tools/`, `tests/`, `notebooks/`, `data/`, `artifacts/`, `reports/`, `research/`, `docs/`) and an explicit exclude list (venvs, caches, `.git`, `dist`, `catboost_info`, `.DS_Store`, `.claude/`, `CLAUDE.md`, `PROGRESS.md`). Ask the user whether to include `.claude/`, `CLAUDE.md`, and `PROGRESS.md`.
2. Run `tools/clean_room.py` (docs/05 section 7). Fix every finding.
3. Ask the user to push to a private GitHub repository and run CI. Fix every failing matrix cell.
4. Final `/self-review` with all four subagents, including `clean-room-tester`.
5. Global Definition of Done (docs/06 section 11).

**Acceptance criteria:**

- [ ] Clean-room report green on this machine, including the fallback step
- [ ] CI green on all 12 matrix cells plus the fallback job (canary allowed to fail, failures logged)
- [ ] ZIP name exact, size reasonable, unzip-and-run verified inside a path with spaces and a comma
- [ ] Global Definition of Done fully ticked

**Commit:** `chore: packaging, clean-room verification, final review`
