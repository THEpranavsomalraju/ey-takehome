# PROGRESS

Living log. Updated after every gate run and every session. Format rules: docs/06 section 10.

## Current phase

Block E: AI agent, relay, API server (Part 2 backend). Status: IN PROGRESS

Schedule per DECISIONS.md D24: blocks 0+1, 2+3, 4+5, 6+7, then the agent, relay, and API server.

## Phase status overview

| Phase | Name | Status | Commit |
|---|---|---|---|
| 0 | Scaffold and environment | DONE (block A) | see git log "feat: scaffold" |
| 1 | Data contract, labels, split (item E) | DONE (block A) | see git log "feat: scaffold" |
| 2 | Exploratory analysis (item F) | DONE (block B) | see git log "feat: exploratory analysis" |
| 3 | Preprocessing and features | DONE (block B) | see git log "feat: exploratory analysis" |
| 4 | Models, CV runner, benchmark | DONE | c1c54e3 |
| 5 | Tuning, blend, selection, calibration (items A to D) | DONE | see git log "feat: logged tuning" |
| 6 | Final training, artifacts, explanations | DONE | see git log "feat: logged tuning" |
| 7 | Prediction service and robustness | DONE (edge-case breaker round 2 pending) | see git log "feat: logged tuning" |
| 8 | Modeling notebook, docs, holdout evaluation | NOT STARTED (holdout waits for user approval) | |
| 9 | Clean room, cross-platform, packaging, final review | NOT STARTED | |

## Finishing items

| Item | Description | Status | Evidence |
|---|---|---|---|
| A | Champion confirmed across 3 seeds by the decision rule | DONE | logodds_blend_p0.25; reports/selection/champion_decision.json; D11 |
| B | Logged hyperparameter search plus nested CV | DONE | reports/tuning/*_trials.csv (30 + 40 + 10 trials), nested_cv.json (0.9548 +- 0.005) |
| C | Log-odds blend with exact additive contributions | DONE | max additivity gap 7e-12 over 590 holdout-feature rows; test_x05, test_blend_contributions_are_exactly_additive; blend_comparison.csv (+0.0008 vs probability blend) |
| D | Calibration check and temperature scaling | DONE | T=1.39, ECE 0.019 -> 0.0066; threshold 0.8 (D14, D17) |
| E | Holdout seed fixed by written rule before modeling | DONE | D03 committed before the split; `data/splits/split_manifest.json` (seed 0, 590 rows) |
| F | Full EDA coverage | DONE | 25 reports in reports/eda (Cramer's V, mutual information, outliers, missing values), notebooks/01_eda.ipynb |

## Block A plan (Phases 0 and 1)

1. Pins per docs/05 section 5.2 verified with `tools/check_wheels.py`.
2. Bottom layer: `config`, `exceptions`, `logging_setup`, `labels`, `values` (cell parsers), `schema`.
3. `io.read_table` with byte-level format sniffing, sheet choice, encoding and delimiter detection.
4. `splitting`: group key, written seed rule, persisted split, holdout files, guard.
5. CLI (`doctor`, `split`), doctor checks 1 to 4, 7, 8. Gate, CI, Dockerfile, README skeleton.
6. Tests: unit, edge cases F/S/V/L/R at the validation layer, static and subprocess portability.

## Acceptance criteria (Phase 0)

- [x] `python tools/check_wheels.py` prints a full matrix with no gaps (evidence: 19 pins x Python 3.11-3.14 x win_amd64, macOS arm64, macOS x86_64, manylinux x86_64, all "ok"; checker verified to detect a known gap, numpy 2.5.1 on 3.11)
- [x] `python -m ledgerclf doctor` runs from the repo root and from another directory, ASCII output only (evidence: tests/portability/test_subprocess.py::test_doctor_runs_from_another_directory, ::test_cli_output_survives_cp1252_console)
- [x] `python tools/gate.py` passes (evidence: full gate PASS in 22s)
- [x] Importing every `ledgerclf` module works with LightGBM blocked (evidence: tests/portability/test_subprocess.py::test_every_module_imports_with_lightgbm_blocked)
- [x] Static portability tests green (evidence: tests/portability/test_static.py, 7 tests)

## Acceptance criteria (Phase 1)

- [x] Label mapping reproduces docs/01 section 3 exactly (evidence: tests/unit/test_labels.py::test_label_mapping_reproduces_docs_counts)
- [x] Holdout seed rule recorded in DECISIONS.md before the split commit, rule result in `split_manifest.json` (evidence: commit c732a06 holds D03 and D22 before any split file existed)
- [x] Holdout: 580 to 600 rows, at least 1 row of C3, C4, C6, zero C5, C1 and C2 shares within 0.01 (evidence: seed 0, 590 rows: C1 521, C2 63, C3 3, C4 2, C5 0, C6 1; test_persisted_split_satisfies_rule)
- [x] No group appears on both sides (evidence: tests/unit/test_splitting.py::test_no_group_on_both_sides)
- [x] `split` run twice produces byte-identical split files (evidence: test_split_in_fresh_copy_is_byte_identical_to_repo; xlsx timestamps pinned)
- [x] Holdout files read back through `read_table` and `coerce_values` without ERROR issues (evidence: test_holdout_files_read_without_errors)
- [x] Reading the raw `.xls` and the holdout `.xlsx` and `.csv` produces identical coerced values for the same rows (evidence: test_raw_and_holdout_files_coerce_identically)

## Acceptance criteria (Phase 2)

- [x] Every `reports/eda/` file from docs/04 section 15 exists and appears in the manifest (evidence: tests/unit/test_eda_reports.py::test_run_eda_writes_every_catalog_report, tests/integration/test_manifest.py)
- [x] Full-file structural facts match docs/01 exactly (evidence: test_structural_facts_match_docs: 5,899 rows, Col2 types 3,526/2,370/3, 326 scientific-style values, 11 raw labels)
- [x] Training-portion statistics close to docs/01 (evidence: test_training_shares_close_to_docs; C1 0.8842 vs 0.8840)
- [x] Notebook executes top to bottom with nbclient in under 2 minutes (evidence: tools/run_notebooks.py, 1.4s)
- [x] Cramer's V and mutual information present with interpretation (evidence: notebooks/01_eda.ipynb section 9)

## Acceptance criteria (Phase 3)

- [x] Pipeline fits on training rows only and transforms holdout rows without error (evidence: tests/unit/test_features.py::test_pipeline_transforms_holdout_with_unseen_values, ::test_fit_uses_only_given_rows)
- [x] Output sparse, finite, no NaN (evidence: test_pipeline_output_is_sparse_and_finite, test_pipeline_handles_missing_and_extreme_values)
- [x] Period clipping (2030 to training max) and amount flags correct (evidence: test_period_encoder_clips_to_training_range, test_amount_flags)
- [x] Every output feature maps to exactly one group (evidence: test_every_feature_maps_to_exactly_one_group)
- [x] Notebook executes cleanly (evidence: notebooks/02_preprocessing.ipynb, 1.0s)

## Acceptance criteria (Phases 4 to 7)

- [x] Leaderboard holds every docs/04 section 4 model with mean and std over 3 seeds and fit seconds (reports/benchmark/leaderboard.csv)
- [x] All reference ranges pass (tools/check_reference_ranges.py: 11 ranges and 3 directions; unseen-vendor reference compared like for like with balanced logistic, Exp 7)
- [x] Random-split accuracy exceeds grouped accuracy (0.9636 > 0.9493)
- [x] Class weight dial reproduces the direction (false rare alarms 1 -> 15 as power rises)
- [x] benchmark --quick under 3 minutes (72 s); full run recorded
- [x] Additivity within 1e-6 for training rows and every holdout-feature row (max gap 7e-12)
- [x] blend_comparison.csv exists; log-odds blend 0.0008 above the probability blend
- [x] Trial logs for all three searches, search space, nested_cv.json
- [x] Champion by the pre-registered rule, trail in champion_decision.json, D10 and D11 updated
- [x] Calibration summary, temperature, threshold by rule, D14 and D17 updated
- [x] Accuracy-by-flag report and D18 policy
- [x] model_fallback.joblib unpickles with LightGBM blocked (test_e01_fallback_bundle_unpickles_without_lightgbm)
- [x] Golden file matches predict_file on holdout .xlsx and .csv (test_r04, tools/compare_golden.py: max diff 5e-11)
- [x] Reproducibility: blend fits are deterministic (test_blend_is_deterministic); split byte-identical
- [x] Every explanation payload passes json.dumps, schema keys match, additivity within 1e-6 (test_x03, test_x05)
- [x] metadata.json holds every docs/02 section 6 field
- [x] Every catalog ID (79) has a passing test; gate check 8 PASS
- [x] Performance budgets met: 590 rows 195 ms, 100,000 rows within 120 s (test_p03), one explanation 27 ms
- [x] Fallback path end to end with LEDGERCLF_FORCE_FALLBACK=1 and with LightGBM blocked (test_e12, test_e01)
- [x] doctor shows all checks (15 PASS), exit codes correct

## Iteration log

| Cycle | Date | Gate | Critical | Warning | Suggestion | Notes |
|---|---|---|---|---|---|---|
| A1 | 2026-10-17 | full PASS 22s | 6 (code) + 9 (edge) | 11 (code) + 3 (leakage) + 14 (edge) | 9 + 3 | Block A review round: every Critical and Warning fixed (details below) |
| B1 | 2026-10-17 | full PASS 33s | 0 (leakage) | 3 (leakage) | 1 | Block B leakage audit: all fixed. Code review round interrupted by a usage limit; rerun with block C/D |
| CD1 | 2026-10-17 | full PASS 134s | - | - | - | Phases 4 to 7 built; 431 tests; review round next |

## Findings during block C/D

- reg_lambda = 0 in the LightGBM search space made one-row classes explode (raw scores up to 51,830; one fold collapsed to the majority rate). Removed from the space (D29).
- Log-odds blending at power 0 loses rare-class recall; the macro F1 guardrail of the pre-registered rule excluded that candidate.
- Explanation text now names the base rate when the class prior carries the prediction.

## Findings and fixes during block A

- pandas 3.0.4 crashes natively on macOS arm64 (bus error in `pd.date_range`); pinned pandas 2.3.3 (D23).
- Duplicate count ignoring Col2 is 1,519 with the spec's monthly key versus 1,508 with full dates (docs/01); key kept on months (D22).
- `Unnamed: 0` index artifacts were reported as extra columns (S14 test caught it); fixed.
- openpyxl stamps save time into xlsx files; timestamps now pinned so split reruns are byte-identical.
- `python -m ledgerclf` from other folders needs the package registered: README install adds `pip install -e . --no-deps`.

## Block A review round: what was fixed

Code reviewer (C6 W11 S9), leakage auditor (C0 W3 S3), edge-case breaker (C9 W14, 35 adversarial tests). All Critical and Warning items fixed; the 35 adversarial tests pass.

- Row numbers: blank CSV lines and blank leading rows now keep true spreadsheet row numbers; row_id derives from the sheet row.
- Amounts: sign before the currency (`-$1,234.50`), unambiguous European format only (`12.500` stays 12.5, D27), invisible characters ignored.
- Dates: only valid dates count as day-first evidence (D28); `2018.08` and numeric `20180831` parse; mixed orders warn.
- Reader: CSV field cap raised (long cells truncate instead of crashing), unclosed quotes re-read line by line with a warning, rows with extra fields flagged, comma-only trailing lines ignored for the row limit, header-only template sheets and hidden sheets lose to visible data sheets, xlsb, HTML-as-xls, encrypted xls, directories, str paths, NUL padding, and lying BOMs all give clear outcomes.
- Headers: invisible format characters removed, unrecognized header rows and repeated header rows detected, look-alike characters explained.
- Split: `split` refuses to change a persisted split without `--force`; `load_split` verifies hashes; D03 result recorded.
- Guard: gate also watches `load_labeled_raw`, raw-file reads, and token internals; the guard test uses a fake project.
- Doctor: `--require-model` fails without model checks; empty error messages no longer crash it.
- Hygiene: constants moved to config, docstrings completed, period formula shared, deterministic xlsx writer and hashing moved to io.

## Open issues

(none)

## Declined suggestions

(none yet)
