# PROGRESS

Living log. Updated after every gate run and every session. Format rules: docs/06 section 10.

## Current phase

Block B (Phases 2 and 3): Exploratory analysis and feature pipeline. Status: IN PROGRESS

Schedule per DECISIONS.md D24: blocks 0+1, 2+3, 4+5, 6+7, then the agent, relay, and API server.

## Phase status overview

| Phase | Name | Status | Commit |
|---|---|---|---|
| 0 | Scaffold and environment | DONE (block A) | see git log "feat: scaffold" |
| 1 | Data contract, labels, split (item E) | DONE (block A) | see git log "feat: scaffold" |
| 2 | Exploratory analysis (item F) | IN PROGRESS | |
| 3 | Preprocessing and features | NOT STARTED | |
| 4 | Models, CV runner, benchmark | NOT STARTED | |
| 5 | Tuning, blend, selection, calibration (items A to D) | NOT STARTED | |
| 6 | Final training, artifacts, explanations | NOT STARTED | |
| 7 | Prediction service and robustness | NOT STARTED | |
| 8 | Modeling notebook, docs, holdout evaluation | NOT STARTED (holdout waits for user approval) | |
| 9 | Clean room, cross-platform, packaging, final review | NOT STARTED | |

## Finishing items

| Item | Description | Status | Evidence |
|---|---|---|---|
| A | Champion confirmed across 3 seeds by the decision rule | | |
| B | Logged hyperparameter search plus nested CV | | |
| C | Log-odds blend with exact additive contributions | | |
| D | Calibration check and temperature scaling | | |
| E | Holdout seed fixed by written rule before modeling | DONE | D03 committed before the split; `data/splits/split_manifest.json` (seed 0, 590 rows) |
| F | Full EDA coverage | | |

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

## Iteration log

| Cycle | Date | Gate | Critical | Warning | Suggestion | Notes |
|---|---|---|---|---|---|---|
| A1 | 2026-10-17 | full PASS 22s | - | - | - | Block A built; review round launched in background |

## Findings and fixes during block A

- pandas 3.0.4 crashes natively on macOS arm64 (bus error in `pd.date_range`); pinned pandas 2.3.3 (D23).
- Duplicate count ignoring Col2 is 1,519 with the spec's monthly key versus 1,508 with full dates (docs/01); key kept on months (D22).
- `Unnamed: 0` index artifacts were reported as extra columns (S14 test caught it); fixed.
- openpyxl stamps save time into xlsx files; timestamps now pinned so split reruns are byte-identical.
- `python -m ledgerclf` from other folders needs the package registered: README install adds `pip install -e . --no-deps`.

## Open issues

(none)

## Declined suggestions

(none yet)
