# CLAUDE.md

Project memory for Claude Code. Read this whole file at the start of every session. Then read `PROGRESS.md` to find the current phase and open issues.

## 1. Mission

Build the data, modeling, and prediction backend for the EY Tax Technology AI Technical Assessment: a supervised classifier for an anonymized accounts payable ledger with 5,899 rows, seven inputs (`Col1` to `Col7`), and one label (`ClassificationLabel`, six classes after cleanup).

In scope for this build (Part 1):

- Data exploration, label cleanup, and a leakage-safe split
- One preprocessing and feature pipeline, applied identically in training and inference
- Model benchmarking, logged tuning, selection, calibration, and a single holdout evaluation
- A prediction backend for uploaded files with full input validation and edge-case handling
- Exact per-row explanation payloads for the AI agent
- Every report the dashboard needs, plus notebooks, README, model card, and decision log

Out of scope until the user says otherwise: the dashboard frontend and the LLM prompt for the agent (Part 2). Build the service facade in `docs/02_ARCHITECTURE.md` section 8, so Part 2 plugs in without touching backend internals.

## 2. How the work gets graded

| Dimension | Grader focus | What this build must show |
|---|---|---|
| Model Accuracy | Prediction performance on test and hold-out data | Highest honest accuracy, measured with grouped CV, an out-of-time test, and one holdout evaluation |
| Technical Depth | Preprocessing, feature engineering, model selection, validation strategy, pipeline robustness | Breadth backed by evidence: leaderboard, ablations, three validation schemes, logged tuning, calibration, edge-case suite |
| Dashboard & UX | Interactivity, seamless pipeline execution, functional AI agent | Fast stateless backend, JSON-ready reports, exact explanation payloads, fallback paths |
| Communication | Data story, clear decisions | DECISIONS.md, notebooks, model card, readable code |

The user cares most about: readability, organization, cleanliness, functionality, and a submission which never breaks on the grader's machine.

## 3. Non-negotiable rules

1. **Holdout protocol.** Never read holdout labels outside the `evaluate-holdout` command. Never fit, tune, select, calibrate, or compute reference statistics on holdout rows. The demo model trains on the 90% training portion only. Ask the user before running `evaluate-holdout`.
2. **No leakage.** Fit every transformer inside the pipeline, on training folds only. No fitting on the full dataset before cross-validation.
3. **Grouped evaluation.** Every CV on this data uses `StratifiedGroupKFold` with the group key from `ledgerclf/splitting.py`. The single exception: the random-split leakage demonstration.
4. **One source of truth.** Feature logic lives only in `ledgerclf/features.py`. Constants live only in `ledgerclf/config.py`. Column names live only in `ledgerclf/schema.py`. Notebooks and the dashboard call package functions and never reimplement logic.
5. **Tests protect behavior.** Never weaken, skip, or delete a test to pass the gate. Never loosen an acceptance threshold without a DECISIONS.md entry and user approval.
6. **Raw data stays untouched.** Never modify anything under `data/raw/`.
7. **Portability first.** Follow every rule in `docs/05_ROBUSTNESS_AND_PORTABILITY.md` section 5: `pathlib` everywhere, paths resolved from `__file__`, explicit `encoding="utf-8"`, ASCII-only CLI and log output, no shell-only tooling, exact dependency pins, no network at runtime, no absolute paths.
8. **Every row gets a prediction.** Flags mark risk. Flags never withhold a prediction.
9. **Determinism.** Seeds come from `config.py`. LightGBM runs with `n_jobs=1` and `deterministic=True`. Sort before iterating any set or dict built from data.
10. **Lazy optional imports.** Import `lightgbm` only inside functions. Importing any `ledgerclf` module must succeed on a machine without LightGBM, so the fallback model loads.
11. **Library code never prints.** Use `logging`. Only the CLI prints, in ASCII.
12. **pandas 2 and 3 compatibility.** Convert label arrays with `.to_numpy(dtype=object)`. Never depend on `object` versus `string` dtype behavior.
13. **Keep the logs current.** Update `PROGRESS.md` after every gate run and every session. Record every decision in `DECISIONS.md`.

## 4. Repository map (target state)

```
<repo root>/
  CLAUDE.md                  this file
  PROGRESS.md                phase status, iteration log, open issues
  DECISIONS.md               decision log with evidence
  README.md                  install and run guide for graders (Windows, macOS, Linux)
  MODEL_CARD.md              model summary, metrics, limitations
  pyproject.toml             project metadata, ruff, pytest, mypy, coverage config
  requirements.txt           runtime pins (exact versions)
  requirements-dev.txt       test, lint, notebook pins
  requirements-research.txt  optional research extras (CatBoost)
  Dockerfile                 optional guaranteed environment
  .github/workflows/ci.yml   cross-OS and cross-Python test matrix
  docs/                      specification documents (read-only unless the user asks)
  ledgerclf/                 the Python package (flat layout, importable from repo root)
  tools/                     developer tooling: gate, clean room, wheel check, packaging
  tests/                     pytest suite (unit, edge cases, golden, portability)
  notebooks/                 01_eda, 02_preprocessing, 03_modeling (thin, executed)
  data/raw/                  Challenge_Data.xls (never modified)
  data/splits/               holdout_row_ids.csv, split_manifest.json
  data/holdout/              holdout_features.xlsx/.csv, holdout_labels.csv, holdout_with_labels.xlsx
  artifacts/                 model_blend.joblib, model_fallback.joblib, metadata.json, reference.json, golden files
  reports/                   EDA, benchmark, tuning, selection, calibration, holdout reports, manifest.json
  research/                  optional scripts not used in production (CatBoost benchmark)
  .claude/agents/            reviewer subagents
  .claude/commands/          slash commands: /phase, /gate, /self-review, /clean-room
```

Full module contracts: `docs/02_ARCHITECTURE.md`.

## 5. Commands

Environment (use Python 3.12 for development):

```
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt

# Windows PowerShell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Pipeline (all commands run from the repo root, and also from any other working directory):

```
python -m ledgerclf doctor                  environment and artifact health check
python -m ledgerclf split                   label cleanup, group key, holdout seed rule, holdout files
python -m ledgerclf eda                     EDA reports on the training portion
python -m ledgerclf benchmark [--quick]     leaderboard, ablations, validation schemes
python -m ledgerclf tune [--quick]          logged hyperparameter search and nested CV
python -m ledgerclf select                  champion confirmation, calibration analysis, thresholds
python -m ledgerclf train                   final fit on the 90%, artifacts, reference stats, golden files
python -m ledgerclf evaluate-holdout        single holdout evaluation (ask the user first)
python -m ledgerclf predict FILE [--out CSV]
python -m ledgerclf explain FILE --row N
```

Quality and delivery:

```
python tools/gate.py [--full]               quality gate (Section 7)
python tools/check_wheels.py                wheel availability across OS and Python matrix
python tools/clean_room.py                  fresh venv, odd path, install, test, predict
python tools/compare_golden.py --tolerance same-machine|cross-platform
python tools/package_submission.py --last LAST --first FIRST
```

## 6. Documents and when to read them

| Document | Read when |
|---|---|
| `docs/01_DATA_FINDINGS.md` | Before any data, EDA, or modeling work. Holds every verified fact and reference result. |
| `docs/02_ARCHITECTURE.md` | Before creating or changing any module, data structure, artifact, or report. |
| `docs/03_BUILD_PLAN.md` | At the start of every phase. Holds tasks and acceptance criteria. |
| `docs/04_MODELING_SPEC.md` | Before work on splits, features, models, tuning, selection, calibration, or explanations. |
| `docs/05_ROBUSTNESS_AND_PORTABILITY.md` | Before work on file reading, validation, prediction, tests, dependencies, or packaging. |
| `docs/06_QUALITY_LOOP.md` | Before declaring any task done. Holds the iteration loop, gate, review rubric, and style guide. |
| `DECISIONS.md` | Before changing any decision. Add an entry for every new decision. |

## 7. Working loop (summary of docs/06)

For every phase:

1. Read the phase in `docs/03_BUILD_PLAN.md` plus the documents listed under "Read first".
2. Write a short plan into `PROGRESS.md`.
3. Build in small steps. Write tests with the code, not after.
4. Run `python tools/gate.py`. Fix every failure. Repeat until green.
5. Run `/self-review` (three reviewer subagents in parallel). Fix every Critical and Warning finding.
6. Repeat steps 4 and 5 until two consecutive cycles produce zero Critical and zero Warning findings.
7. Tick every acceptance criterion in `PROGRESS.md` with evidence (command output, report path, test name).
8. Commit with the message given in the build plan. Stop and report to the user.

Never declare a phase done on a red gate, an unticked criterion, or an open Critical or Warning finding.

## 8. When to stop and ask the user

- A rule in Section 3 conflicts with a task.
- A result deviates from the reference ranges in `docs/01_DATA_FINDINGS.md` section 15 after a real investigation.
- A phase fails the gate after 5 full iterations of the loop.
- Any change to metrics, the champion decision rule, thresholds, or the holdout protocol.
- Before running `evaluate-holdout`.
- Before adding a dependency not listed in `docs/02_ARCHITECTURE.md` section 12.

## 9. Style in one paragraph

Small modules with one responsibility. Functions short enough to read in one screen. Descriptive names over comments. Type hints on every function. Google-style docstrings on every public function and class. No magic numbers outside `config.py`. No commented-out code, no stray TODOs, no dead code. User-facing error messages in plain English with the fix. Full guide: `docs/06_QUALITY_LOOP.md` section 7.
