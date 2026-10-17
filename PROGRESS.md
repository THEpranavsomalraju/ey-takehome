# PROGRESS

Living log. Claude Code updates this file after every gate run and every session. Format rules: docs/06 section 10.

## Current phase

Phase 0: Scaffold and environment. Status: NOT STARTED

## Phase status overview

| Phase | Name | Status | Commit |
|---|---|---|---|
| 0 | Scaffold and environment | NOT STARTED | |
| 1 | Data contract, labels, split (item E) | NOT STARTED | |
| 2 | Exploratory analysis (item F) | NOT STARTED | |
| 3 | Preprocessing and features | NOT STARTED | |
| 4 | Models, CV runner, benchmark | NOT STARTED | |
| 5 | Tuning, blend, selection, calibration (items A to D) | NOT STARTED | |
| 6 | Final training, artifacts, explanations | NOT STARTED | |
| 7 | Prediction service and robustness | NOT STARTED | |
| 8 | Modeling notebook, docs, holdout evaluation | NOT STARTED | |
| 9 | Clean room, cross-platform, packaging, final review | NOT STARTED | |

## Finishing items

| Item | Description | Status | Evidence |
|---|---|---|---|
| A | Champion confirmed across 3 seeds by the decision rule | | |
| B | Logged hyperparameter search plus nested CV | | |
| C | Log-odds blend with exact additive contributions | | |
| D | Calibration check and temperature scaling | | |
| E | Holdout seed fixed by written rule before modeling | | |
| F | Full EDA coverage | | |

## Phase 0 plan

(Claude Code writes the plan here before starting.)

## Acceptance criteria (Phase 0)

- [ ] `python tools/check_wheels.py` prints a full matrix with no gaps
- [ ] `python -m ledgerclf doctor` runs from the repo root and from another directory, ASCII output only
- [ ] `python tools/gate.py` passes
- [ ] Importing every `ledgerclf` module works with LightGBM blocked
- [ ] Static portability tests green

## Iteration log

| Cycle | Date | Gate | Critical | Warning | Suggestion | Notes |
|---|---|---|---|---|---|---|

## Rubric (current phase)

| Category | Score | Justification |
|---|---|---|

## Open issues

(none yet)

## Declined suggestions

(none yet)
