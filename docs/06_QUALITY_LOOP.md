# 06. Quality Loop, Gate, and Style

This document defines how work gets verified and when work counts as done. Follow the loop for every phase and every non-trivial change. "Fully satisfied" has one meaning here: section 5.

## 1. The iteration loop

```
PLAN      Read the phase in docs/03 and its "Read first" documents.
          Write a short plan in PROGRESS.md: steps, files touched, tests to add, risks.

BUILD     Work in small steps (one module or one feature at a time).
          Write or update tests in the same step as the code.
          Run the relevant tests after each step: python -m pytest tests/unit/test_<module>.py -q

GATE      python tools/gate.py            (fast gate, every few steps)
          python tools/gate.py --full     (before review, before phase end)
          Fix every failure. Rerun until green. Never edit a test to hide a bug.

REVIEW    /self-review: run code-reviewer, leakage-auditor, and edge-case-breaker
          in parallel. Each returns findings tagged Critical, Warning, or Suggestion.
          Fix every Critical and Warning. Fix Suggestions which improve readability
          at low risk. Record declined Suggestions with a one-line reason in PROGRESS.md.

VERIFY    Rerun GATE (--full) after fixes. Rerun REVIEW.
          Repeat GATE plus REVIEW until two consecutive cycles return
          zero Critical and zero Warning findings with a green full gate.

SCORE     Score the phase against the rubric in section 3. Every category needs 4 or more.
          Any score below 4: write the gap in PROGRESS.md, fix, return to VERIFY.

CLOSE     Tick every acceptance criterion in PROGRESS.md with evidence
          (command, report path, or test name). Update DECISIONS.md.
          Commit with the phase message. Stop and report to the user (section 6).
```

Iteration cap: 5 full VERIFY cycles per phase. On the fifth failed cycle, stop and escalate (section 6) instead of looping further.

## 2. The gate

`tools/gate.py` runs these checks in order, prints one ASCII line per check (`PASS`, `FAIL`, `SKIP` with reason), prints a summary, and exits non-zero on any `FAIL`. Checks whose inputs do not exist yet (for example reports before Phase 2) print `SKIP` with the phase where they start.

| # | Check | Command or rule | Fast | Full |
|---|---|---|---|---|
| 1 | Formatting | `ruff format --check .` | yes | yes |
| 2 | Lint | `ruff check .` | yes | yes |
| 3 | Types | `mypy ledgerclf` | yes | yes |
| 4 | Tests | `pytest -m "not slow" -q` | yes | |
| 5 | All tests with coverage | `pytest -q --cov=ledgerclf --cov-fail-under=90` | | yes |
| 6 | Portability static checks | `pytest tests/portability -q` | yes | yes |
| 7 | Doctor | `python -m ledgerclf doctor` exits 0 (add `--require-model` from Phase 6 onward) | yes | yes |
| 8 | Edge-case coverage | parse every catalog ID from the docs/05 section 4 tables (regex on the ID column), each maps to a test whose name starts with the lowercase ID | yes | yes |
| 9 | Reference ranges | `python tools/check_reference_ranges.py` | | yes |
| 10 | Manifest integrity | every report listed exists, every report file listed | yes | yes |
| 11 | Notebooks | execute all notebooks with `nbclient`, fail on any error | | yes |
| 12 | Golden predictions | `python tools/compare_golden.py --tolerance same-machine` | | yes |
| 13 | Hygiene | no `print(` in `ledgerclf/` outside `cli.py`, no `TODO`, `FIXME`, `XXX`, no commented-out code blocks (3 or more consecutive commented lines of code), no unused files in `ledgerclf/` | yes | yes |
| 14 | Holdout guard | grep confirms `holdout_labels.csv` appears only in `splitting.py` guard code, `cli.py` evaluate command, and tests | yes | yes |

The fast gate should finish in under 2 minutes. The full gate takes longer: record its duration in PROGRESS.md.

## 3. Self-review rubric

Score each category 1 to 5 at the end of every phase. A 4 means "a senior engineer would approve with no required changes". Record scores and one line of justification per category in PROGRESS.md.

| Category | 5 looks like |
|---|---|
| Readability | A reader new to the project understands any function in one pass. Names say what things are. No clever tricks. Docstrings explain why, not only what. |
| Organization | Each module has one job, matching docs/02. No circular imports. Public surface small and deliberate. Related code lives together. |
| Cleanliness | No duplication, dead code, commented-out code, stray prints, magic numbers, or leftover experiments. Consistent formatting and naming. |
| Functionality | Every acceptance criterion met with evidence. Commands work from any directory. Results reproduce. |
| Robustness | Every edge case in scope handled with a test. Errors read as plain English with a fix. No crash path. |
| Portability | All docs/05 section 5 rules hold. Clean-room and CI results green (from Phase 9). |
| Correctness of method | No leakage. Grouped evaluation everywhere. Holdout untouched. Metrics computed as defined in docs/04 section 14. |
| Documentation | Docstrings complete. DECISIONS.md, README, and notebooks match the code and the numbers in reports. |

## 4. Subagents

| Subagent | Use when | Output |
|---|---|---|
| `code-reviewer` | After every completed step set, and in every REVIEW stage | Findings by priority with file, line, and concrete fix |
| `leakage-auditor` | Any change to splitting, features, CV, tuning, calibration, reference statistics, or evaluation | Findings by priority, each tied to a rule in docs/04 or CLAUDE.md |
| `edge-case-breaker` | Phase 1 onward for reading and coercion, Phase 7 for the full service | New failing tests in `tests/adversarial/` plus a findings list |
| `clean-room-tester` | Phase 0 (wheel check) and Phase 9 (full clean room) | Step-by-step pass or fail report with logs |

Give each subagent a precise delegation prompt: the phase, the files changed, and what to focus on. Subagents start without this conversation, so include the context they need.

## 5. Stop criteria: "fully satisfied"

A phase counts as done only when all of these hold at the same time:

1. Full gate green.
2. Two consecutive REVIEW cycles with zero Critical and zero Warning findings.
3. Every rubric category scored 4 or higher with written justification.
4. Every acceptance criterion ticked with evidence in PROGRESS.md.
5. DECISIONS.md updated for every decision made in the phase.
6. Working tree committed with the phase commit message.

The final product counts as done when every phase meets the above and the global Definition of Done (section 11) holds.

## 6. Escalation and reporting

Stop and report to the user when a phase completes, when CLAUDE.md section 8 applies, or when the iteration cap hits. Report format:

```
Phase N status: DONE | BLOCKED
Gate: full gate PASS/FAIL (duration)
Review: cycles run, final findings count by priority
Rubric: category scores
Key results: 3 to 6 lines with numbers and report paths
Decisions made: ids and one line each
Open issues or questions for the user: numbered list
Next phase: name and first step
```

## 7. Code style guide

### 7.1 Structure

- One responsibility per module, matching docs/02. Move code when a module starts doing two jobs.
- Functions short enough to read in one screen (aim for 30 lines or fewer, hard limit 60 excluding docstring). Split long functions into named steps.
- Keep the public surface small. Prefix private helpers with an underscore.
- No circular imports. Dependency direction: `config, exceptions, schema, labels` at the bottom, then `io, splitting, features, models, reference, flags`, then `evaluation, tuning, calibration, selection, explain, artifacts, predict`, then `service, doctor, reports`, then `cli`.

### 7.2 Naming

- Modules and functions: `snake_case` verbs for actions (`build_group_key`, `load_bundle`), nouns for values.
- Classes: `PascalCase` nouns. Constants: `UPPER_SNAKE_CASE`.
- Column names only through constants in `schema.py`. Never type `"Col4"` outside `schema.py` and tests.
- Avoid abbreviations except standard ones (`cv`, `oof`, `df` only in short local scopes, `proba`).

### 7.3 Types and docstrings

- Type hints on every function signature, including return types. Use `from __future__ import annotations`.
- Google-style docstrings on every public module, class, and function: one-line summary, blank line, details on why, `Args`, `Returns`, `Raises`.
- Dataclasses (frozen where possible) for structured values. No dicts with fixed key sets passed between modules, except JSON payloads at the edge.

### 7.4 Errors

- Raise `LedgerClfError` subclasses with messages in the form: what happened, why, how to fix.
- Never catch bare `Exception` except at the outermost CLI and service boundary, where the handler logs the traceback at DEBUG and raises or returns a clean error.
- Never swallow errors silently. Every caught exception gets logged or converted.

### 7.5 Data handling

- Never mutate input frames. Copy, then transform.
- Use `.loc` for assignment. No chained assignment, no `inplace=True`.
- Labels as `numpy` object arrays via `.to_numpy(dtype=object)`.
- Sort before iterating sets or dict keys built from data.
- Random state always passed explicitly from config.

### 7.6 Comments and cleanliness

- Comments explain why. Code explains what.
- No commented-out code, no `TODO` or `FIXME` at phase end, no debug prints, no unused imports, variables, or files.
- No magic numbers: thresholds, limits, seeds, and hyperparameters live in `config.py` or `champion.json`.

### 7.7 Tests

- Test names state behavior: `test_coerce_amount_parses_parentheses_as_negative`.
- Arrange, act, assert. One behavior per test. Parametrize families of inputs.
- Fixtures in `conftest.py`. Generate files in `tmp_path`.
- Tests run offline, in any order, and in parallel without shared state.
- A bug fix starts with a failing test.

## 8. Notebook standards

- Thin: import from `ledgerclf`, load report files, plot, explain. No business logic, no copied feature code.
- Structure: title, purpose, setup cell, then sections with a markdown lead-in stating the question, the chart or table, and a two to four sentence takeaway with numbers.
- Charts: matplotlib, labeled axes with units, readable titles, consistent class colors defined once in a notebook helper module (`notebooks/_style.py`).
- Executed top to bottom in a fresh kernel before each commit. Outputs saved. No errors, no warnings left visible without comment.
- Paths through `ledgerclf.config.get_settings()`.

## 9. Commits

- Conventional prefixes: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`.
- Commit at every green gate inside a phase (small commits) and at phase close (phase message from docs/03).
- Never commit a red gate on the main branch.

## 10. PROGRESS.md format

```
# PROGRESS

## Current phase
Phase N: name. Status: IN PROGRESS | DONE | BLOCKED

## Phase N plan
1. ...

## Acceptance criteria (Phase N)
- [x] criterion  (evidence: tests/unit/test_labels.py::test_mapping_counts)
- [ ] criterion

## Iteration log
| Cycle | Date | Gate | Critical | Warning | Suggestion | Notes |
|---|---|---|---|---|---|---|

## Rubric (Phase N)
| Category | Score | Justification |

## Open issues
1. ...

## Declined suggestions
- finding: reason
```

## 11. Global Definition of Done

- [ ] Every phase closed under section 5
- [ ] Six finishing items A to F complete with evidence (docs/03 table)
- [ ] Holdout evaluated once, results in `reports/holdout/`, access log consistent
- [ ] Champion chosen by the pre-registered rule, documented in DECISIONS.md and `champion_decision.json`
- [ ] Exact explanations: additivity tests green
- [ ] Full gate green, coverage at least 90%
- [ ] Every edge-case catalog ID tested and passing, matrix generated
- [ ] Clean-room report green, CI matrix green, fallback path verified
- [ ] README, MODEL_CARD, DECISIONS, notebooks consistent with reports (spot-check 10 numbers across documents)
- [ ] Submission ZIP built with the exact name and verified by unzip-and-run in an odd path
- [ ] Final rubric: every category 5, or 4 with a written reason the user accepted
