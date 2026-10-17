# START HERE

Your guide to running this kit with Claude Code. This file is for you, not for Claude Code.

## 1. What the kit contains

| File | Purpose | Goes where in your repo |
|---|---|---|
| `CLAUDE.md` | Project memory Claude Code loads every session: mission, rubric, hard rules, commands | repo root |
| `PROGRESS.md` | Living status log Claude Code updates after every gate run | repo root |
| `DECISIONS.md` | Decision log, seeded with 22 decisions and their evidence | repo root |
| `docs/01_DATA_FINDINGS.md` | Every verified data fact and experiment result, plus reference ranges | `docs/` |
| `docs/02_ARCHITECTURE.md` | Module layout, contracts, data structures, artifacts, service facade | `docs/` |
| `docs/03_BUILD_PLAN.md` | Ten phases with tasks and acceptance criteria, the six finishing items | `docs/` |
| `docs/04_MODELING_SPEC.md` | Splits, features, models, blend math, tuning, decision rule, calibration | `docs/` |
| `docs/05_ROBUSTNESS_AND_PORTABILITY.md` | Input contract, 79 edge cases, cross-platform rules, tests, CI | `docs/` |
| `docs/06_QUALITY_LOOP.md` | Iteration loop, gate, review rubric, stop criteria, style guide | `docs/` |
| `.claude/agents/*.md` | Four reviewer subagents: code reviewer, leakage auditor, edge-case breaker, clean-room tester | `.claude/agents/` |
| `.claude/commands/*.md` | Slash commands: `/phase`, `/gate`, `/self-review`, `/clean-room` | `.claude/commands/` |

## 2. Setup (once)

1. Install Python 3.12 and Git. Install Claude Code.
2. Create an empty folder for the project, then run `git init` inside the folder.
3. Copy everything from this kit into the folder, keeping the structure above. The `.claude` folder starts with a dot, so check hidden files got copied.
4. Create `data/raw/` and copy `Challenge_Data.xls` into the new folder.
5. Commit: `git add -A` then `git commit -m "chore: add Claude Code kit and raw data"`.
6. Start Claude Code from the project folder.

## 3. First session prompt

Paste this before any building:

```
Read CLAUDE.md, PROGRESS.md, DECISIONS.md, and every file in docs/. Do not write code yet.
Then reply with:
1. A 10-line summary of the project, the rubric, and the plan.
2. Any contradictions or gaps you found across the documents.
3. Any decision you think is wrong, with your reasoning.
4. Questions you need answered before Phase 0.
```

Answer the questions. When Claude Code finds a real contradiction, let Claude Code fix the docs and log the change in DECISIONS.md.

## 4. Run the phases

Start a fresh Claude Code session for each phase. State lives in `PROGRESS.md`, so nothing gets lost, and each phase starts with a clean context.

For each phase, type:

```
/phase 0
```

Then `/phase 1`, and so on through `/phase 9`. Claude Code builds, runs the gate, runs the reviewer subagents, fixes findings, repeats until two clean cycles, ticks the acceptance criteria, commits, and stops with a status report.

Use these any time:

- `/gate` runs the full quality gate and fixes failures.
- `/self-review` runs the gate plus three reviewers in parallel and loops until clean.
- `/clean-room` tests a fresh install in an odd path.

## 5. What to check yourself after each phase

You will defend every decision in the interview, so read the evidence, not only the summary.

| Phase | Check personally |
|---|---|
| 0 | `python -m ledgerclf doctor` works for you. The wheel matrix printout shows no gaps. |
| 1 | DECISIONS.md D03 has the holdout seed and class counts. Open `data/holdout/holdout_features.xlsx`. |
| 2 | Run notebook 01. Explain every chart to yourself in one sentence. |
| 3 | Run notebook 02. Say why each transformation exists. |
| 4 | Open `reports/benchmark/leaderboard.csv` and `validation_schemes.csv`. Compare with docs/01. |
| 5 | Read `reports/selection/champion_decision.json` step by step. Check DECISIONS.md D10, D11, D14, D17, D18. |
| 6 | Run `python -m ledgerclf explain data/holdout/holdout_features.xlsx --row 1`. Does the text make sense? |
| 7 | Upload a messy file of your own through `python -m ledgerclf predict`. Try to break the pipeline on purpose. |
| 8 | Claude Code asks before the holdout evaluation. Say yes once. Read MODEL_CARD.md fully. |
| 9 | Unzip the submission somewhere new on your machine and follow README from scratch. |

## 6. Cross-platform check (strongly recommended)

Your biggest risk: the project works on your laptop and breaks on the grader's. The kit defends against this in four layers: exact pins with a wheel matrix check, a LightGBM fallback model, a clean-room test, and a GitHub Actions matrix (Windows, macOS, Linux on Python 3.11 to 3.14).

To use the matrix: create a private GitHub repository, push, open the Actions tab, and paste any failing job logs into Claude Code. Keep the repository private, since the repository holds client-provided data.

## 7. Decisions reserved for you

Claude Code will ask about these. Have answers ready:

1. Approval before the single holdout evaluation (Phase 8).
2. Your last and first name for the ZIP (Phase 9).
3. Whether the ZIP includes `.claude/`, `CLAUDE.md`, and `PROGRESS.md` (Phase 9). Including them shows your AI-assisted workflow openly, which the brief permits. Excluding them keeps the submission lean.
4. Any deviation from the reference ranges in docs/01 section 15.

## 8. Handing off to Part 2

Once Phase 9 closes, the dashboard only needs `ledgerclf.service.PredictionService` (docs/02 section 8) and the files listed in `reports/manifest.json`. Every chart for Tab 1 comes from a report file. Tab 2 calls `predict_file` and `explain`. The agent receives the explanation payload JSON and falls back to `fallback_text` when no LLM key exists on the grader's machine.
