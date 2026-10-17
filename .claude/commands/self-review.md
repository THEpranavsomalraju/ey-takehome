---
description: Gate plus parallel reviewer subagents, fix findings, repeat until two clean cycles
---

Run the REVIEW and VERIFY stages of `docs/06_QUALITY_LOOP.md` section 1.

1. Run `python tools/gate.py --full`. Fix every failure first.
2. Delegate in parallel to three subagents. Give each one the current phase, the list of changed files (`git diff --stat` against the last phase commit), and the focus areas:
   - `code-reviewer`: readability, organization, cleanliness, project rules.
   - `leakage-auditor`: holdout isolation, fold discipline, grouping, metric correctness.
   - `edge-case-breaker`: hostile inputs against io, schema, predict, service, explain (from Phase 1 onward).
3. Consolidate every finding into the iteration log and open issues in `PROGRESS.md`, tagged Critical, Warning, or Suggestion with the reporting subagent.
4. Fix every Critical and Warning. Fix low-risk Suggestions which improve readability. Record declined Suggestions with a one-line reason.
5. Rerun the full gate, then rerun all three subagents.
6. Repeat until two consecutive cycles return zero Critical and zero Warning findings. Cap: 5 cycles, then stop and escalate per section 6.
7. Score the rubric (section 3) in `PROGRESS.md`. Any category below 4: fix and repeat.
8. Report: cycles run, findings fixed by priority, rubric scores, anything you need from me.
