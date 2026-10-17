---
description: Run the full quality gate and fix every failure until green
---

1. Run `python tools/gate.py --full`. If `tools/gate.py` does not exist yet, build the tool first per `docs/06_QUALITY_LOOP.md` section 2.
2. For each FAIL: find the root cause, fix the code (never weaken a test or threshold), and rerun the failing check alone to confirm.
3. Rerun `python tools/gate.py --full` until every check shows PASS or a documented SKIP.
4. Append a row to the iteration log in `PROGRESS.md`: date, gate result, duration, what you fixed.
5. Report the final gate summary to me in under 15 lines.
