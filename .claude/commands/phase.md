---
description: Execute one build phase end to end with the full quality loop
argument-hint: <phase number 0-9>
---

Execute Phase $ARGUMENTS of `docs/03_BUILD_PLAN.md`.

1. Read `CLAUDE.md`, `PROGRESS.md`, the Phase $ARGUMENTS section of `docs/03_BUILD_PLAN.md`, and every document listed under "Read first" for this phase. Read `docs/06_QUALITY_LOOP.md` sections 1, 2, and 5.
2. Confirm the previous phase shows DONE in `PROGRESS.md`. If not, stop and tell me why.
3. Write the Phase $ARGUMENTS plan and copy its acceptance criteria into `PROGRESS.md`. Set the status to IN PROGRESS.
4. Build in small steps. Write tests with the code. Run the relevant tests after each step. Commit at each green fast gate.
5. Run the loop from `docs/06_QUALITY_LOOP.md` section 1: GATE, REVIEW (use /self-review), VERIFY, SCORE. Repeat until the stop criteria in section 5 hold. Respect the 5-cycle cap.
6. Tick every acceptance criterion in `PROGRESS.md` with evidence. Update `DECISIONS.md`. Commit with the phase commit message from the build plan.
7. Stop and report in the format of `docs/06_QUALITY_LOOP.md` section 6. Do not start the next phase.

Stop early and ask me whenever `CLAUDE.md` section 8 applies.
