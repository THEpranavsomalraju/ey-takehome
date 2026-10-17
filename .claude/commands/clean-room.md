---
description: Verify the project installs and runs on a fresh environment in an odd path
---

1. Delegate to the `clean-room-tester` subagent. Tell the subagent the current phase and whether `tools/clean_room.py` and `tools/package_submission.py` exist.
2. Read its report. For every Critical and Warning: fix the root cause in the project (dependencies, paths, encodings, README, packaging, fallback), then rerun the failing step yourself to confirm.
3. Rerun `clean-room-tester` until the report shows zero Critical findings and zero matrix gaps.
4. Save the final report summary to `reports/robustness/clean_room_summary.md` and log the cycle in `PROGRESS.md`.
5. Remind me to push to the private GitHub repository so CI runs the full OS and Python matrix, and ask me to paste back any failing jobs.
