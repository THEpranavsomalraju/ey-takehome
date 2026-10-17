"""Quality gate (docs/06 section 2).

Usage:
    python tools/gate.py          fast gate
    python tools/gate.py --full   full gate (coverage, notebooks, golden, reference ranges)

Prints one ASCII line per check (PASS, FAIL, or SKIP with the reason) and
exits non-zero when any check fails.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "ledgerclf"
EDGE_CATALOG = ROOT / "docs" / "05_ROBUSTNESS_AND_PORTABILITY.md"
EDGE_TESTS = ROOT / "tests" / "edge_cases"
EDGE_FAMILIES_NEEDING_MODEL = ("E", "X", "P")
MODEL_MARKER = ROOT / "artifacts" / "metadata.json"
HYGIENE_MARKERS = re.compile(r"\b(TODO|FIXME|XXX)\b")
HOLDOUT_SENSITIVE = (
    "HOLDOUT_LABELS_FILE",
    "HOLDOUT_WITH_LABELS_FILE",
    "holdout_labels.csv",
    "holdout_with_labels",
    "read_holdout_labels",
    "issue_holdout_token",
    "HoldoutAccessToken",
    "_issuer_key",
    "load_labeled_raw",
    "LabeledFrame",
    "load_full_file_structure",
    ".raw_data",
    "Challenge_Data",
)
# eda.py may call load_full_file_structure for the label-free schema audit and the raw
# label mapping, both sanctioned on the full file (docs/03 Phase 2, DECISIONS.md D26).
# config.py only defines the raw file path.
HOLDOUT_ALLOWED: dict[str, tuple[str, ...]] = {
    "ledgerclf/splitting.py": HOLDOUT_SENSITIVE,
    "ledgerclf/cli.py": HOLDOUT_SENSITIVE,
    "ledgerclf/holdout_eval.py": HOLDOUT_SENSITIVE,
    "ledgerclf/config.py": ("Challenge_Data",),
    "ledgerclf/eda.py": ("load_full_file_structure",),
}
SCANNED_DIRS = ("ledgerclf", "tools", "notebooks", "research", "dashboard")


@dataclass
class Outcome:
    """Result of one gate check."""

    status: str
    detail: str = ""


def _run(command: list[str], timeout: int = 1800) -> Outcome:
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Outcome("FAIL", f"timed out after {timeout}s")
    seconds = time.time() - started
    if completed.returncode == 0:
        return Outcome("PASS", f"{seconds:.0f}s")
    tail = (completed.stdout + completed.stderr).strip().splitlines()[-25:]
    return Outcome("FAIL", f"exit {completed.returncode}\n      " + "\n      ".join(tail))


def _python(*args: str) -> list[str]:
    return [sys.executable, *args]


def check_format() -> Outcome:
    """Check 1: code formatting matches ruff format."""
    return _run(_python("-m", "ruff", "format", "--check", "."))


def check_lint() -> Outcome:
    """Check 2: ruff lint rules pass."""
    return _run(_python("-m", "ruff", "check", "."))


def check_types() -> Outcome:
    """Check 3: mypy passes on the package."""
    return _run(_python("-m", "mypy", "ledgerclf"))


def check_fast_tests() -> Outcome:
    """Check 4: every test except slow ones passes."""
    return _run(_python("-m", "pytest", "-m", "not slow", "-q", "-x"))


def check_full_tests() -> Outcome:
    """Check 5: every test passes with at least 90% line coverage."""
    return _run(_python("-m", "pytest", "-q", "--cov=ledgerclf", "--cov-fail-under=90"))


def check_portability() -> Outcome:
    """Check 6: static and subprocess portability tests pass."""
    return _run(_python("-m", "pytest", "tests/portability", "-q"))


def check_doctor() -> Outcome:
    """Check 7: the doctor command exits 0 (requiring a model once artifacts exist)."""
    args = ["-m", "ledgerclf", "doctor"]
    if MODEL_MARKER.exists():
        args.append("--require-model")
    return _run(_python(*args))


def catalog_ids() -> list[str]:
    """Parse every edge-case ID from the docs/05 section 4 tables."""
    text = EDGE_CATALOG.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"^\| ([FSVLREXP]\d{2}) \|", text, flags=re.MULTILINE)))


def tested_ids() -> set[str]:
    """Collect catalog IDs that prefix a test function name in tests/edge_cases."""
    found: set[str] = set()
    for path in EDGE_TESTS.rglob("test_*.py"):
        found.update(
            m.upper()
            for m in re.findall(r"def test_([a-z]\d{2})_", path.read_text(encoding="utf-8"))
        )
    return found


def check_edge_coverage() -> Outcome:
    """Check 8: every edge-case catalog ID has a test named after it."""
    ids = catalog_ids()
    if not ids:
        return Outcome("FAIL", "no catalog IDs parsed from docs/05")
    required = [i for i in ids if MODEL_MARKER.exists() or i[0] not in EDGE_FAMILIES_NEEDING_MODEL]
    missing = sorted(set(required) - tested_ids())
    if missing:
        return Outcome("FAIL", "untested catalog IDs: " + ", ".join(missing))
    deferred = len(ids) - len(required)
    note = f" ({deferred} model-dependent IDs deferred until artifacts exist)" if deferred else ""
    return Outcome("PASS", f"{len(required)} IDs covered{note}")


def check_reference_ranges() -> Outcome:
    """Check 9: benchmark results sit inside the docs/01 reference ranges."""
    if not (ROOT / "reports" / "benchmark" / "leaderboard.csv").exists():
        return Outcome("SKIP", "benchmark reports not built yet (Phase 4)")
    return _run(_python("tools/check_reference_ranges.py"))


def check_manifest() -> Outcome:
    """Check 10: every listed report exists and every report file is listed."""
    if not (ROOT / "reports" / "manifest.json").exists():
        return Outcome("SKIP", "reports/manifest.json not built yet (Phase 2)")
    return _run(_python("-m", "pytest", "tests/integration/test_manifest.py", "-q"))


def check_notebooks() -> Outcome:
    """Check 11: every notebook executes top to bottom without errors."""
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not notebooks:
        return Outcome("SKIP", "no notebooks yet (Phase 2)")
    return _run(_python("tools/run_notebooks.py"), timeout=3600)


def check_golden() -> Outcome:
    """Check 12: predictions match the golden file on this machine."""
    if not (ROOT / "artifacts" / "golden_holdout_predictions.csv").exists():
        return Outcome("SKIP", "golden file not built yet (Phase 6)")
    return _run(_python("tools/compare_golden.py", "--tolerance", "same-machine"))


def _python_files() -> list[Path]:
    files: list[Path] = []
    for name in SCANNED_DIRS:
        files.extend(sorted((ROOT / name).rglob("*.py")))
    return files


def _commented_code_blocks(path: Path) -> list[int]:
    """Return line numbers starting 3+ consecutive comment lines that parse as code."""
    lines = path.read_text(encoding="utf-8").splitlines()
    starts, block, block_start = [], [], 0
    for number, line in enumerate([*lines, ""], start=1):
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            if not block:
                block_start = number
            block.append(stripped.lstrip("#").strip())
            continue
        if len(block) >= 3 and _looks_like_code(block):
            starts.append(block_start)
        block = []
    return starts


def _looks_like_code(block: list[str]) -> bool:
    text = "\n".join(block)
    if not re.search(r"[=()\[\]:]", text):
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(
        not isinstance(node, ast.Expr) or isinstance(node.value, ast.Call) for node in tree.body
    )


def check_hygiene() -> Outcome:
    """Check 13: no stray prints, markers, commented-out code, or unused modules."""
    problems: list[str] = []
    for path in _python_files():
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        if (
            relative.startswith("ledgerclf/")
            and relative != "ledgerclf/cli.py"
            and "print(" in text
        ):
            problems.append(f"{relative}: print( outside cli.py")
        if HYGIENE_MARKERS.search(text) and relative != "tools/gate.py":
            problems.append(f"{relative}: TODO/FIXME/XXX marker")
        for line in _commented_code_blocks(path):
            problems.append(f"{relative}:{line}: commented-out code")
    problems.extend(_unused_modules())
    if problems:
        return Outcome("FAIL", "\n      ".join(problems))
    return Outcome("PASS", f"{len(_python_files())} files scanned")


def _unused_modules() -> list[str]:
    sources = "\n".join(
        p.read_text(encoding="utf-8") for p in [*_python_files(), *(ROOT / "tests").rglob("*.py")]
    )
    unused = []
    for module in sorted(PACKAGE.glob("*.py")):
        name = module.stem
        if name in ("__init__", "__main__"):
            continue
        if not re.search(
            rf"ledgerclf(\.|\s+import\s+){name}\b|from ledgerclf import .*\b{name}\b", sources
        ):
            unused.append(f"ledgerclf/{module.name}: never imported")
    return unused


def check_holdout_guard() -> Outcome:
    """Check 14: holdout label files are referenced only by guarded code."""
    offenders = []
    for path in [*_python_files(), *sorted((ROOT / "notebooks").glob("*.ipynb"))]:
        relative = path.relative_to(ROOT).as_posix()
        if relative == "tools/gate.py":
            continue
        text = path.read_text(encoding="utf-8")
        allowed = HOLDOUT_ALLOWED.get(relative, ())
        hits = [token for token in HOLDOUT_SENSITIVE if token in text and token not in allowed]
        if hits:
            offenders.append(f"{relative}: {', '.join(hits)}")
    if offenders:
        return Outcome(
            "FAIL", "holdout label access outside the guard:\n      " + "\n      ".join(offenders)
        )
    return Outcome("PASS", "holdout labels referenced only by guarded code and tests")


Check = tuple[int, str, Callable[[], Outcome]]
FAST_CHECKS: list[Check] = [
    (1, "formatting", check_format),
    (2, "lint", check_lint),
    (3, "types", check_types),
    (4, "tests (not slow)", check_fast_tests),
    (6, "portability", check_portability),
    (7, "doctor", check_doctor),
    (8, "edge-case coverage", check_edge_coverage),
    (10, "manifest integrity", check_manifest),
    (13, "hygiene", check_hygiene),
    (14, "holdout guard", check_holdout_guard),
]
FULL_ONLY: list[Check] = [
    (5, "all tests + coverage", check_full_tests),
    (9, "reference ranges", check_reference_ranges),
    (11, "notebooks", check_notebooks),
    (12, "golden predictions", check_golden),
]


def main() -> int:
    """Run the selected gate checks and return the process exit code."""
    parser = argparse.ArgumentParser(description="Run the quality gate.")
    parser.add_argument("--full", action="store_true", help="run the full gate")
    args = parser.parse_args()
    checks = list(FAST_CHECKS)
    if args.full:
        checks = sorted([c for c in checks if c[0] != 4] + FULL_ONLY, key=lambda c: c[0])
    started = time.time()
    failures = 0
    for number, name, check in checks:
        outcome = check()
        failures += outcome.status == "FAIL"
        print(f"{outcome.status:<4}  {number} {name}  {outcome.detail}", flush=True)
    mode = "full" if args.full else "fast"
    verdict = "PASS" if failures == 0 else f"FAIL ({failures} failing)"
    print(f"\nGate ({mode}): {verdict} in {time.time() - started:.0f}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
