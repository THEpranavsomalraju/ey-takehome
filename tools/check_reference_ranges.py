"""Check benchmark results against the docs/01 reference ranges (gate check 9).

Usage:
    python tools/check_reference_ranges.py

A result outside its range is not automatically wrong, since the build runs
on the 90% training portion. Every miss needs an investigation note in
PROGRESS.md (docs/03 Phase 4), so the gate fails until the range is met.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RANGES = ROOT / "tools" / "reference_ranges.json"


def _value(report: str, row: dict[str, object], column: str) -> float:
    """Return one cell of a report, selected by column equality filters."""
    frame = pd.read_csv(ROOT / "reports" / report, encoding="utf-8")
    for key, wanted in row.items():
        frame = frame[frame[key] == wanted]
    if len(frame) != 1:
        raise ValueError(f"{report}: expected one row for {row}, found {len(frame)}")
    return float(frame[column].iloc[0])


def main() -> int:
    """Print one line per range and direction; exit 1 on any miss."""
    spec = json.loads(RANGES.read_text(encoding="utf-8"))
    failures = 0
    for item in spec["ranges"]:
        value = _value(item["report"], item["row"], item["column"])
        ok = abs(value - item["reference"]) <= item["tolerance"]
        failures += not ok
        status = "PASS" if ok else "FAIL"
        reference = f"reference {item['reference']} +- {item['tolerance']}"
        print(f"{status}  {item['name']}: {value:.4f} ({reference})")
    for item in spec["directions"]:
        greater = _value(item["report"], item["greater"], item["column"])
        lesser = _value(item["report"], item["lesser"], item["column"])
        ok = greater > lesser
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {item['name']}: {greater:.4f} > {lesser:.4f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
