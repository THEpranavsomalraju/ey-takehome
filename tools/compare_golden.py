"""Compare fresh predictions with artifacts/golden_holdout_predictions.csv.

Usage:
    python tools/compare_golden.py --tolerance same-machine|cross-platform

Same machine: identical labels, probabilities within 1e-6.
Cross-platform: labels agree on at least 99.5% of rows, probabilities within 1e-4.
Prints SKIP and exits 0 while the golden file does not exist yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOLERANCES = {"same-machine": (1.0, 1e-6), "cross-platform": (0.995, 1e-4)}


def main() -> int:
    """Predict the holdout features with every bundle and compare with the golden file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tolerance", choices=sorted(TOLERANCES), default="same-machine")
    args = parser.parse_args()
    golden_path = ROOT / "artifacts" / "golden_holdout_predictions.csv"
    if not golden_path.exists():
        print("SKIP: the golden file does not exist yet. Run: python -m ledgerclf train")
        return 0

    from ledgerclf.golden import compare_with_golden

    min_agreement, max_prob_diff = TOLERANCES[args.tolerance]
    report = compare_with_golden(golden_path, min_agreement, max_prob_diff)
    for line in report.lines():
        print(line)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
