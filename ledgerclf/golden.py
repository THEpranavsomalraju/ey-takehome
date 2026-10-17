"""Compare fresh predictions with the golden file written at training time.

Same machine: identical labels, probabilities within 1e-6. Cross-platform:
labels agree on at least 99.5% of rows, probabilities within 1e-4 (BLAS
and compiler differences move the last digits). A model that cannot load here (LightGBM missing) is
reported and skipped, never compared against a different model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ledgerclf.artifacts import CHAMPION_BUNDLE, FALLBACK_BUNDLE, load_bundle
from ledgerclf.config import HOLDOUT_FEATURES_STEM, get_settings
from ledgerclf.exceptions import ArtifactError
from ledgerclf.predict import PROBABILITY_COLUMNS, predict_file
from ledgerclf.training import CHAMPION_NAME, FALLBACK_NAME


@dataclass
class GoldenReport:
    """Per-model comparison lines and the overall verdict."""

    passed: bool = True
    details: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        """Return printable ASCII lines."""
        return [*self.details, f"Golden comparison: {'PASS' if self.passed else 'FAIL'}"]


def compare_with_golden(
    golden_path: Path, min_agreement: float, max_prob_diff: float
) -> GoldenReport:
    """Predict the holdout features with each bundle and compare with the golden file.

    Args:
        golden_path: Path of ``golden_holdout_predictions.csv``.
        min_agreement: Required share of rows with identical labels.
        max_prob_diff: Largest allowed absolute probability difference.

    Returns:
        The report.
    """
    settings = get_settings()
    golden = pd.read_csv(golden_path, encoding="utf-8")
    source = settings.paths.holdout / f"{HOLDOUT_FEATURES_STEM}.xlsx"
    report = GoldenReport()
    for name, file in ((CHAMPION_NAME, CHAMPION_BUNDLE), (FALLBACK_NAME, FALLBACK_BUNDLE)):
        try:
            bundle = load_bundle(file, settings.paths)
        except ArtifactError as error:
            report.details.append(f"SKIP {name}: cannot load here ({error.code})")
            continue
        fresh = predict_file(bundle, source, source.name, settings.limits).predictions
        expected = golden[golden["model"] == name].reset_index(drop=True)
        agreement = float(
            (fresh["predicted_label"].to_numpy() == expected["predicted_label"].to_numpy()).mean()
        )
        diff = float(
            np.abs(
                fresh[list(PROBABILITY_COLUMNS)].to_numpy()
                - expected[list(PROBABILITY_COLUMNS)].to_numpy()
            ).max()
        )
        ok = agreement >= min_agreement and diff <= max_prob_diff
        report.passed &= ok
        report.details.append(
            f"{'PASS' if ok else 'FAIL'} {name}: label agreement {agreement:.4f}, "
            f"max probability difference {diff:.2e}"
        )
    return report
