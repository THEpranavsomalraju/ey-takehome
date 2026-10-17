"""The labeled-holdout guard must not depend on the exact file bytes.

Holdout labels are read only by evaluate-holdout. A guard that withheld metrics
only when the upload's SHA-256 equals data/holdout/holdout_with_labels.xlsx would
fail on a CSV re-save (or an open-and-save in Excel), which changes the bytes and
would let the holdout labels be scored.

The test never touches the real holdout labels: it points a service at a temp
project whose "holdout_with_labels.xlsx" is a synthetic labeled file.
"""

from __future__ import annotations

import io

import pandas as pd

from ledgerclf.config import HOLDOUT_FEATURES_STEM, Paths, Settings
from ledgerclf.service import PredictionService
from ledgerclf.splitting import HOLDOUT_WITH_LABELS_FILE
from tests.adversarial._support import HEADER, ROW_A, ROW_B

LABELED = pd.DataFrame(
    [[*ROW_A, "Category_1"], [*ROW_B, "Category_2"]], columns=[*HEADER, "ClassificationLabel"]
)


def _guarded_service(service, tmp_path, with_features: bool = True) -> PredictionService:
    settings = Settings(paths=Paths.from_root(tmp_path))
    settings.paths.holdout.mkdir(parents=True)
    LABELED.to_excel(settings.paths.holdout / HOLDOUT_WITH_LABELS_FILE, index=False)
    if with_features:
        features = LABELED.drop(columns=["ClassificationLabel"])
        features.to_excel(settings.paths.holdout / f"{HOLDOUT_FEATURES_STEM}.xlsx", index=False)
    return PredictionService(service.bundle, service.fallback_reason, settings)


def test_holdout_rows_resaved_as_csv_still_withhold_metrics(service, tmp_path):
    guarded = _guarded_service(service, tmp_path)
    exact = (guarded.settings.paths.holdout / HOLDOUT_WITH_LABELS_FILE).read_bytes()
    assert guarded.predict_file(exact, "holdout_with_labels.xlsx").metrics is None

    buffer = io.StringIO()
    LABELED.to_csv(buffer, index=False)
    resaved = guarded.predict_file(buffer.getvalue().encode("utf-8"), "holdout_with_labels.csv")
    assert resaved.metrics is None


def test_missing_holdout_features_file_keeps_every_label_unread(service, tmp_path):
    guarded = _guarded_service(service, tmp_path, with_features=False)
    buffer = io.StringIO()
    LABELED.to_csv(buffer, index=False)
    result = guarded.predict_file(buffer.getvalue().encode("utf-8"), "any_labeled.csv")
    assert result.metrics is None
    assert "HOLDOUT_METRICS_WITHHELD" in [i.code for i in result.report.issues]
