"""The row-level holdout guard must not hinge on Col2, a column the model never reads.

``overlap.row_fingerprints`` joins all five text columns, Col2 (invoice number)
included. Col2 is optional and display-only (edge case V16), so dropping it,
or letting Excel mangle it on a re-save, leaves every prediction unchanged but
changes every fingerprint. The holdout labels must still stay unread: only
evaluate-holdout may read them, and holdout rows are recognized by row content.

The test never touches the real holdout labels: it points a service at a temp
project whose holdout files are synthetic (features file plus labeled copy).
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from ledgerclf.config import HOLDOUT_FEATURES_STEM, Paths, Settings
from ledgerclf.schema import COL_INVOICE, LABEL_COLUMN
from ledgerclf.service import PredictionService
from ledgerclf.splitting import HOLDOUT_WITH_LABELS_FILE
from tests.adversarial._support import HEADER, ROW_A, ROW_B

LABELED = pd.DataFrame(
    [[*ROW_A, "Category_1"], [*ROW_B, "Category_2"]], columns=[*HEADER, LABEL_COLUMN]
)


def _guarded_service(service, tmp_path) -> PredictionService:
    settings = Settings(paths=Paths.from_root(tmp_path))
    settings.paths.holdout.mkdir(parents=True)
    LABELED.to_excel(settings.paths.holdout / HOLDOUT_WITH_LABELS_FILE, index=False)
    features = LABELED.drop(columns=[LABEL_COLUMN])
    features.to_excel(settings.paths.holdout / f"{HOLDOUT_FEATURES_STEM}.xlsx", index=False)
    return PredictionService(service.bundle, service.fallback_reason, settings)


def _csv(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(LABELED.drop(columns=[COL_INVOICE]), id="col2_dropped"),
        pytest.param(LABELED.assign(**{COL_INVOICE: ["1.07E+04", "6.6E+04"]}), id="col2_resaved"),
    ],
)
def test_holdout_rows_with_col2_changed_still_withhold_labels(service, tmp_path, variant):
    guarded = _guarded_service(service, tmp_path)
    result = guarded.predict_file(_csv(variant), "holdout_copy.csv")
    codes = [issue.code for issue in result.report.issues]
    assert result.metrics is None, "holdout labels were scored after only Col2 changed"
    assert "HOLDOUT_METRICS_WITHHELD" in codes
