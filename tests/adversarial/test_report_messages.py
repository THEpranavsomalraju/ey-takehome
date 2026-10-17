"""File-level issue messages stay readable and honest about what the metrics mean."""

from __future__ import annotations

from ledgerclf.schema import Severity
from tests.adversarial._support import HEADER, ROW_A, csv_text

MESSAGE_LIMIT = 1_000


def test_extra_columns_message_is_capped_like_row_lists(service):
    extras = [f"X{i}" for i in range(5_000)]
    data = csv_text([ROW_A + ["1"] * len(extras)], header=HEADER + extras).encode("utf-8")
    result = service.predict_file(data, "wide.csv")
    issue = next(i for i in result.report.issues if i.code == "EXTRA_COLUMNS_IGNORED")
    assert len(issue.message) < MESSAGE_LIMIT  # today: about 34,000 characters


def test_raw_training_file_metrics_warn_that_rows_were_seen_in_training(service, settings):
    raw = (settings.paths.raw_data).read_bytes()
    result = service.predict_file(raw, "Challenge_Data.xls")
    assert result.metrics is not None and result.metrics["accuracy"] > 0.99
    caveats = [
        i
        for i in result.report.issues
        if i.code != "LABELS_DETECTED"
        and i.severity in (Severity.WARNING, Severity.INFO)
        and "training" in i.message.lower()
    ]
    assert caveats, "in-sample accuracy is shown with no training-overlap caveat"
