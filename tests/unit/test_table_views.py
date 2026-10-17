"""Dashboard table views: search, filters, sort, paging, and upload insights."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from ledgerclf.schema import INPUT_COLUMNS
from ledgerclf.table_views import RowQuery, filtered_rows, insights, page


@pytest.fixture
def predictions() -> pd.DataFrame:
    rows = [
        (1, "Word1 Word2", "Word9", "Category_1", 0.99, False, ""),
        (2, "Word3", None, "Category_2", 0.60, True, "LOW_CONFIDENCE"),
        (3, "Word1", "Word8", "Category_1", 0.85, True, "UNSEEN_VENDOR|UNSEEN_ACCOUNT"),
        (4, "Word5", "Word9", "Category_3", 0.85, False, "UNSEEN_VENDOR_ACCOUNT_PAIR"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "row_number",
            "Col1",
            "Col4",
            "predicted_label",
            "confidence",
            "review_required",
            "flags",
        ],
    )
    for column in INPUT_COLUMNS:
        if column not in frame:
            frame[column] = "x"
    return frame


def _rows(table: pd.DataFrame) -> list[int]:
    return table["row_number"].tolist()


def test_search_ignores_missing_cells(predictions):
    assert _rows(filtered_rows(predictions, RowQuery(search="none"))) == []
    assert _rows(filtered_rows(predictions, RowQuery(search="nan"))) == []


def test_search_is_case_insensitive_and_literal(predictions):
    assert _rows(filtered_rows(predictions, RowQuery(search="WORD1"))) == [1, 3]
    assert _rows(filtered_rows(predictions, RowQuery(search="word1 ("))) == []


def test_filters_combine(predictions):
    query = RowQuery(search="word1", review_only=True)
    assert _rows(filtered_rows(predictions, query)) == [3]
    assert _rows(filtered_rows(predictions, RowQuery(label="Category_1"))) == [1, 3]


def test_flag_filter_matches_whole_codes(predictions):
    assert _rows(filtered_rows(predictions, RowQuery(flag="UNSEEN_VENDOR"))) == [3]


@pytest.mark.parametrize(
    ("sort", "expected"),
    [("row", [1, 2, 3, 4]), ("confidence_asc", [2, 3, 4, 1]), ("confidence_desc", [1, 3, 4, 2])],
)
def test_sort_orders_break_ties_by_row(predictions, sort, expected):
    assert _rows(filtered_rows(predictions, RowQuery(sort=sort))) == expected


def test_page_past_the_end_is_empty(predictions):
    result = page(predictions, offset=10, limit=25, total=4)
    assert result["rows"] == [] and result["matched"] == 4 and result["total"] == 4


def test_insights_count_every_row_once(predictions):
    summary = insights(SimpleNamespace(predictions=predictions), threshold=0.8, bins=10)
    histogram = summary["confidence_histogram"]
    assert sum(b["auto"] + b["review"] for b in histogram) == 4
    assert sum(b["review"] for b in histogram) == 2
    assert sum(c["rows"] for c in summary["by_class"]) == 4


def test_flag_counts_are_ordered_by_count_then_code(predictions):
    flags = insights(SimpleNamespace(predictions=predictions), 0.8, 10)["flags"]
    assert [f["flag"] for f in flags] == [
        "LOW_CONFIDENCE",
        "UNSEEN_ACCOUNT",
        "UNSEEN_VENDOR",
        "UNSEEN_VENDOR_ACCOUNT_PAIR",
    ]


def test_upload_without_flags_has_no_flag_counts(predictions):
    clean = predictions.assign(flags="")
    assert insights(SimpleNamespace(predictions=clean), 0.8, 10)["flags"] == []
