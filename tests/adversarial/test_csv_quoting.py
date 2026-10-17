"""CSV cells beyond the csv module field limit, unclosed quotes, and unquoted delimiters."""

from __future__ import annotations

import contextlib

import pytest

from ledgerclf.exceptions import LedgerClfError
from tests.adversarial._support import ROW_A, ROW_B, codes, csv_text, run_path, with_value

CSV_FIELD_LIMIT = 131_072  # default csv.field_size_limit()


def test_text_cell_longer_than_csv_field_limit_is_truncated_not_crashing(tmp_path):
    giant = "x" * (CSV_FIELD_LIMIT + 10)
    path = tmp_path / "giant_cell.csv"
    path.write_text(csv_text([with_value(ROW_A, 3, giant), ROW_B]), encoding="utf-8")
    try:
        frame, issues = run_path(path, path.name)
    except LedgerClfError:
        pytest.fail("a long text cell is a value problem, not a structural error (docs/05 V18)")
    assert len(frame) == 2
    assert "TEXT_TRUNCATED" in codes(issues)


def test_unclosed_quote_in_large_file_never_raises_raw_csv_error(tmp_path):
    rows = [ROW_A, with_value(ROW_A, 3, '"6 inch pipe')] + [ROW_B] * 3_000
    path = tmp_path / "unclosed_quote_large.csv"
    path.write_text(csv_text(rows), encoding="utf-8")
    # A clear LedgerClfError is acceptable; a raw _csv.Error is not (principle 3).
    with contextlib.suppress(LedgerClfError):
        run_path(path, path.name)


def test_unclosed_quote_does_not_silently_swallow_following_rows(tmp_path):
    rows = [ROW_A, with_value(ROW_A, 3, '"6 inch pipe'), ROW_B, ROW_A]
    path = tmp_path / "unclosed_quote.csv"
    path.write_text(csv_text(rows), encoding="utf-8")
    frame, issues = run_path(path, path.name)
    swallowed_warning = any(
        i.severity == "WARNING" and "quote" in i.message.lower() for i in issues
    )
    assert len(frame) == 4 or swallowed_warning, (
        f"4 data rows became {len(frame)} with no quote warning (docs/05 principle 4)"
    )


def test_row_with_extra_unquoted_delimiter_is_not_silently_shifted(tmp_path):
    shifted = with_value(ROW_A, 0, "Acme, Inc")  # 8 fields under a 7-field header
    path = tmp_path / "unquoted_comma.csv"
    path.write_text(csv_text([shifted, ROW_B]), encoding="utf-8")
    frame, issues = run_path(path, path.name)
    first = frame.iloc[0]
    row_named = any(
        i.severity == "WARNING" and 2 in i.row_numbers and i.code != "DATE_UNPARSEABLE"
        for i in issues
    )
    assert first["amount"] != 10719.0 or row_named, (
        "invoice number 10719 was read as the amount with no warning naming row 2"
    )
