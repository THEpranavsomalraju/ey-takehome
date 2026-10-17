"""Header rows that fail to resolve must never become predicted data rows or shift columns."""

from __future__ import annotations

import pytest

from ledgerclf.exceptions import SchemaError
from tests.adversarial._support import HEADER, ROW_A, ROW_B, codes, csv_text, run_path


def test_no_header_file_with_leading_empty_column_maps_inputs_unshifted(tmp_path):
    rows = [["", *ROW_A], ["", *ROW_B]]
    path = tmp_path / "leading_blank_column.csv"
    path.write_text(csv_text(rows, header=None), encoding="utf-8")
    frame, _ = run_path(path, path.name)
    # docs/05 2.4.3: drop columns with an empty header and no values.
    assert frame["Col1"].tolist() == [ROW_A[0], ROW_B[0]]
    assert frame["amount"].tolist() == [1200.0, 75.0]
    assert "ClassificationLabel" not in frame.columns


def test_header_names_with_left_to_right_mark_resolve(tmp_path):
    header = [name + "\u200e" for name in HEADER]  # invisible mark from RTL-locale exports
    path = tmp_path / "lrm_header.csv"
    path.write_text(csv_text([ROW_A, ROW_B], header=header), encoding="utf-8")
    frame, issues = run_path(path, path.name)
    assert "POSITIONAL_MAPPING" not in codes(issues)
    assert len(frame) == 2


@pytest.mark.parametrize(
    "header",
    [
        [f"Column {letter}" for letter in "ABCDEFG"],
        ["Vendor", "Invoice", "Amount", "Description", "Date", "Account", "DocType"],
    ],
    ids=["column_letters", "friendly_names"],
)
def test_unrecognized_header_row_is_not_predicted_silently(tmp_path, header):
    path = tmp_path / "named_header.csv"
    path.write_text(csv_text([ROW_A, ROW_B], header=header), encoding="utf-8")
    frame, issues = run_path(path, path.name)
    header_flagged = any(1 in i.row_numbers and "header" in i.message.lower() for i in issues)
    assert len(frame) == 2 or header_flagged, (
        "the header row became data row 1 with only generic amount/date warnings"
    )


def test_repeated_header_row_mid_file_is_not_predicted_silently(tmp_path):
    path = tmp_path / "concatenated.csv"
    path.write_text(csv_text([ROW_A, HEADER, ROW_B]), encoding="utf-8")
    frame, issues = run_path(path, path.name)
    header_flagged = any(3 in i.row_numbers and "header" in i.message.lower() for i in issues)
    assert len(frame) == 2 or header_flagged


def test_homoglyph_header_error_explains_the_lookalike_character(tmp_path):
    header = ["\u0421ol1", *HEADER[1:]]  # Cyrillic capital Es instead of Latin C
    path = tmp_path / "homoglyph.csv"
    path.write_text(csv_text([ROW_A], header=header), encoding="utf-8")
    with pytest.raises(SchemaError) as caught:
        run_path(path, path.name)
    message = caught.value.message
    # "Missing Col1. Found: Col1, ..." reads as a contradiction to the user.
    assert "\u0421" not in message or "character" in message.lower()
