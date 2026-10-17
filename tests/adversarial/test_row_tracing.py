"""source_row must match the spreadsheet row; empty rows never count against limits."""

from __future__ import annotations

from tests.adversarial._support import HEADER, ROW_A, ROW_B, run_path


def test_blank_line_between_rows_keeps_spreadsheet_row_numbers(tmp_path):
    text = ",".join(HEADER) + "\n" + ",".join(ROW_A) + "\n\n" + ",".join(ROW_B) + "\n"
    path = tmp_path / "blank_line.csv"
    path.write_text(text, encoding="utf-8")
    frame, _ = run_path(path, path.name)
    # Header is spreadsheet row 1, ROW_A row 2, the blank line row 3, ROW_B row 4.
    assert frame["source_row"].tolist() == [2, 4]


def test_blank_lines_above_header_keep_spreadsheet_row_numbers(tmp_path):
    text = "\n\n" + ",".join(HEADER) + "\n" + ",".join(ROW_A) + "\n"
    path = tmp_path / "blank_top.csv"
    path.write_text(text, encoding="utf-8")
    frame, _ = run_path(path, path.name)
    assert frame["source_row"].tolist() == [4]


def test_trailing_comma_only_rows_do_not_trigger_row_limit(tmp_path):
    data_rows = "\n".join(",".join(ROW_A) for _ in range(590))
    text = ",".join(HEADER) + "\n" + data_rows + "\n" + ",,,,,,\n" * 250_000
    path = tmp_path / "formatted_tail.csv"
    path.write_text(text, encoding="utf-8")
    frame, _ = run_path(path, path.name)  # the limit is 200,000 *data* rows
    assert len(frame) == 590
