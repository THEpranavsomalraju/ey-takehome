"""Edge cases F01 to F16 (file level): reading and validation."""

from __future__ import annotations

import io

import openpyxl
import pandas as pd
import pytest

from ledgerclf.config import Limits
from ledgerclf.exceptions import FileReadError
from ledgerclf.io import read_table
from ledgerclf.schema import INPUT_COLUMNS, LABEL_COLUMN
from tests.conftest import SAMPLE_ROWS, codes, csv_bytes, process, sample_frame, xlsx_bytes

pytestmark = pytest.mark.edge
LIMITS = Limits()


def _error_code(data: bytes, name: str) -> str:
    with pytest.raises(FileReadError) as caught:
        read_table(data, name, LIMITS)
    return caught.value.code


def test_f01_zero_byte_file_is_rejected():
    assert _error_code(b"", "empty.csv") == "FILE_EMPTY"


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("report.pdf", b"%PDF-1.7\n%binary\n"),
        ("data.json", b'{"Col1": "Word1"}'),
        ("letter.docx", None),
    ],
)
def test_f02_unsupported_formats_are_rejected(name, data):
    if data is None:
        buffer = io.BytesIO()
        import zipfile

        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", "<doc/>")
        data = buffer.getvalue()
    with pytest.raises(FileReadError) as caught:
        read_table(data, name, LIMITS)
    assert caught.value.code == "UNSUPPORTED_FORMAT"
    assert ".xlsx" in caught.value.message


def test_f03_csv_named_xlsx_reads_as_csv_with_warning():
    frame, issues, info = process(csv_bytes(SAMPLE_ROWS), "mislabeled.xlsx")
    assert info.detected_format == "csv"
    assert "EXTENSION_MISMATCH" in codes(issues)
    assert len(frame) == len(SAMPLE_ROWS)


def test_f04_truncated_xlsx_is_unreadable():
    data = xlsx_bytes({"data": sample_frame()})
    assert _error_code(data[: len(data) // 2], "broken.xlsx") == "FILE_UNREADABLE"


def test_f05_password_protected_workbook_is_reported():
    ole = (
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512 + "EncryptedPackage".encode("utf-16-le")
    )
    assert _error_code(ole + b"\x00" * 512, "locked.xlsx") == "PASSWORD_PROTECTED"


def test_f06_oversized_file_is_rejected():
    tiny = Limits(max_file_bytes=100)
    with pytest.raises(FileReadError) as caught:
        read_table(csv_bytes(SAMPLE_ROWS * 5), "big.csv", tiny)
    assert caught.value.code == "FILE_TOO_LARGE"


ACCENTED_ROWS = [
    [f"Word1 Caf\u00e9 {i}", "1", 10.0 + i, "Word2", "2018-08-31", "Word3", "NoDoc"]
    for i in range(3)
]


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"])
def test_f07_encodings_produce_identical_frames(encoding):
    reference, _, _ = process(csv_bytes(ACCENTED_ROWS), "ref.csv")
    frame, issues, info = process(csv_bytes(ACCENTED_ROWS, encoding=encoding), "enc.csv")
    pd.testing.assert_frame_equal(frame, reference)
    assert info.encoding is not None
    assert "ENCODING_FALLBACK" not in codes(issues)


def test_f07_undecodable_bytes_fall_back_to_latin1_with_warning():
    data = csv_bytes(SAMPLE_ROWS).replace(b"Word61", b"Word61\x81")
    frame, issues, info = process(data, "odd.csv")
    assert info.encoding == "latin-1"
    assert "ENCODING_FALLBACK" in codes(issues)
    assert len(frame) == len(SAMPLE_ROWS)


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_f08_delimiters_produce_identical_frames(delimiter):
    reference, _, _ = process(csv_bytes(SAMPLE_ROWS), "ref.csv")
    frame, _, info = process(csv_bytes(SAMPLE_ROWS, delimiter=delimiter), "delim.csv")
    assert info.delimiter == delimiter
    pd.testing.assert_frame_equal(frame, reference)


def test_f09_quoted_fields_with_commas_and_newlines_parse():
    rows = [
        ["Word1, Word2", "1", 5.0, "line one\nline two", "2018-08-31", "Word3; Word4", "NoDoc"]
    ] * 3
    frame, _, _ = process(csv_bytes(rows), "quoted.csv")
    assert len(frame) == 3
    assert frame.loc[0, "Col1"] == "Word1, Word2"
    assert frame.loc[0, "Col4"] == "line one line two"


@pytest.mark.parametrize("ending", ["\r\n", "\n", "\r"])
def test_f10_line_endings_produce_identical_frames(ending):
    reference, _, _ = process(csv_bytes(SAMPLE_ROWS), "ref.csv")
    frame, _, _ = process(csv_bytes(SAMPLE_ROWS, line_ending=ending), "endings.csv")
    pd.testing.assert_frame_equal(frame, reference)


def test_f11_data_on_second_sheet_is_selected():
    data = xlsx_bytes({"Cover": pd.DataFrame(), "Ledger": sample_frame()})
    frame, issues, info = process(data, "two_sheets.xlsx")
    assert info.sheet_name == "Ledger"
    assert "SHEET_SELECTED" in codes(issues)
    assert len(frame) == len(SAMPLE_ROWS)


def test_f12_title_rows_above_header_are_skipped():
    rows = [["Ledger export"], ["Generated 2018-12-31"], [], list(INPUT_COLUMNS), *SAMPLE_ROWS]
    frame, _, info = process(csv_bytes(rows, header=None), "titled.csv")
    assert info.header_row_index == 3  # the blank line is kept, so row numbers stay true
    assert len(frame) == len(SAMPLE_ROWS)
    assert frame.loc[0, "Col1"] == SAMPLE_ROWS[0][0]
    assert frame.loc[0, "source_row"] == 5


def test_f13_original_raw_file_reads_with_labels(settings):
    frame, issues, info = process(settings.paths.raw_data, settings.paths.raw_data.name)
    assert info.detected_format == "xls"
    assert len(frame) == 5899
    assert LABEL_COLUMN in frame.columns
    assert not issues


def test_f14_formula_cells_without_cached_values_become_missing(tmp_path):
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(list(INPUT_COLUMNS))
    for row in SAMPLE_ROWS:
        sheet.append(list(row))
    sheet["C2"] = "=A1*2"
    path = tmp_path / "formulas.xlsx"
    book.save(path)
    frame, issues, _ = process(path, path.name)
    assert len(frame) == len(SAMPLE_ROWS)
    assert bool(frame.loc[0, "amount_was_missing"])
    assert "AMOUNT_UNPARSEABLE" in codes(issues)


def test_f15_odd_file_names_read_normally(tmp_path):
    path = tmp_path / "ledger export, M\u00fcller (final) \u2013 v2.csv"
    path.write_bytes(csv_bytes(SAMPLE_ROWS))
    frame, _, info = process(path, path.name)
    assert len(frame) == len(SAMPLE_ROWS)
    assert info.file_name == path.name


def test_f16_xlsm_reads_like_xlsx():
    data = xlsx_bytes({"data": sample_frame()})
    frame, issues, info = process(data, "macro.xlsm")
    assert info.detected_format == "xlsx"
    assert "EXTENSION_MISMATCH" not in codes(issues)
    assert len(frame) == len(SAMPLE_ROWS)


def test_f06_row_limit_counts_data_rows_only():
    small = Limits(max_rows=3)
    process_rows = csv_bytes(SAMPLE_ROWS)
    raw, info, _ = read_table(process_rows, "three.csv", small)
    assert len(raw) == 4 and info.header_row_index == 0
    with pytest.raises(FileReadError) as caught:
        read_table(csv_bytes(SAMPLE_ROWS + SAMPLE_ROWS[:1]), "four.csv", small)
    assert caught.value.code == "TOO_MANY_ROWS"


def test_f01_missing_path_gives_clear_error(tmp_path):
    with pytest.raises(FileReadError) as caught:
        read_table(tmp_path / "nope.csv", "nope.csv", LIMITS)
    assert caught.value.code == "FILE_NOT_FOUND"


def test_f07_utf16_without_bom_reads():
    data = csv_bytes(ACCENTED_ROWS, encoding="utf-16-le")
    frame, _, info = process(data, "utf16le.csv")
    reference, _, _ = process(csv_bytes(ACCENTED_ROWS), "ref.csv")
    assert info.encoding == "utf-16-le"
    pd.testing.assert_frame_equal(frame, reference)


def test_f09_stray_nul_characters_are_removed_with_warning():
    data = csv_bytes(SAMPLE_ROWS).replace(b"Word61", b"Word\x0061")
    frame, issues, _ = process(data, "nul.csv")
    assert "NUL_BYTES_REMOVED" in codes(issues)
    assert frame.loc[0, "Col4"] == "Word61"
