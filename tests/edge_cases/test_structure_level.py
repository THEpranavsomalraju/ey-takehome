"""Edge cases S01 to S14 (structure level): column resolution."""

from __future__ import annotations

import pandas as pd
import pytest

from ledgerclf.exceptions import SchemaError
from ledgerclf.schema import INPUT_COLUMNS, LABEL_COLUMN
from tests.conftest import HEADER, SAMPLE_ROWS, codes, csv_bytes, process, sample_frame, xlsx_bytes

pytestmark = pytest.mark.edge


def _schema_error(data: bytes, name: str = "file.csv") -> SchemaError:
    with pytest.raises(SchemaError) as caught:
        process(data, name)
    return caught.value


def test_s01_header_only_has_no_data_rows():
    assert _schema_error(csv_bytes([])).code == "NO_DATA_ROWS"


def test_s02_missing_required_columns_are_named():
    frame = sample_frame().drop(columns=["Col4", "Col6"])
    error = _schema_error(xlsx_bytes({"data": frame}), "missing.xlsx")
    assert error.code == "MISSING_REQUIRED_COLUMNS"
    assert "Col4, Col6" in error.message
    assert "Found columns: Col1, Col2, Col3, Col5, Col7" in error.message


def test_s03_extra_columns_are_ignored_with_warning():
    frame = sample_frame().assign(Notes="check", row_id=range(3))
    out, issues, _ = process(xlsx_bytes({"data": frame}), "extra.xlsx")
    assert "EXTRA_COLUMNS_IGNORED" in codes(issues)
    message = next(i.message for i in issues if i.code == "EXTRA_COLUMNS_IGNORED")
    assert "Notes" in message and "row_id" in message
    assert "Notes" not in out.columns


def test_s04_shuffled_columns_match_canonical_order():
    reference, _, _ = process(xlsx_bytes({"data": sample_frame()}), "ref.xlsx")
    shuffled = sample_frame()[["Col7", "Col3", "Col1", "Col6", "Col2", "Col5", "Col4"]]
    out, _, _ = process(xlsx_bytes({"data": shuffled}), "shuffled.xlsx")
    pd.testing.assert_frame_equal(out, reference)


@pytest.mark.parametrize("style", ["col{}", "COL{}", " Col {} ", "col_{}", "Col-{}"])
def test_s05_header_variants_resolve(style):
    header = [style.format(i) for i in range(1, 8)]
    out, issues, _ = process(csv_bytes(SAMPLE_ROWS, header=header), "variants.csv")
    assert list(out.columns[2:9]) == list(INPUT_COLUMNS)
    assert "POSITIONAL_MAPPING" not in codes(issues)


def test_s06_no_header_seven_columns_maps_by_position():
    out, issues, _ = process(csv_bytes(SAMPLE_ROWS, header=None), "noheader.csv")
    assert "POSITIONAL_MAPPING" in codes(issues)
    assert len(out) == len(SAMPLE_ROWS)
    assert out.loc[0, "Col1"] == SAMPLE_ROWS[0][0]


def test_s07_no_header_eight_columns_reads_labels():
    rows = [[*row, "Category 3"] for row in SAMPLE_ROWS]
    out, issues, _ = process(csv_bytes(rows, header=None), "noheader8.csv")
    assert "POSITIONAL_MAPPING" in codes(issues)
    assert out[LABEL_COLUMN].tolist() == ["Category 3"] * len(SAMPLE_ROWS)


def test_s08_duplicate_headers_keep_first_with_warning():
    header = [*HEADER, "Col3"]
    rows = [[*row, 999999] for row in SAMPLE_ROWS]
    out, issues, _ = process(csv_bytes(rows, header=header), "dupes.csv")
    assert "DUPLICATE_COLUMN" in codes(issues)
    assert out.loc[0, "amount"] == pytest.approx(SAMPLE_ROWS[0][2])


@pytest.mark.parametrize("label_header", ["ClassificationLabel", "label", "Class", "TARGET", "H"])
def test_s09_label_column_variants_are_detected(label_header):
    rows = [[*row, "Category_2"] for row in SAMPLE_ROWS]
    out, _, _ = process(csv_bytes(rows, header=[*HEADER, label_header]), "labels.csv")
    assert out[LABEL_COLUMN].tolist() == ["Category_2"] * len(SAMPLE_ROWS)


def test_s10_missing_invoice_column_warns_only():
    out, issues, _ = process(
        xlsx_bytes({"data": sample_frame().drop(columns=["Col2"])}), "nocol2.xlsx"
    )
    assert "OPTIONAL_COLUMN_MISSING" in codes(issues)
    assert out["Col2"].isna().all()
    assert len(out) == len(SAMPLE_ROWS)


def test_s11_empty_rows_are_dropped_and_source_rows_kept():
    rows = [
        SAMPLE_ROWS[0],
        [None] * 7,
        SAMPLE_ROWS[1],
        ["", " ", "", "", "", "", ""],
        SAMPLE_ROWS[2],
    ]
    out, issues, _ = process(csv_bytes(rows), "gaps.csv")
    assert len(out) == 3
    assert out["source_row"].tolist() == [2, 4, 6]
    assert out["row_number"].tolist() == [1, 2, 3]
    dropped = next(i for i in issues if i.code == "EMPTY_ROWS_DROPPED")
    assert dropped.row_numbers == (3, 5)


def test_s12_single_data_row():
    out, _, _ = process(csv_bytes(SAMPLE_ROWS[:1]), "one.csv")
    assert len(out) == 1


def test_s13_letter_headers_resolve():
    out, issues, _ = process(csv_bytes(SAMPLE_ROWS, header=list("ABCDEFG")), "letters.csv")
    assert list(out.columns[2:9]) == list(INPUT_COLUMNS)
    assert "POSITIONAL_MAPPING" not in codes(issues)


def test_s14_pandas_index_column_is_ignored():
    data = sample_frame().to_csv(index=True).encode("utf-8")
    out, issues, _ = process(data, "indexed.csv")
    assert "EXTRA_COLUMNS_IGNORED" not in codes(issues)
    assert len(out) == len(SAMPLE_ROWS)
    named = sample_frame().reset_index().rename(columns={"index": "Unnamed: 0"})
    out_named, issues_named, _ = process(xlsx_bytes({"data": named}), "unnamed.xlsx")
    assert "EXTRA_COLUMNS_IGNORED" not in codes(issues_named)
    assert len(out_named) == len(SAMPLE_ROWS)


def test_s11_blank_csv_lines_keep_true_source_rows():
    text = csv_bytes(SAMPLE_ROWS).decode("utf-8").splitlines()
    data = "\n".join([text[0], text[1], "", text[2], ",,,,,,", text[3]]).encode("utf-8") + b"\n\n"
    out, issues, _ = process(data, "blank_lines.csv")
    assert out["source_row"].tolist() == [2, 4, 6]
    dropped = next(i for i in issues if i.code == "EMPTY_ROWS_DROPPED")
    assert dropped.row_numbers == (3, 5)


def test_s11_blank_xlsx_rows_keep_true_source_rows():
    import io as std_io

    import openpyxl

    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(HEADER)
    sheet.append(SAMPLE_ROWS[0])
    sheet.append([])
    sheet.append(SAMPLE_ROWS[1])
    buffer = std_io.BytesIO()
    book.save(buffer)
    out, _, _ = process(buffer.getvalue(), "blank_rows.xlsx")
    assert out["source_row"].tolist() == [2, 4]
