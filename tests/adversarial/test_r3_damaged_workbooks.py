"""Damaged workbooks end in FILE_UNREADABLE, never a raw parser exception.

Reading never crashes with a raw exception, and a corrupted .xlsx ends in the
ERROR FILE_UNREADABLE (edge case F04). ``io._EXCEL_READ_ERRORS`` must cover
``xml.etree.ElementTree.ParseError`` (a zip that holds damaged sheet,
workbook, or styles XML) and the errors xlrd raises for a damaged OLE
container (``CompDocError``, ``struct.error``). Otherwise they escape as raw
exceptions, the CLI answers "Internal error" with exit code 3 instead of a
user error, and the API returns a plain-text 500.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from ledgerclf.cli import main
from ledgerclf.config import EXIT_USER_ERROR
from ledgerclf.exceptions import FileReadError
from tests.adversarial._support import HEADER, ROW_A, xlsx_from

OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _workbook() -> bytes:
    def build(book) -> None:
        book.active.append(HEADER)
        book.active.append(ROW_A)

    return xlsx_from(build)


def _with_part(data: bytes, part: str, damage) -> bytes:
    source = zipfile.ZipFile(io.BytesIO(data))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            target.writestr(info.filename, damage(payload) if info.filename == part else payload)
    return buffer.getvalue()


def _half(payload: bytes) -> bytes:
    return payload[: len(payload) // 2]


def _raw_xls_minus_tail(settings) -> bytes:
    return settings.paths.raw_data.read_bytes()[:-10]


CASES = {
    "sheet_xml_truncated": lambda s: (
        _with_part(_workbook(), "xl/worksheets/sheet1.xml", _half),
        "ledger.xlsx",
    ),
    "workbook_xml_truncated": lambda s: (
        _with_part(_workbook(), "xl/workbook.xml", _half),
        "ledger.xlsx",
    ),
    "styles_xml_garbage": lambda s: (
        _with_part(_workbook(), "xl/styles.xml", lambda _: b"garbage"),
        "ledger.xlsx",
    ),
    "ole_header_only": lambda s: (OLE_MAGIC + b"\x00" * 600, "ledger.xls"),
    "xls_missing_last_bytes": lambda s: (_raw_xls_minus_tail(s), "ledger.xls"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_damaged_workbook_raises_file_unreadable(service, settings, case):
    data, name = CASES[case](settings)
    with pytest.raises(FileReadError) as caught:
        service.predict_file(data, name)
    assert caught.value.code == "FILE_UNREADABLE"


def test_cli_reports_damaged_workbook_as_user_error(tmp_path, capsys):
    path = tmp_path / "damaged.xlsx"
    path.write_bytes(_with_part(_workbook(), "xl/worksheets/sheet1.xml", _half))
    code = main(["predict", str(path)])
    output = capsys.readouterr().out
    assert code == EXIT_USER_ERROR
    assert "internal error" not in output.lower()
