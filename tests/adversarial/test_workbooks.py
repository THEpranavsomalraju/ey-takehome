"""Workbook layouts: template sheets, hidden sheets, offset data, odd containers."""

from __future__ import annotations

import io
import zipfile
from unittest import mock

import pytest
import xlrd

from ledgerclf.config import get_settings
from ledgerclf.exceptions import FileReadError, LedgerClfError
from ledgerclf.io import read_table
from tests.adversarial._support import HEADER, ROW_A, ROW_B, run_path, xlsx_from


def test_header_only_template_sheet_does_not_hide_data_sheet(tmp_path):
    def build(book):
        book.active.title = "Template"
        book.active.append(HEADER)
        data = book.create_sheet("Data")
        for row in (HEADER, ROW_A, ROW_B):
            data.append(row)

    path = tmp_path / "template_first.xlsx"
    path.write_bytes(xlsx_from(build))
    frame, _ = run_path(path, path.name)  # edge case F11: pick the sheet holding data
    assert len(frame) == 2


def test_hidden_sheet_chosen_over_visible_sheet_is_reported(tmp_path):
    def build(book):
        hidden = book.active
        hidden.title, hidden.sheet_state = "old_backup", "hidden"
        hidden.append(HEADER)
        hidden.append(["OLD", *ROW_A[1:]])
        visible = book.create_sheet("current")
        for row in (HEADER, ROW_A, ROW_B):
            visible.append(row)

    path = tmp_path / "hidden_first.xlsx"
    path.write_bytes(xlsx_from(build))
    frame, issues = run_path(path, path.name)
    hidden_reported = any("hidden" in i.message.lower() for i in issues)
    assert frame["Col1"].tolist() == [ROW_A[0], ROW_B[0]] or hidden_reported


def test_no_header_block_at_c3_error_does_not_echo_blank_cells(tmp_path):
    def build(book):
        for r, row in enumerate((ROW_A, ROW_B), start=3):
            for c, value in enumerate(row, start=3):
                book.active.cell(row=r, column=c, value=value)

    path = tmp_path / "offset_block.xlsx"
    path.write_bytes(xlsx_from(build))
    try:
        frame, _ = run_path(path, path.name)
    except LedgerClfError as error:
        assert "None, None" not in error.message, error.message
    else:
        assert len(frame) == 2  # blank rows and columns are dropped


def test_xlsb_container_is_reported_as_unsupported_format():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/workbook.bin", b"\x00\x01")
    with pytest.raises(FileReadError) as caught:
        read_table(buffer.getvalue(), "ledger.xlsb", get_settings().limits)
    assert caught.value.code == "UNSUPPORTED_FORMAT", caught.value.message


def test_legacy_encrypted_xls_is_reported_as_password_protected():
    ole = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 600
    encrypted = xlrd.biffh.XLRDError("Workbook is encrypted")
    with (
        mock.patch("ledgerclf.io.pd.ExcelFile", side_effect=encrypted),
        pytest.raises(FileReadError) as caught,
    ):
        read_table(ole, "locked.xls", get_settings().limits)
    assert caught.value.code == "PASSWORD_PROTECTED"  # the reader reports encryption


def test_html_table_saved_as_xls_gets_a_clear_message():
    cells = "".join(f"<td>{h}</td>" for h in HEADER)
    html = f"<html><body><table><tr>{cells}</tr></table></body></html>".encode()
    with pytest.raises(LedgerClfError) as caught:
        run_path(html, "erp_export.xls")
    assert "<html>" not in caught.value.message, caught.value.message
