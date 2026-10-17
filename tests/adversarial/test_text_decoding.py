"""Byte-level decoding: BOM that lies about the encoding, NUL padding."""

from __future__ import annotations

from tests.adversarial._support import HEADER, ROW_A, codes, run_path, with_value


def test_utf8_bom_with_cp1252_body_warns_about_replaced_characters(tmp_path):
    body = ",".join(HEADER) + "\n" + ",".join(with_value(ROW_A, 0, "Caf\u00e9 Word1")) + "\n"
    path = tmp_path / "bom_lies.csv"
    path.write_bytes(b"\xef\xbb\xbf" + body.encode("cp1252"))
    frame, issues = run_path(path, path.name)
    replaced = "\ufffd" in frame.loc[0, "Col1"]
    warned = any(i.severity == "WARNING" and "encod" in i.code.lower() for i in issues)
    assert not replaced or warned, "text silently changed to U+FFFD without a warning"


def test_csv_with_nul_padded_field_reads_with_nul_warning(tmp_path):
    padded = with_value(ROW_A, 0, ROW_A[0] + "\x00\x00\x00\x00")
    path = tmp_path / "nul_padded.csv"
    path.write_bytes((",".join(HEADER) + "\n" + ",".join(padded) + "\n").encode("utf-8"))
    frame, issues = run_path(path, path.name)  # a .csv must not be called "not a supported type"
    assert frame.loc[0, "Col1"] == ROW_A[0]
    assert "NUL_BYTES_REMOVED" in codes(issues)
