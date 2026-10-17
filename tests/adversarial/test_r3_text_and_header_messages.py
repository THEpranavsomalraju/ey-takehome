"""Phantom rows from decoding, and messages that contradict the file.

- A UTF-16 CSV with one stray trailing byte (a truncated copy or an exporter's
  padding byte) decodes with ``errors="replace"``: the lone byte becomes a
  U+FFFD row that is predicted as data, with no warning. docs/05 section 3:
  one prediction per non-empty input row; section 2.3 records lossy decoding
  with an ENCODING_FALLBACK warning (as the UTF-8 BOM path already does).
- A header row named exactly Col1 to Col7 below 20 blank lines is outside the
  scan window (docs/05 section 2.4), so positional mapping applies. The
  HEADER_NOT_RECOGNIZED warning then claims the names "do not match Col1 to
  Col7" and tells the user to rename them, which is false and does not help.
"""

from __future__ import annotations

from tests.adversarial._support import HEADER, ROW_A, csv_text


def test_utf16_csv_with_stray_trailing_byte_adds_no_silent_row(service):
    data = b"\xff\xfe" + csv_text([ROW_A]).encode("utf-16-le") + b"\x00"
    result = service.predict_file(data, "utf16_padded.csv")
    codes = [issue.code for issue in result.report.issues]
    assert len(result.predictions) == 1 or "ENCODING_FALLBACK" in codes, (
        f"{len(result.predictions)} predictions for 1 data row, issues {codes}"
    )


def test_canonical_header_below_scan_window_is_not_called_mismatched(service):
    data = ("\n" * 20 + csv_text([ROW_A], header=HEADER)).encode("utf-8")
    result = service.predict_file(data, "deep_header.csv")
    skipped = [i for i in result.report.issues if i.code == "HEADER_NOT_RECOGNIZED"]
    for issue in skipped:
        assert "do not match" not in issue.message, issue.message
