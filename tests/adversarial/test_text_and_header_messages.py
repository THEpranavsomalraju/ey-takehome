"""Phantom rows from decoding, and messages that contradict the file.

- A UTF-16 CSV with one stray trailing byte (a truncated copy or an exporter's
  padding byte) decodes with ``errors="replace"``: the lone byte would become a
  U+FFFD row predicted as data, with no warning. Every non-empty input row gets
  one prediction, so a phantom row must not appear, and lossy decoding must
  raise an ENCODING_FALLBACK warning (as the UTF-8 BOM path does).
- A header row named exactly Col1 to Col7 below 20 blank lines is outside the
  header scan window, so positional mapping applies. The HEADER_NOT_RECOGNIZED
  warning must not claim the names "do not match Col1 to Col7" and tell the
  user to rename them, which would be false and would not help.
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
