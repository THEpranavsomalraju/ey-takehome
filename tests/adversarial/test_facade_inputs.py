"""PredictionService.predict_file never lets a raw Python exception escape (docs/05 section 1.3).

A Streamlit ``UploadedFile`` is an ``io.BytesIO`` subclass, so passing it straight
to the facade is the most likely Part 2 integration mistake. Today these inputs
raise TypeError or ValueError instead of a LedgerClfError with a fix.
"""

from __future__ import annotations

import io

import pytest

from ledgerclf.exceptions import LedgerClfError
from tests.adversarial._support import ROW_A, ROW_B, csv_text

CSV = csv_text([ROW_A, ROW_B]).encode("utf-8")


def _predict_or_ledger_error(service, data, file_name):
    try:
        return service.predict_file(data, file_name)
    except LedgerClfError:
        return None


@pytest.mark.parametrize(
    ("data", "file_name"),
    [
        (io.BytesIO(CSV), "upload.csv"),
        (None, "upload.csv"),
        (CSV, None),
        ("Col1\x00Col2", "upload.csv"),
    ],
    ids=["bytesio_like_streamlit_upload", "none_data", "none_file_name", "str_with_nul"],
)
def test_facade_turns_bad_source_types_into_ledgerclf_errors(service, data, file_name):
    result = _predict_or_ledger_error(service, data, file_name)
    if result is not None:
        assert len(result.predictions) == 2
