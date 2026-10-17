"""A pandas index column (empty header, running 0..n-1) must be ignored, not flagged as ragged rows.

``DataFrame.to_csv()`` and ``DataFrame.to_excel()`` write the index by default
with an empty header cell. Pandas index artifacts are dropped (edge case S14);
otherwise every row would get ROW_HAS_EXTRA_FIELDS, the critical flag
ROW_STRUCTURE_SUSPECT, and review_required=True.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from tests.adversarial._support import HEADER, ROW_A, ROW_B


def _frame() -> pd.DataFrame:
    return pd.DataFrame([ROW_A, ROW_B, ROW_A], columns=HEADER)


def _csv_with_index() -> bytes:
    return _frame().to_csv(index=True).encode("utf-8")


def _xlsx_with_index() -> bytes:
    buffer = io.BytesIO()
    _frame().to_excel(buffer, index=True, engine="openpyxl")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("build", "name"), [(_csv_with_index, "export.csv"), (_xlsx_with_index, "export.xlsx")]
)
def test_pandas_default_index_export_is_not_flagged_as_shifted_rows(service, build, name):
    with_index = service.predict_file(build(), name)
    plain = service.predict_file(_frame().to_csv(index=False).encode("utf-8"), "plain.csv")

    codes = [issue.code for issue in with_index.report.issues]
    assert "ROW_HAS_EXTRA_FIELDS" not in codes
    assert not with_index.predictions["flags"].str.contains("ROW_STRUCTURE_SUSPECT").any()
    assert with_index.predictions["review_required"].tolist() == (
        plain.predictions["review_required"].tolist()
    )
