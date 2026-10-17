"""Reading files stays silent and within budget, and extreme values raise no numpy warnings.

- xlrd prints "WARNING *** File is truncated" straight to stdout for a damaged
  .xls, mixing third-party text into CLI output (CLAUDE.md rule 11: library
  code never prints; docs/02 section 11: one ASCII-safe logging handler).
- A one-row workbook whose used range reaches the last sheet row (a stray
  space in A1048576, common after deleting rows in Excel) takes about 9 s,
  far above P01 (590 rows under 2 s; tests allow three times the budget).
- Amount 1e308 still overflows ``abs(amount) * 100`` in AmountFlags and emits
  a RuntimeWarning on every predict and explain; the round 2 fix covered
  1e17 only. V09 requires finite features without noise for huge amounts.
"""

from __future__ import annotations

import subprocess
import sys
import time
import warnings

from tests.adversarial._support import HEADER, ROW_A, csv_text, with_value, xlsx_from

P01_BUDGET_SECONDS = 2.0
TIMEOUT_FACTOR = 3


def test_damaged_xls_prints_only_the_cli_message(settings, tmp_path):
    damaged = tmp_path / "truncated.xls"
    damaged.write_bytes(settings.paths.raw_data.read_bytes()[:65536])
    completed = subprocess.run(
        [sys.executable, "-m", "ledgerclf", "predict", str(damaged)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=settings.paths.root,
        timeout=120,
        check=False,
    )
    console = completed.stdout + completed.stderr
    assert "could not be opened as an Excel workbook" in console
    assert "OLE2" not in console, console
    assert "Trying to access sector" not in console, console


def test_one_row_workbook_with_far_used_range_meets_p01_budget(service):
    def build(book) -> None:
        sheet = book.active
        sheet.append(HEADER)
        sheet.append(ROW_A)
        sheet.cell(row=1_048_576, column=1, value=" ")

    data = xlsx_from(build)
    started = time.perf_counter()
    result = service.predict_file(data, "used_range.xlsx")
    elapsed = time.perf_counter() - started
    assert len(result.predictions) == 1
    assert elapsed < P01_BUDGET_SECONDS * TIMEOUT_FACTOR, f"{elapsed:.1f} s for one data row"


def test_largest_finite_amount_predicts_and_explains_without_runtime_warning(service):
    data = csv_text([with_value(ROW_A, 2, "1e308")]).encode("utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = service.predict_file(data, "max_amount.csv")
        payload = service.explain(result, 1)
    assert "AMOUNT_OUT_OF_RANGE" in result.predictions.loc[0, "flags"]
    assert payload["contributions"]["total"] is not None
