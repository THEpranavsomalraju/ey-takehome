"""Amount and date cells that parse wrong silently, or fail where a sibling form succeeds."""

from __future__ import annotations

import pytest

from ledgerclf.schema import Severity
from tests.adversarial._support import HEADER, ROW_A, csv_text, run_path, with_value, xlsx_from

AMOUNT, PERIOD = 2, 4


def _csv_column(tmp_path, column: int, values: list[str]):
    rows = [with_value(ROW_A, column, value) for value in values]
    path = tmp_path / "values.csv"
    path.write_text(csv_text(rows), encoding="utf-8")
    return run_path(path, path.name)


def test_invalid_date_does_not_flip_other_rows_to_day_first(tmp_path):
    frame, issues = _csv_column(tmp_path, PERIOD, ["08/09/2018", "45/08/2018"])
    # 45 is no valid day, so it is no evidence of day-first order (docs/05 2.5 Col5).
    flip_reported = any(i.code.startswith("DATE_ORDER") for i in issues)
    assert frame.loc[0, "Col5"] == "2018-08" or flip_reported, (
        "08/09/2018 silently became September because of one typo row"
    )


def test_three_decimal_amount_in_dot_decimal_column_keeps_magnitude(tmp_path):
    frame, issues = _csv_column(tmp_path, AMOUNT, ['"1,234.50"', "617.600"])
    flagged = any(i.severity is Severity.WARNING and 3 in i.row_numbers for i in issues)
    assert frame.loc[1, "amount"] == pytest.approx(617.6) or flagged, (
        f"617.600 read as {frame.loc[1, 'amount']} with no warning in a dot-decimal column"
    )


@pytest.mark.parametrize("text", ["-$500", "-$1,234.50", "-\u20ac99"])
def test_leading_minus_before_currency_symbol_parses(tmp_path, text):
    frame, _ = _csv_column(tmp_path, AMOUNT, [f'"{text}"'])
    expected = -float(text.strip("-$\u20ac").replace(",", ""))
    assert frame.loc[0, "amount"] == pytest.approx(expected)  # "$-500" already parses


@pytest.mark.parametrize("mark", ["\u200b", "\ufeff", "\u200e"], ids=["zwsp", "bom", "lrm"])
def test_invisible_characters_in_amount_and_date_are_ignored(tmp_path, mark):
    row = with_value(with_value(ROW_A, AMOUNT, "1200" + mark), PERIOD, mark + "2018-08-31")
    path = tmp_path / "invisible.csv"
    path.write_text(csv_text([row]), encoding="utf-8")
    frame, _ = run_path(path, path.name)  # docs/05 V04: same result as the clean value
    assert frame.loc[0, "amount"] == 1200.0
    assert frame.loc[0, "Col5"] == "2018-08"


def test_year_dot_month_period_parses_like_year_dash_month(tmp_path):
    frame, _ = _csv_column(tmp_path, PERIOD, ["2018-08", "2018.08"])
    assert frame["Col5"].tolist() == ["2018-08", "2018-08"]


def test_numeric_yyyymmdd_cell_parses_like_text_yyyymmdd(tmp_path):
    def build(book):
        book.active.append(HEADER)
        book.active.append(with_value(with_value(ROW_A, AMOUNT, 1200), PERIOD, "20180831"))
        book.active.append(with_value(with_value(ROW_A, AMOUNT, 1200), PERIOD, 20180831))

    path = tmp_path / "numeric_date.xlsx"
    path.write_bytes(xlsx_from(build))
    frame, _ = run_path(path, path.name)
    assert frame["Col5"].tolist() == ["2018-08", "2018-08"]
