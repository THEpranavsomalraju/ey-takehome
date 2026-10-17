"""Edge cases V01 to V20 (docs/05 section 4, value level): coercion.

Prediction-level assertions (flags, review policy) live in
``test_prediction_level.py`` and run once trained artifacts exist.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ledgerclf.schema import TEXT_TRUNCATED
from tests.conftest import SAMPLE_ROWS, codes, csv_bytes, process, sample_frame, xlsx_bytes

pytestmark = pytest.mark.edge


def _one_row(**overrides) -> list[object]:
    row = list(SAMPLE_ROWS[0])
    columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7"]
    for name, value in overrides.items():
        row[columns.index(name)] = value
    return row


def _xlsx_frame(rows: list[list[object]]):
    frame = pd.DataFrame(rows, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7"])
    return process(xlsx_bytes({"data": frame}), "values.xlsx")


def test_v01_numbers_in_text_columns_become_strings():
    out, _, _ = _xlsx_frame([_one_row(Col1=12345, Col4=3.5, Col6=7.0)])
    assert out.loc[0, "Col1"] == "12345"
    assert out.loc[0, "Col4"] == "3.5"
    assert out.loc[0, "Col6"] == "7"


def test_v02_missing_vendor_or_account_stays_missing():
    out, _, _ = _xlsx_frame([_one_row(Col1=None), _one_row(Col6="")])
    assert out.loc[0, "Col1"] is None
    assert out.loc[1, "Col6"] is None


def test_v03_missing_description_stays_missing_without_error():
    out, issues, _ = _xlsx_frame([_one_row(Col4=None)])
    assert out.loc[0, "Col4"] is None
    assert not [i for i in issues if i.severity == "ERROR"]


def test_v04_invisible_whitespace_matches_clean_value():
    messy = _one_row(Col1="  Word16\u00a0Word17\tWord18 \u200bWord7  ")
    out, _, _ = _xlsx_frame([messy, _one_row()])
    assert out.loc[0, "Col1"] == out.loc[1, "Col1"]


def test_v05_lowercase_tokens_are_kept_for_model_normalization():
    out, _, _ = _xlsx_frame([_one_row(Col6="word563 word564 word9 word26")])
    assert out.loc[0, "Col6"] == "word563 word564 word9 word26"


def test_v06_unseen_values_pass_through():
    out, issues, _ = _xlsx_frame([_one_row(Col1="Word99999 Word88888", Col6="Word77777")])
    assert out.loc[0, "Col1"] == "Word99999 Word88888"
    assert not issues


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1,234.50", 1234.5),
        ("1234.50 USD", 1234.5),
        ("(500.00)", -500.0),
        ("500-", -500.0),
        ("1.5e3", 1500.0),
        (" 42 ", 42.0),
        ("1.234,56", 1234.56),
    ],
)
def test_v07_amount_strings_parse(text, expected):
    out, _, _ = process(csv_bytes([_one_row(Col3=text)]), "amounts.csv")
    assert out.loc[0, "amount"] == pytest.approx(expected)
    assert not out.loc[0, "amount_was_missing"]


@pytest.mark.parametrize("text", ["abc", "", "inf"])
def test_v08_bad_amounts_are_missing_and_reported(text):
    out, issues, _ = process(csv_bytes([_one_row(Col3=text)]), "bad_amount.csv")
    assert np.isnan(out.loc[0, "amount"])
    assert out.loc[0, "amount_was_missing"]
    assert "AMOUNT_UNPARSEABLE" in codes(issues)


@pytest.mark.parametrize("value", [1e15, -1e15])
def test_v09_extreme_amounts_stay_finite(value):
    out, _, _ = _xlsx_frame([_one_row(Col3=value)])
    assert out.loc[0, "amount"] == value
    assert np.isfinite(out.loc[0, "amount"])


@pytest.mark.parametrize("value", [0, -790.66])
def test_v10_zero_and_negative_amounts_are_valid(value):
    out, issues, _ = _xlsx_frame([_one_row(Col3=value)])
    assert out.loc[0, "amount"] == value
    assert "AMOUNT_UNPARSEABLE" not in codes(issues)


@pytest.mark.parametrize(
    "value",
    [
        pd.Timestamp("2018-08-31"),
        "2018-08-31",
        "08/31/2018",
        "31/08/2018",
        "Aug 31, 2018",
        "2018-08-31 13:45:00",
        43343,
    ],
)
def test_v11_date_formats_map_to_august_2018(value):
    out, _, _ = _xlsx_frame([_one_row(Col5=value)])
    assert out.loc[0, "Col5"] == "2018-08"
    assert out.loc[0, "period"] == 2018 * 12 + 8


def test_v11_timezone_aware_dates_map_to_period():
    frame = sample_frame().astype({"Col5": object})
    frame["Col5"] = [pd.Timestamp("2018-08-31 10:00", tz="UTC")] * len(frame)
    out, _, _ = process(csv_bytes(frame.to_numpy().tolist()), "tz.csv")
    assert (out["Col5"] == "2018-08").all()


def test_v12_ambiguous_dates_read_month_first_with_info():
    out, issues, _ = process(csv_bytes([_one_row(Col5="03/04/2018")]), "ambiguous.csv")
    assert out.loc[0, "Col5"] == "2018-03"
    assert "DATE_ORDER_ASSUMED_MONTH_FIRST" in codes(issues)


def test_v12_any_day_above_twelve_switches_column_to_day_first():
    rows = [_one_row(Col5="03/04/2018"), _one_row(Col5="31/08/2018")]
    out, issues, _ = process(csv_bytes(rows), "dayfirst.csv")
    assert out["Col5"].tolist() == ["2018-04", "2018-08"]
    assert "DATE_ORDER_ASSUMED_MONTH_FIRST" not in codes(issues)


@pytest.mark.parametrize("value", [None, "not a date"])
def test_v13_bad_dates_are_missing_and_reported(value):
    out, issues, _ = process(csv_bytes([_one_row(Col5=value)]), "bad_date.csv")
    assert np.isnan(out.loc[0, "period"])
    assert out.loc[0, "period_was_missing"]
    assert "DATE_UNPARSEABLE" in codes(issues)


@pytest.mark.parametrize(
    ("value", "label"),
    [("2015-06-30", "2015-06"), ("2019-01-31", "2019-01"), ("2030-12-31", "2030-12")],
)
def test_v14_out_of_range_dates_parse(value, label):
    out, _, _ = process(csv_bytes([_one_row(Col5=value)]), "range.csv")
    assert out.loc[0, "Col5"] == label


@pytest.mark.parametrize(
    ("value", "expected"),
    [("doc1", "Doc1"), (" NoDoc ", "NoDoc"), ("NODOC", "NoDoc"), ("Doc4", "Doc4")],
)
def test_v15_document_types_are_canonicalized(value, expected):
    out, _, _ = process(csv_bytes([_one_row(Col7=value)]), "doc.csv")
    assert out.loc[0, "Col7"] == expected


@pytest.mark.parametrize(
    ("value", "expected"), [("4.80P+11", "4.80P+11"), (10719, "10719"), (108542.1, "108542.1")]
)
def test_v16_invoice_numbers_are_display_strings(value, expected):
    out, _, _ = _xlsx_frame([_one_row(Col2=value)])
    assert out.loc[0, "Col2"] == expected


def test_v17_mixed_types_in_one_column_coerce_per_rule():
    rows = [_one_row(Col3=12.5), _one_row(Col3="$7"), _one_row(Col3="x"), _one_row(Col3=True)]
    out, _, _ = _xlsx_frame(rows)
    assert out["amount"].iloc[:2].tolist() == [12.5, 7.0]
    assert out["amount_was_missing"].tolist() == [False, False, True, True]


def test_v18_long_text_is_truncated_and_marked(settings):
    out, issues, _ = process(csv_bytes([_one_row(Col4="x" * 10_000)]), "long.csv")
    assert len(out.loc[0, "Col4"]) == settings.limits.max_text_chars
    assert out.loc[0, TEXT_TRUNCATED]
    assert "TEXT_TRUNCATED" in codes(issues)


def test_v19_non_ascii_and_emoji_text_is_json_safe():
    out, _, _ = process(
        csv_bytes([_one_row(Col4="Caf\u00e9 \U0001f600 \u4e2d\u6587")]), "emoji.csv"
    )
    assert out.loc[0, "Col4"] == "Caf\u00e9 \U0001f600 \u4e2d\u6587"
    json.dumps(out.loc[0, "Col4"])


def test_v20_boolean_cells_in_text_columns_become_strings():
    out, _, _ = _xlsx_frame([_one_row(Col1=True, Col4=False)])
    assert out.loc[0, "Col1"] == "True"
    assert out.loc[0, "Col4"] == "False"
