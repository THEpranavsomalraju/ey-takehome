from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from ledgerclf.values import (
    clean_text,
    is_ambiguous_numeric_date,
    parse_amount,
    parse_date,
    period_label,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  Word563\u00a0 Word564\t", "Word563 Word564"),
        ("Word\u200b563", "Word563"),
        (10719.0, "10719"),
        (108542.1, "108542.1"),
        (True, "True"),
        ("N/A", None),
        ("-", None),
        ("nan", None),
        (None, None),
        (np.nan, None),
        ("\uff37ord1", "Word1"),
    ],
)
def test_clean_text_normalizes_values(value, expected):
    assert clean_text(value, max_chars=1000)[0] == expected


def test_clean_text_truncates_long_text():
    text, truncated = clean_text("x" * 50, max_chars=10)
    assert text == "x" * 10
    assert truncated


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1234.5, 1234.5),
        (42, 42.0),
        ("$1,234.50", 1234.5),
        ("1234.50 USD", 1234.5),
        ("(500.00)", -500.0),
        ("500-", -500.0),
        ("1.5e3", 1500.0),
        (" 42 ", 42.0),
        ("1.234,56", 1234.56),
        ("\u20ac 99", 99.0),
        ("1'234.50", 1234.5),
        ("1 234.50", 1234.5),
        ("\u2212250", -250.0),
        ("-0", 0.0),
        ("-$1,234.50", -1234.5),
        ("-1234.50 USD", -1234.5),
        ("$-5", -5.0),
        ("+5", 5.0),
        ("1.234", 1.234),
        ("12.500", 12.5),
        ("1.234.567", 1234567.0),
    ],
)
def test_parse_amount_accepts_common_formats(value, expected):
    assert parse_amount(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [
        "abc",
        "",
        "inf",
        "1e309",
        "--5",
        "5e",
        "$",
        "(-5)",
        "1,2,3",
        True,
        None,
        np.nan,
        math.inf,
        [1],
    ],
)
def test_parse_amount_rejects_malformed(value):
    assert parse_amount(value) is None


AUG_2018 = 2018 * 12 + 8


@pytest.mark.parametrize(
    "value",
    [
        pd.Timestamp("2018-08-31"),
        dt.datetime(2018, 8, 31, 13, 45),
        dt.date(2018, 8, 31),
        np.datetime64("2018-08-31"),
        "2018-08-31",
        "2018-08-31 13:45:00",
        "2018-08",
        "2018/8/31",
        "20180831",
        "08/31/2018",
        "Aug 31, 2018",
        "31 August 2018",
        "August 2018",
        43343,
        43343.0,
        "43343",
        pd.Timestamp("2018-08-31 23:00", tz="US/Eastern"),
    ],
)
def test_parse_date_maps_formats_to_period(value):
    parsed = parse_date(value, day_first=False)
    assert parsed is not None
    assert parsed.period_index == AUG_2018
    assert parsed.label == "2018-08"


def test_parse_date_day_first_reads_day_month_year():
    assert parse_date("31/08/2018", day_first=True).label == "2018-08"
    assert parse_date("03/04/2018", day_first=True).label == "2018-04"
    assert parse_date("03/04/2018", day_first=False).label == "2018-03"


@pytest.mark.parametrize(
    "value", ["not a date", "2018-13-01", "Q3 2018", 0, -5, 10, "", None, True, "31/31/2018"]
)
def test_parse_date_rejects_invalid(value):
    assert parse_date(value, day_first=False) is None


def test_period_label_round_trips():
    assert period_label(AUG_2018) == "2018-08"
    assert period_label(2018 * 12 + 12) == "2018-12"


def test_ambiguity_detection():
    assert is_ambiguous_numeric_date("03/04/2018")
    assert not is_ambiguous_numeric_date("04/04/2018")
    assert not is_ambiguous_numeric_date("13/04/2018")
    assert not is_ambiguous_numeric_date(43343)
