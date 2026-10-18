"""Header matching: business column names, and refusing to guess a column order."""

from __future__ import annotations

import pandas as pd
import pytest

from ledgerclf.config import Limits
from ledgerclf.exceptions import SchemaError
from ledgerclf.schema import LABEL_COLUMN, resolve_columns

ROWS = [
    ["Word16 Word17", "10719", "1200", "Word61", "2018-08-31", "Word563 Word575", "NoDoc"],
    ["Word19 Word20", "65975", "75", "Word562", "2018-07-31", "Word563", "Doc1"],
    ["Word8 Word9", "4410", "310.5", "Word70", "2018-06-30", "Word564", "NoDoc"],
]


def _raw(header: list[str] | None, rows: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame(([header] if header else []) + rows, dtype=object)


def test_business_names_in_any_order_map_by_name():
    header = [
        "Amount",
        "Vendor",
        "GL Account",
        "Description",
        "Doc Type",
        "Posting Date",
        "Invoice #",
    ]
    order = [2, 0, 5, 3, 6, 4, 1]
    frame, mapping, issues = resolve_columns(
        _raw(header, [[r[i] for i in order] for r in ROWS]), Limits()
    )
    assert len(frame) == 3
    assert frame["Col1"].tolist() == [r[0] for r in ROWS]
    assert frame["Col3"].tolist() == [r[2] for r in ROWS]
    assert mapping["GL Account"] == "Col6" and mapping["Invoice #"] == "Col2"
    assert not [i for i in issues if i.code in ("POSITIONAL_MAPPING", "HEADER_NOT_RECOGNIZED")]


def test_unknown_names_in_the_wrong_order_are_refused():
    header = ["Fld A", "Fld B", "Fld C", "Fld D", "Fld E", "Fld F", "Fld G"]
    order = [2, 0, 5, 3, 6, 4, 1]
    with pytest.raises(SchemaError) as caught:
        resolve_columns(_raw(header, [[r[i] for i in order] for r in ROWS]), Limits())
    assert caught.value.code == "MISSING_REQUIRED_COLUMNS"


def test_unknown_header_in_the_right_order_gives_one_clear_warning():
    header = ["Fld A", "Fld B", "Fld C", "Fld D", "Fld E", "Fld F", "Fld G"]
    frame, _, issues = resolve_columns(_raw(header, ROWS), Limits())
    assert len(frame) == 3
    assert [i.code for i in issues] == ["HEADER_NOT_RECOGNIZED"]


def test_extra_column_without_labels_is_not_used_as_the_label():
    rows = [[*r, "free text note"] for r in ROWS]
    frame, _, issues = resolve_columns(_raw(None, rows), Limits())
    assert LABEL_COLUMN not in frame.columns
    assert "was ignored" in issues[0].message


def test_extra_column_with_labels_is_the_label():
    rows = [[*r, "Category_1"] for r in ROWS]
    frame, _, _ = resolve_columns(_raw(None, rows), Limits())
    assert frame[LABEL_COLUMN].tolist() == ["Category_1"] * 3
