from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ledgerclf.labels import CANONICAL_CLASSES, FIVE_CLASSES, normalize_label, normalize_labels
from ledgerclf.schema import LABEL_COLUMN

DOCS_MAPPING = {
    "Category_1": 5204,
    "category_1": 11,
    "Category_2": 627,
    "Category2": 4,
    "Category_3": 12,
    "Category 3": 10,
    "Category _3": 1,
    "Category4": 16,
    "Category 5": 2,
    "Categry_6": 10,
    "Category_6": 2,
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Category_1", "Category_1"),
        ("category_1", "Category_1"),
        ("Category2", "Category_2"),
        ("Category _3", "Category_3"),
        ("Category 3", "Category_3"),
        ("Categry_6", "Category_6"),
        ("  CATEGORY_4  ", "Category_4"),
        ("Category_9", None),
        ("foo", None),
        ("", None),
        (None, None),
        (np.nan, None),
        (True, None),
    ],
)
def test_normalize_label_maps_variants(raw, expected):
    assert normalize_label(raw) == expected


def test_label_mapping_reproduces_docs_counts(labeled_raw):
    _, table = normalize_labels(labeled_raw.frame[LABEL_COLUMN])
    observed = dict(zip(table["raw_label"], table["rows"], strict=True))
    assert observed == DOCS_MAPPING
    assert set(table["canonical_label"]) == set(CANONICAL_CLASSES)


def test_canonical_class_counts_match_docs(labeled_raw):
    counts = pd.Series(labeled_raw.y).value_counts().to_dict()
    assert counts == {
        "Category_1": 5215,
        "Category_2": 631,
        "Category_3": 23,
        "Category_4": 16,
        "Category_6": 12,
        "Category_5": 2,
    }


def test_five_classes_exclude_category_5():
    assert "Category_5" not in FIVE_CLASSES
    assert len(FIVE_CLASSES) == 5


def test_normalize_labels_returns_none_for_unmapped():
    canonical, table = normalize_labels(pd.Series(["Category_1", "bad", None], dtype=object))
    assert canonical.tolist() == ["Category_1", None, None]
    assert table["rows"].sum() == 3
