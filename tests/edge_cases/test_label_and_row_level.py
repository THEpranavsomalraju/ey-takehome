"""Edge cases L01 to L04 and R01 to R04 at the validation layer.

Metric and prediction assertions for the same IDs live in
``test_prediction_level.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ledgerclf.labels import normalize_labels
from ledgerclf.schema import INPUT_COLUMNS, LABEL_COLUMN
from tests.conftest import HEADER, SAMPLE_ROWS, csv_bytes, process

pytestmark = pytest.mark.edge

RAW_SPELLINGS = [
    "Category_1",
    "category_1",
    "Category_2",
    "Category2",
    "Category_3",
    "Category 3",
    "Category _3",
    "Category4",
    "Category 5",
    "Categry_6",
    "Category_6",
]


def _labeled(labels: list[object]) -> pd.DataFrame:
    rows = [[*SAMPLE_ROWS[i % len(SAMPLE_ROWS)], label] for i, label in enumerate(labels)]
    out, _, _ = process(csv_bytes(rows, header=[*HEADER, LABEL_COLUMN]), "labeled.csv")
    return out


def test_l01_all_raw_spellings_normalize():
    out = _labeled(RAW_SPELLINGS)
    canonical, _ = normalize_labels(out[LABEL_COLUMN])
    assert canonical.notna().all()
    assert set(canonical) == {f"Category_{i}" for i in range(1, 7)}


def test_l02_unknown_labels_map_to_none():
    out = _labeled(["Category_9", "foo", None, "Category_1"])
    canonical, _ = normalize_labels(out[LABEL_COLUMN])
    assert canonical.tolist() == [None, None, None, "Category_1"]
    assert len(out) == 4


def test_l03_empty_label_column_keeps_every_row():
    out = _labeled([None, None, None])
    assert len(out) == 3
    assert out[LABEL_COLUMN].isna().all()


def test_l04_partial_labels_keep_every_row():
    out = _labeled(["Category_2", None, "Category_1"])
    canonical, _ = normalize_labels(out[LABEL_COLUMN])
    assert canonical.notna().sum() == 2
    assert len(out) == 3


def test_r01_duplicate_rows_coerce_identically():
    out, _, _ = process(csv_bytes([SAMPLE_ROWS[0]] * 4), "dupes.csv")
    values = out[list(INPUT_COLUMNS)].astype(str)
    assert values.duplicated(keep="first").sum() == 3


def test_r02_row_order_is_preserved():
    shuffled = [SAMPLE_ROWS[2], SAMPLE_ROWS[0], SAMPLE_ROWS[1]]
    out, _, _ = process(csv_bytes(shuffled), "order.csv")
    assert out["Col1"].tolist() == [row[0] for row in shuffled]
    assert out["row_number"].tolist() == [1, 2, 3]


def test_r03_training_rows_reuploaded_coerce_like_training(training, tmp_path):
    subset = training.frame.head(50)
    path = tmp_path / "train_subset.csv"
    subset[list(INPUT_COLUMNS)].to_csv(path, index=False)
    out, _, _ = process(path, path.name)
    compare = [*INPUT_COLUMNS, "amount", "period"]
    pd.testing.assert_frame_equal(
        out[compare].reset_index(drop=True),
        subset[compare].reset_index(drop=True),
        check_dtype=False,
    )


def test_r04_holdout_xlsx_and_csv_coerce_identically(settings):
    xlsx, _, _ = process(settings.paths.holdout / "holdout_features.xlsx", "holdout_features.xlsx")
    csv_frame, _, _ = process(
        settings.paths.holdout / "holdout_features.csv", "holdout_features.csv"
    )
    compare = [*INPUT_COLUMNS, "amount", "period", "row_number"]
    pd.testing.assert_frame_equal(xlsx[compare], csv_frame[compare], check_dtype=False)
