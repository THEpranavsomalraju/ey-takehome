"""Prediction-level assertions for the F, S, V, L, and R families (docs/05 section 4).

The validation-layer assertions live in the sibling files; these tests run
the full service so that every catalog expectation about predictions,
flags, review routing, and metrics is checked end to end.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from ledgerclf.schema import INPUT_COLUMNS
from tests.conftest import csv_bytes, xlsx_bytes

pytestmark = pytest.mark.edge


@pytest.fixture(scope="module")
def holdout_rows(settings) -> pd.DataFrame:
    frame = pd.read_csv(
        settings.paths.holdout / "holdout_features.csv", dtype=object, keep_default_na=False
    )
    return frame.head(40)


@pytest.fixture(scope="module")
def unseen_rows(holdout_rows) -> pd.DataFrame:
    """Holdout features with new amounts, so they match neither holdout nor training rows.

    Labels attached to real holdout rows stay unread before the holdout
    evaluation (DECISIONS.md D35), so label-handling tests use these rows.
    """
    return holdout_rows.assign(Col3=[f"{777000 + i}.13" for i in range(len(holdout_rows))])


def _labels(result) -> list[str]:
    return result.predictions["predicted_label"].tolist()


def _probabilities(result) -> np.ndarray:
    return result.predictions[
        [c for c in result.predictions.columns if c.startswith("p_")]
    ].to_numpy()


def _csv(rows: pd.DataFrame, **kwargs) -> bytes:
    return csv_bytes(rows.to_numpy().tolist(), header=list(rows.columns), **kwargs)


def test_f07_encodings_give_identical_predictions(service, holdout_rows):
    reference = service.predict_file(_csv(holdout_rows), "utf8.csv")
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        result = service.predict_file(_csv(holdout_rows, encoding=encoding), "enc.csv")
        np.testing.assert_allclose(_probabilities(result), _probabilities(reference), atol=1e-12)


def test_f08_delimiters_give_identical_predictions(service, holdout_rows):
    reference = service.predict_file(_csv(holdout_rows), "comma.csv")
    for delimiter in (";", "\t", "|"):
        result = service.predict_file(_csv(holdout_rows, delimiter=delimiter), "delim.csv")
        assert _labels(result) == _labels(reference)


def test_f10_line_endings_give_identical_predictions(service, holdout_rows):
    reference = service.predict_file(_csv(holdout_rows), "lf.csv")
    for ending in ("\r\n", "\r"):
        assert _labels(
            service.predict_file(_csv(holdout_rows, line_ending=ending), "e.csv")
        ) == _labels(reference)


def test_f13_raw_file_gives_5899_predictions_and_metric_keys(service, settings):
    result = service.predict_file(
        settings.paths.raw_data.read_bytes(), settings.paths.raw_data.name
    )
    assert len(result.predictions) == 5899
    # Structure only: the raw file contains holdout rows (DECISIONS.md D26, D35).
    assert {"accuracy", "macro_f1_5", "labeled_rows"} <= set(result.metrics)
    holdout = len(pd.read_csv(settings.paths.holdout / "holdout_features.csv", dtype=object))
    assert result.metrics["labeled_rows"] == 5899 - holdout
    codes = {i.code for i in result.report.issues}
    assert {"HOLDOUT_METRICS_WITHHELD", "TRAINING_ROWS_IN_UPLOAD"} <= codes


def test_s04_shuffled_columns_give_identical_predictions(service, holdout_rows):
    reference = service.predict_file(_csv(holdout_rows), "ordered.csv")
    shuffled = holdout_rows[["Col7", "Col3", "Col1", "Col6", "Col2", "Col5", "Col4"]]
    assert _labels(service.predict_file(_csv(shuffled), "shuffled.csv")) == _labels(reference)


def test_s07_positional_file_with_labels_scores_them(service, unseen_rows):
    rows = unseen_rows.assign(label="Category_1").to_numpy().tolist()
    result = service.predict_file(csv_bytes(rows, header=None), "noheader.csv")
    assert result.metrics is not None and result.metrics["labeled_rows"] == len(rows)


def test_v02_missing_vendor_is_flagged_for_review(service, holdout_rows):
    rows = holdout_rows.head(3).copy()
    rows.loc[rows.index[0], "Col1"] = ""
    result = service.predict_file(_csv(rows), "missing.csv")
    assert "TEXT_MISSING" in result.predictions["flags"].iloc[0]
    assert bool(result.predictions["review_required"].iloc[0])


def test_v04_v05_messy_whitespace_and_case_match_clean_values(service, holdout_rows):
    clean = service.predict_file(_csv(holdout_rows.head(10)), "clean.csv")
    messy = holdout_rows.head(10).copy()
    for column in ("Col1", "Col4", "Col6"):
        messy[column] = messy[column].map(lambda v: f"  {str(v).lower()}\u00a0 " if v else v)
    result = service.predict_file(_csv(messy), "messy.csv")
    np.testing.assert_allclose(_probabilities(result), _probabilities(clean), atol=1e-12)


@pytest.mark.parametrize(
    ("column", "value", "flag"),
    [
        ("Col1", "Word999991 Word999992", "UNSEEN_VENDOR"),
        ("Col3", "abc", "AMOUNT_IMPUTED"),
        ("Col3", "1e15", "AMOUNT_OUT_OF_RANGE"),
        ("Col5", "not a date", "DATE_IMPUTED"),
        ("Col5", "2030-06-30", "DATE_OUT_OF_RANGE"),
        ("Col7", "Doc4", "UNSEEN_DOC_TYPE"),
        ("Col4", "x" * 5000, "TEXT_TRUNCATED"),
    ],
)
def test_v06_v08_v09_v13_v14_v15_v18_bad_values_predict_with_flags(
    service, holdout_rows, column, value, flag
):
    rows = holdout_rows.head(2).copy()
    rows.loc[rows.index[0], column] = value
    result = service.predict_file(_csv(rows), "flagged.csv")
    assert len(result.predictions) == 2
    assert flag in result.predictions["flags"].iloc[0].split("|")
    assert np.isfinite(_probabilities(result)).all()


def test_l01_l02_l04_label_variants_and_unknowns(service, unseen_rows):
    labels = ["Category 3", "category_1", "Categry_6", "Category_9", "", "foo"] + ["Category_1"] * (
        len(unseen_rows) - 6
    )
    result = service.predict_file(
        _csv(unseen_rows.assign(ClassificationLabel=labels)), "labels.csv"
    )
    assert len(result.predictions) == len(unseen_rows)
    assert result.metrics["labeled_rows"] == len(unseen_rows) - 3
    assert "UNKNOWN_LABELS" in [i.code for i in result.report.issues]
    assert result.predictions["correct"].iloc[3] is None


def test_l03_empty_label_column_gives_no_metrics(service, holdout_rows):
    result = service.predict_file(_csv(holdout_rows.assign(ClassificationLabel="")), "nolabels.csv")
    assert result.metrics is None
    assert len(result.predictions) == len(holdout_rows)


def test_r01_duplicate_rows_get_identical_predictions(service, holdout_rows):
    rows = pd.concat([holdout_rows.head(1)] * 4, ignore_index=True)
    result = service.predict_file(_csv(rows), "dupes.csv")
    assert len(set(map(tuple, _probabilities(result).round(12)))) == 1


def test_r02_output_order_equals_input_order(service, holdout_rows):
    shuffled = holdout_rows.sample(frac=1.0, random_state=3)
    result = service.predict_file(_csv(shuffled), "shuffled_rows.csv")
    assert result.predictions["Col1"].tolist() == shuffled["Col1"].tolist()
    assert result.predictions["row_number"].tolist() == list(range(1, len(shuffled) + 1))


def test_r03_reuploaded_training_rows_match_in_memory_pipeline(service, training):
    rows = training.raw.head(60)[list(INPUT_COLUMNS)]
    result = service.predict_file(xlsx_bytes({"data": rows}), "train_rows.xlsx")
    direct = service.bundle.pipeline.predict_proba(training.frame.head(60))
    np.testing.assert_allclose(_probabilities(result), direct, atol=1e-9)


def test_r04_holdout_xlsx_and_csv_match_each_other_and_golden(service, settings, holdout_result):
    csv_result = service.predict_file(
        (settings.paths.holdout / "holdout_features.csv").read_bytes(), "h.csv"
    )
    np.testing.assert_allclose(
        _probabilities(csv_result), _probabilities(holdout_result), atol=1e-12
    )
    golden = pd.read_csv(settings.paths.artifacts / "golden_holdout_predictions.csv")
    expected = golden[golden["model"] == service.bundle.name]
    assert (
        holdout_result.predictions["predicted_label"].tolist()
        == expected["predicted_label"].tolist()
    )
    np.testing.assert_allclose(
        _probabilities(holdout_result),
        expected[
            [
                f"p_{c}"
                for c in (
                    "Category_1",
                    "Category_2",
                    "Category_3",
                    "Category_4",
                    "Category_5",
                    "Category_6",
                )
            ]
        ].to_numpy(),
        atol=1e-6,
    )


def test_every_prediction_respects_output_guarantees(holdout_result):
    predictions = holdout_result.predictions
    probabilities = _probabilities(holdout_result)
    assert set(predictions["predicted_label"]) <= {f"Category_{i}" for i in range(1, 7)}
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    import json

    json.dumps(holdout_result.to_records())
    json.dumps(holdout_result.summary())
    assert io.StringIO(predictions.to_csv(index=False)).getvalue()
