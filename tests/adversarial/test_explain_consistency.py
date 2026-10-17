"""explain() must describe the row it was asked for, with the model that predicted it.

docs/02 section 8: ``explain(result, row_number)`` takes the row number as shown in
the predictions table. Today it indexes by table position, so a re-ordered table
(dashboards sort by confidence) yields a payload mixing two rows. It also accepts a
result produced by another bundle, mixing one model's probabilities with the other
model's contributions.
"""

from __future__ import annotations

import pytest

from ledgerclf.artifacts import FALLBACK_BUNDLE, load_bundle
from ledgerclf.exceptions import LedgerClfError
from ledgerclf.service import PredictionService
from tests.adversarial._support import ROW_A, ROW_B, csv_text

ROW_C = ["Word990001", "1", "5", "Word990002", "2018-01-31", "Word990003", "NoDoc"]


def test_explain_finds_row_by_row_number_after_table_is_reordered(service):
    result = service.predict_file(csv_text([ROW_A, ROW_B, ROW_C]).encode("utf-8"), "three.csv")
    expected = service.explain(result, 1)
    result.predictions = result.predictions.iloc[::-1].reset_index(drop=True)
    payload = service.explain(result, 1)
    assert payload["row_number"] == 1
    assert payload["input"] == expected["input"]
    assert payload["contributions"] == expected["contributions"]


def test_explain_refuses_or_matches_a_result_from_another_model(service, settings):
    if service.bundle.is_fallback:
        pytest.skip("needs the champion model to produce the result")
    result = service.predict_file(csv_text([ROW_A, ROW_B]).encode("utf-8"), "two.csv")
    fallback = PredictionService(load_bundle(FALLBACK_BUNDLE, settings.paths), "test", settings)
    try:
        payload = fallback.explain(result, 1)
    except LedgerClfError:
        return
    assert payload["model"]["name"] == result.model_used
