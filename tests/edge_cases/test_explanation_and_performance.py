"""Edge cases X01 to X05 (explanations) and P01 to P04 (performance budgets)."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from ledgerclf.explain import ExplanationError
from tests.conftest import csv_bytes

pytestmark = pytest.mark.edge

PAYLOAD_KEYS = {
    "schema_version",
    "row_number",
    "source_row",
    "input",
    "prediction",
    "probabilities",
    "model",
    "contributions",
    "evidence",
    "flags",
    "context",
    "fallback_text",
}
# Test limits are three times the docs/05 section 9 budgets, to avoid flaky failures.
SLACK = 3


@pytest.mark.parametrize("row", [0, -1, 591, 10_000, 1.5, True, "2"])
def test_x01_invalid_row_numbers_raise_clear_errors(service, holdout_result, row):
    with pytest.raises(ExplanationError) as caught:
        service.explain(holdout_result, row)
    assert "between 1 and 590" in caught.value.message


def test_x02_row_with_every_token_unseen_explains_with_flags(service):
    row = [
        ["Word990001 Word990002", "1", 1234.5, "Word990003", "2018-08-31", "Word990004", "NoDoc"]
    ]
    result = service.predict_file(csv_bytes(row), "unseen.csv")
    payload = service.explain(result, 1)
    codes = {flag["code"] for flag in payload["flags"]}
    assert {"UNSEEN_VENDOR", "UNSEEN_ACCOUNT", "ALL_TOKENS_UNSEEN"} <= codes
    assert payload["prediction"]["review_required"] is True
    assert not payload["evidence"][0]["seen_in_training"]


def test_x03_payload_is_json_safe_and_matches_schema(service, holdout_result):
    for row in (1, 100, 590):
        payload = service.explain(holdout_result, row)
        assert set(payload) == PAYLOAD_KEYS
        assert payload["schema_version"] == "1.0"
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload
        assert set(payload["probabilities"]) == {f"Category_{i}" for i in range(1, 7)}


def test_x04_fallback_text_is_present_and_deterministic(service, holdout_result):
    first = service.explain(holdout_result, 7)["fallback_text"]
    second = service.explain(holdout_result, 7)["fallback_text"]
    assert first == second
    assert first.startswith("Predicted Category_")
    assert first.isascii()


def test_x05_contributions_add_up_to_the_logit_margin(service, holdout_result):
    for row in range(1, 591, 7):
        payload = service.explain(holdout_result, row)
        parts = payload["contributions"]
        groups = sum(group["contribution"] for group in parts["by_group"])
        assert abs(parts["bias"] + groups - parts["total"]) < 1e-6
        assert abs(parts["total"] - payload["prediction"]["logit_margin"]) < 1e-6


def _rows(settings, count: int) -> bytes:
    holdout = pd.read_csv(
        settings.paths.holdout / "holdout_features.csv", dtype=object, keep_default_na=False
    )
    repeated = pd.concat([holdout] * (count // len(holdout) + 1), ignore_index=True).head(count)
    return repeated.to_csv(index=False).encode("utf-8")


def test_p01_holdout_predicts_within_budget(service, holdout_xlsx):
    started = time.perf_counter()
    result = service.predict_file(holdout_xlsx, "h.xlsx")
    assert len(result.predictions) == 590
    assert time.perf_counter() - started < 2.0 * SLACK


def test_p02_ten_thousand_rows_within_budget(service, settings):
    data = _rows(settings, 10_000)
    started = time.perf_counter()
    result = service.predict_file(data, "ten_thousand.csv")
    assert len(result.predictions) == 10_000
    assert time.perf_counter() - started < 15.0 * SLACK


@pytest.mark.slow
def test_p03_hundred_thousand_rows_within_budget(service, settings):
    import resource

    data = _rows(settings, 100_000)
    started = time.perf_counter()
    result = service.predict_file(data, "hundred_thousand.csv")
    seconds = time.perf_counter() - started
    assert len(result.predictions) == 100_000
    assert seconds < 120.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if __import__("sys").platform == "darwin" else peak * 1024
    assert peak_bytes < 2 * 1024**3


def test_p04_one_explanation_within_budget(service, holdout_result):
    timings = []
    for row in (1, 2, 3):
        started = time.perf_counter()
        service.explain(holdout_result, row)
        timings.append(time.perf_counter() - started)
    assert float(np.median(timings)) < 0.3 * SLACK


@pytest.mark.parametrize("bundle_file", ["model_champion.joblib", "model_fallback.joblib"])
def test_x05_production_bundles_are_additive_for_every_class(settings, training, bundle_file):
    from ledgerclf.artifacts import load_bundle
    from ledgerclf.models import softmax
    from tests.conftest import process

    bundle = load_bundle(bundle_file, settings.paths)
    features = bundle.pipeline.named_steps["features"]
    model = bundle.pipeline.named_steps["model"]
    holdout, _, _ = process(settings.paths.holdout / "holdout_features.xlsx", "h.xlsx")
    rows = np.random.default_rng(settings.random_state).choice(
        len(training.frame), 200, replace=False
    )
    for frame in (holdout, training.frame.iloc[np.sort(rows)]):
        parts = model.contributions(features.transform(frame))
        reconstructed = parts.values.sum(axis=2) + parts.bias
        np.testing.assert_allclose(reconstructed, parts.logits, atol=1e-6)
        np.testing.assert_allclose(
            softmax(parts.logits), bundle.pipeline.predict_proba(frame), atol=1e-9
        )
