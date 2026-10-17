from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ledgerclf.labels import CANONICAL_CLASSES
from ledgerclf.models import (
    ABSENT_CLASS_LOGIT,
    PoweredClassWeight,
    class_weights,
    make_blend,
    make_complement_nb,
    make_dummy,
    make_lightgbm,
    make_linear_svc,
    make_logistic,
    make_lookup,
    make_probability_blend,
    softmax,
    to_canonical,
)

SMALL_TREE = {"n_estimators": 40, "learning_rate": 0.1}


@pytest.fixture(scope="module")
def sample(training):
    rng = np.random.default_rng(0)
    rare = np.flatnonzero(training.y != "Category_1")
    common = rng.choice(np.flatnonzero(training.y == "Category_1"), 900, replace=False)
    rows = np.sort(np.concatenate([rare, common]))
    return training.frame.iloc[rows].reset_index(drop=True), training.y[rows]


@pytest.fixture(scope="module")
def fitted_blend(sample, settings):
    frame, y = sample
    tree = replace(settings.tree, **SMALL_TREE)
    return make_blend(settings.linear, tree, settings.blend, 0).fit(frame, y)


def test_class_weights_follow_power():
    y = np.array(["a"] * 8 + ["b"] * 2, dtype=object)
    assert class_weights(y, 0.0) == {"a": 1.0, "b": 1.0}
    assert class_weights(y, 1.0) == {"a": 1.0, "b": 4.0}
    assert class_weights(y, 0.5)["b"] == pytest.approx(2.0)


def test_powered_class_weight_sets_weights_per_fit(sample):
    frame, y = sample
    from ledgerclf.config import FeatureParams
    from ledgerclf.features import build_feature_pipeline

    X = build_feature_pipeline(FeatureParams()).fit_transform(frame)
    model = PoweredClassWeight(LogisticRegression(max_iter=2000), power=1.0).fit(X, y)
    weights = model.estimator_.get_params()["class_weight"]
    assert weights["Category_1"] == 1.0 and weights["Category_5"] > 100
    unweighted = PoweredClassWeight(LogisticRegression(max_iter=2000), power=0.0).fit(X, y)
    assert unweighted.estimator_.get_params()["class_weight"] is None
    assert model.predict_proba(X).shape == (len(y), len(model.classes_))


def test_to_canonical_fills_absent_classes():
    values = np.array([[1.0, 2.0]])
    out = to_canonical(values, np.array(["Category_2", "Category_1"]), fill=-5.0)
    assert out.tolist() == [[2.0, 1.0, -5.0, -5.0, -5.0, -5.0]]


def test_softmax_handles_absent_class_logits():
    logits = np.array([[1.0, 0.0, ABSENT_CLASS_LOGIT]])
    proba = softmax(logits)
    assert proba.sum() == pytest.approx(1.0)
    assert proba[0, 2] == 0.0


def test_blend_probabilities_are_softmax_of_logits(fitted_blend, sample):
    frame, _ = sample
    model = fitted_blend.named_steps["model"]
    X = fitted_blend.named_steps["features"].transform(frame.head(50))
    np.testing.assert_allclose(
        model.predict_proba(X), softmax(model.decision_function(X)), atol=1e-12
    )
    assert list(model.classes_) == list(CANONICAL_CLASSES)


def test_blend_contributions_are_exactly_additive(fitted_blend, sample):
    frame, _ = sample
    model = fitted_blend.named_steps["model"]
    X = fitted_blend.named_steps["features"].transform(frame.head(60))
    for temperature in (1.0, 1.7):
        model.temperature = temperature
        parts = model.contributions(X)
        np.testing.assert_allclose(parts.values.sum(axis=2) + parts.bias, parts.logits, atol=1e-6)
    model.temperature = 1.0


def test_blend_member_class_orders_match(fitted_blend):
    model = fitted_blend.named_steps["model"]
    assert list(model.linear_.classes_) == list(model.tree_.classes_)


def test_blend_without_a_class_gives_absent_logits(sample, settings):
    frame, y = sample
    keep = y != "Category_5"
    tree = replace(settings.tree, **SMALL_TREE)
    pipeline = make_blend(settings.linear, tree, settings.blend, 0).fit(frame[keep], y[keep])
    X = pipeline.named_steps["features"].transform(frame.head(5))
    parts = pipeline.named_steps["model"].contributions(X)
    c5 = CANONICAL_CLASSES.index("Category_5")
    assert (parts.logits[:, c5] == ABSENT_CLASS_LOGIT).all()
    assert (parts.values[:, c5, :] == 0).all()
    assert pipeline.predict_proba(frame.head(5))[:, c5].max() == 0.0


def test_blend_is_deterministic(sample, settings):
    frame, y = sample
    tree = replace(settings.tree, **SMALL_TREE)
    first = (
        make_blend(settings.linear, tree, settings.blend, 0)
        .fit(frame, y)
        .predict_proba(frame.head(30))
    )
    second = (
        make_blend(settings.linear, tree, settings.blend, 0)
        .fit(frame, y)
        .predict_proba(frame.head(30))
    )
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "factory",
    [
        lambda s: make_dummy(),
        lambda s: make_lookup(),
        lambda s: make_complement_nb(0.1),
        lambda s: make_linear_svc(1.0, 0.5, 0, s.benchmark.svc_max_iter),
        lambda s: make_logistic(s.linear),
        lambda s: make_lightgbm(replace(s.tree, **SMALL_TREE), 0),
        lambda s: make_probability_blend(s.linear, replace(s.tree, **SMALL_TREE), 0),
    ],
    ids=[
        "dummy",
        "lookup",
        "complement_nb",
        "linear_svc",
        "logistic",
        "lightgbm",
        "probability_blend",
    ],
)
def test_every_model_fits_and_predicts_canonical_labels(factory, sample, settings):
    frame, y = sample
    model = factory(settings).fit(frame, y)
    predictions = model.predict(frame.head(20))
    assert set(predictions) <= set(CANONICAL_CLASSES)
    assert len(predictions) == 20


def test_lookup_falls_back_from_pair_to_vendor_to_global(sample):
    frame, y = sample
    lookup = make_lookup().fit(frame, y)
    unseen = frame.head(2).copy()
    unseen.loc[:, "Col6"] = "Word999999"
    unseen.loc[1, "Col1"] = "Word888888"
    predictions = lookup.predict(unseen)
    vendor = lookup._keys(frame.head(1))["vendor"].iloc[0]
    assert predictions[0] == lookup.vendor_[(vendor,)]
    assert predictions[1] == lookup.global_
