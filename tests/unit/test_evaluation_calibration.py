from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ledgerclf.calibration import (
    brier_score,
    choose_threshold,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    reliability_table,
    threshold_table,
)
from ledgerclf.evaluation import (
    compute_metrics,
    confusion_table,
    grouped_splits,
    pair_seen_mask,
    per_class_report,
    random_splits,
    run_cv,
    run_train_test,
)
from ledgerclf.labels import CANONICAL_CLASSES
from ledgerclf.models import make_logistic, softmax
from ledgerclf.splitting import out_of_time_mask

C = list(CANONICAL_CLASSES)


def test_compute_metrics_on_known_case():
    y_true = np.array(["Category_1", "Category_1", "Category_2", "Category_3"], dtype=object)
    y_pred = np.array(["Category_1", "Category_2", "Category_2", "Category_1"], dtype=object)
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.5
    assert metrics["recall_Category_2"] == 1.0
    assert np.isnan(metrics["recall_Category_5"])
    assert 0 < metrics["macro_f1_5"] < metrics["accuracy"]


def test_compute_metrics_with_probabilities():
    y = np.array(["Category_1", "Category_2"], dtype=object)
    proba = np.zeros((2, 6))
    proba[0, 0], proba[1, 1] = 1.0, 1.0
    metrics = compute_metrics(y, y, proba)
    assert metrics["brier"] == 0.0 and metrics["ece"] == 0.0
    assert metrics["log_loss"] < 1e-9


def test_reports_tables_have_canonical_shape():
    y = np.array(["Category_1", "Category_2"], dtype=object)
    assert per_class_report(y, y)["class"].tolist() == C
    table = confusion_table(y, y)
    assert table.shape == (6, 7)
    assert table.set_index("true_label").loc["Category_1", "Category_1"] == 1


def test_temperature_recovers_known_scale():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, size=(4000, 6))
    labels = np.array([rng.choice(C, p=p) for p in softmax(true_logits)], dtype=object)
    temperature = fit_temperature(true_logits * 3.0, labels, (0.05, 20.0))
    assert temperature == pytest.approx(3.0, rel=0.1)
    assert negative_log_likelihood(softmax(true_logits), labels) < negative_log_likelihood(
        softmax(true_logits * 3.0), labels
    )


def test_ece_brier_reliability_and_thresholds():
    proba = np.array([[0.9, 0.1, 0, 0, 0, 0], [0.6, 0.4, 0, 0, 0, 0], [0.2, 0.8, 0, 0, 0, 0]])
    y = np.array(["Category_1", "Category_2", "Category_2"], dtype=object)
    table = reliability_table(proba, y, 10)
    assert table["rows"].sum() == 3
    assert expected_calibration_error(proba, y, 10) == pytest.approx((0.1 + 0.6 + 0.2) / 3)
    assert brier_score(proba, y) == pytest.approx((0.02 + 0.72 + 0.08) / 3)
    thresholds = threshold_table(proba, y, [0.5, 0.7, 0.85])
    assert thresholds["rows_flagged"].tolist() == [0, 1, 2]
    assert thresholds["accuracy_kept"].tolist() == pytest.approx([2 / 3, 1.0, 1.0])
    assert choose_threshold(thresholds, 0.975, 0.9) == 0.7
    assert choose_threshold(thresholds.assign(accuracy_kept=0.5), 0.975, 0.9) == 0.9


@pytest.fixture(scope="module")
def small(training):
    rows = np.arange(0, len(training.y), 4)
    return training.frame.iloc[rows].reset_index(drop=True), training.y[rows], training.groups[rows]


def test_run_cv_pools_out_of_fold_predictions(small, settings):
    frame, y, groups = small
    splits = grouped_splits(y, groups, (0, 1), 3)
    result = run_cv("logistic", make_logistic(settings.linear), frame, y, splits, n_jobs=1)
    assert len(result.per_fold) == 6 and len(result.per_seed) == 2
    for seed in (0, 1):
        assert (result.fold_of_row[seed] >= 0).all()
        assert result.oof_proba[seed].shape == (len(y), 6)
        np.testing.assert_allclose(result.oof_proba[seed].sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(
            softmax(result.oof_logits[seed]), result.oof_proba[seed], atol=1e-9
        )
    assert 0.85 < result.summary["accuracy_mean"] <= 1.0


def test_grouped_splits_never_share_groups(small):
    _, y, groups = small
    for split in grouped_splits(y, groups, (0,), 5):
        assert not set(groups[split.train_idx]) & set(groups[split.test_idx])
    assert len(random_splits(y, 0, 5)) == 5


def test_train_test_counts_rare_alarms(small, settings):
    frame, y, _ = small
    mask = out_of_time_mask(frame, settings.oot_cutoff)
    held = run_train_test(make_logistic(settings.linear), frame, y, mask)
    assert held.train_rows == mask.sum() and held.test_rows == (~mask).sum()
    assert held.rare_alarms >= 0 and "accuracy" in held.metrics


def test_pair_seen_mask():
    frame = pd.DataFrame({"Col1": ["A", "A", "B"], "Col6": ["x", "x", "x"]})
    assert pair_seen_mask(frame, np.array([0]), np.array([1, 2])).tolist() == [True, False]
