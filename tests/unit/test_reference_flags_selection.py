from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ledgerclf.config import get_settings
from ledgerclf.evaluation import CVResult
from ledgerclf.flags import (
    DEFAULT_SEVERITIES,
    LOW_CONFIDENCE,
    FlagSeverity,
    ReviewPolicy,
    apply_review_policy,
    compute_row_flags,
    flag_strings,
)
from ledgerclf.models import softmax
from ledgerclf.reference import ReferenceStats, build_reference, pair_key
from ledgerclf.selection import (
    CandidateScore,
    accuracy_by_flag,
    apply_decision_rule,
    calibrate,
    candidate_name,
    evidence_policy,
    fallback_name,
)
from ledgerclf.tuning import (
    TunedParams,
    linear_grid,
    search_space,
    tree_distributions,
)
from tests.conftest import process, sample_frame, xlsx_bytes


@pytest.fixture(scope="module")
def reference(training):
    return build_reference(training.frame, training.y)


def test_reference_counts_and_round_trip(reference, training):
    assert reference.training_rows == 5309
    assert sum(reference.class_counts.values()) == 5309
    assert "word16 word17 word18 word7" in reference.known_vendors
    rebuilt = ReferenceStats.from_dict(json.loads(json.dumps(reference.to_dict())))
    assert rebuilt == reference
    assert reference.amount_percentile(reference.amount_min) == 0.0
    assert reference.amount_percentile(reference.amount_max) == 1.0
    assert 0.4 < reference.amount_percentile(reference.amount_median) < 0.6


def test_pair_key_normalizes():
    assert pair_key(" Word1 ", "WORD2") == "word1 || word2"


def _flags_for(rows: list[list[object]], reference) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7"])
    coerced, _, _ = process(xlsx_bytes({"data": frame}), "flags.xlsx")
    return compute_row_flags(coerced, reference)


def test_known_row_has_no_flags(reference):
    flags = _flags_for(sample_frame().head(1).to_numpy().tolist(), reference)
    assert not flags.to_numpy().any()


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({0: "Word999999 Word888888"}, {"UNSEEN_VENDOR"}),
        ({5: "Word777777"}, {"UNSEEN_ACCOUNT"}),
        ({6: "Doc9"}, {"UNSEEN_DOC_TYPE"}),
        ({0: None}, {"TEXT_MISSING"}),
        ({2: "abc"}, {"AMOUNT_IMPUTED"}),
        ({2: 1e15}, {"AMOUNT_OUT_OF_RANGE"}),
        ({4: "not a date"}, {"DATE_IMPUTED"}),
        ({4: "2030-01-31"}, {"DATE_OUT_OF_RANGE"}),
        ({3: "x" * 5000}, {"TEXT_TRUNCATED"}),
        (
            {0: "Word900001", 3: "Word900002", 5: "Word900003"},
            {"UNSEEN_VENDOR", "UNSEEN_ACCOUNT", "ALL_TOKENS_UNSEEN"},
        ),
    ],
)
def test_each_flag_fires_on_its_condition(reference, change, expected):
    row = list(sample_frame().iloc[0])
    for position, value in change.items():
        row[position] = value
    flags = _flags_for([row], reference)
    assert {c for c in flags.columns if flags[c].iloc[0]} == expected


def test_unseen_pair_of_known_parts(reference, training):
    first = training.frame.iloc[0]
    other_account = next(
        a for a in training.frame["Col6"] if a.lower() != str(first["Col6"]).lower()
    )
    row = [first["Col1"], "1", 100.0, "Word61", "2018-08-31", other_account, "NoDoc"]
    known_pair = pair_key(first["Col1"], other_account) in reference.known_pairs
    flags = _flags_for([row], reference)
    assert bool(flags["UNSEEN_VENDOR_ACCOUNT_PAIR"].iloc[0]) is (not known_pair)


def test_review_policy_routes_critical_and_low_confidence():
    flags = pd.DataFrame(
        {"UNSEEN_VENDOR": [True, False, False], "UNSEEN_VENDOR_ACCOUNT_PAIR": [False, True, False]}
    )
    policy = ReviewPolicy(threshold=0.8)
    full, review = apply_review_policy(flags, np.array([0.99, 0.99, 0.5]), policy)
    assert review.tolist() == [True, False, True]
    assert full[LOW_CONFIDENCE].tolist() == [False, False, True]
    assert flag_strings(full) == ["UNSEEN_VENDOR", "UNSEEN_VENDOR_ACCOUNT_PAIR", "LOW_CONFIDENCE"]
    assert ReviewPolicy.from_dict(json.loads(json.dumps(policy.to_dict()))) == policy


def _candidate(name, family, power, cv, f1, oot):
    return CandidateScore(name, family, power, cv, 0.01, f1, oot, 0.7, 1)


def test_decision_rule_clear_winner():
    candidates = [
        _candidate("logistic_p0", "logistic", 0.0, 0.950, 0.72, 0.94),
        _candidate("logodds_blend_p0", "logodds_blend", 0.0, 0.956, 0.73, 0.945),
    ]
    decision = apply_decision_rule(candidates, get_settings().decision_rule)
    assert decision.champion == "logodds_blend_p0"
    assert any("chosen" in step for step in decision.trail)


def test_decision_rule_guardrails_exclude_weak_candidates():
    candidates = [
        _candidate("a", "logodds_blend", 0.0, 0.960, 0.60, 0.95),
        _candidate("b", "logodds_blend", 0.25, 0.955, 0.75, 0.95),
        _candidate("c", "logistic", 0.0, 0.954, 0.74, 0.90),
    ]
    decision = apply_decision_rule(candidates, get_settings().decision_rule)
    assert decision.eligible == ["b"]
    assert decision.champion == "b"


def test_decision_rule_ties_go_to_out_of_time_then_simplicity():
    rule = get_settings().decision_rule
    tie_oot = [
        _candidate("logistic_p0", "logistic", 0.0, 0.9540, 0.74, 0.940),
        _candidate("logodds_blend_p0", "logodds_blend", 0.0, 0.9545, 0.74, 0.945),
    ]
    assert apply_decision_rule(tie_oot, rule).champion == "logodds_blend_p0"
    full_tie = [
        _candidate("logodds_blend_p0", "logodds_blend", 0.0, 0.9545, 0.74, 0.9450),
        _candidate("logistic_p0.25", "logistic", 0.25, 0.9540, 0.74, 0.9445),
    ]
    assert apply_decision_rule(full_tie, rule).champion == "logistic_p0.25"


def test_fallback_is_best_logistic_and_names():
    candidates = [
        _candidate("logistic_p0", "logistic", 0.0, 0.951, 0.7, 0.94),
        _candidate("logistic_p0.25", "logistic", 0.25, 0.952, 0.7, 0.94),
        _candidate("logodds_blend_p0", "logodds_blend", 0.0, 0.96, 0.7, 0.94),
    ]
    assert fallback_name(candidates) == "logistic_p0.25"
    assert candidate_name("logodds_blend", 0.25) == "logodds_blend_p0.25"


def test_evidence_policy_upgrades_and_keeps_defaults():
    table = pd.DataFrame(
        {
            "flag": [
                "UNSEEN_VENDOR",
                "UNSEEN_ACCOUNT",
                "UNSEEN_VENDOR_ACCOUNT_PAIR",
                "TEXT_MISSING",
                LOW_CONFIDENCE,
                "(all rows)",
            ],
            "rows": [100, 50, 300, 0, 400, 5000],
            "accuracy": [0.95, 0.85, 0.94, np.nan, 0.6, 0.954],
        }
    )
    policy, notes = evidence_policy(table, 0.8, get_settings())
    assert policy.severities["UNSEEN_VENDOR"] is FlagSeverity.WARNING
    assert policy.severities["UNSEEN_ACCOUNT"] is FlagSeverity.CRITICAL
    assert policy.severities["TEXT_MISSING"] is DEFAULT_SEVERITIES["TEXT_MISSING"]
    assert policy.severities[LOW_CONFIDENCE] is FlagSeverity.CRITICAL
    assert any("keeps default" in n for n in notes)


def test_accuracy_by_flag_totals():
    flags = pd.DataFrame({"A": [True, False, False, True]})
    table = accuracy_by_flag(flags, np.array([True, True, False, False]))
    assert table.set_index("flag").loc["A", "accuracy"] == 0.5
    assert table.set_index("flag").loc["(all rows)", "rows"] == 4


def test_calibrate_lowers_ece_on_overconfident_logits():
    rng = np.random.default_rng(1)
    classes = np.array(
        ["Category_1", "Category_2", "Category_3", "Category_4", "Category_5", "Category_6"],
        dtype=object,
    )
    logits = rng.normal(0, 1.5, size=(3000, 6))
    y = np.array([rng.choice(classes, p=p) for p in softmax(logits)], dtype=object)
    result = CVResult("m", pd.DataFrame(), pd.DataFrame(), {}, oof_logits={0: logits * 2.5})
    outcome = calibrate(result, y, get_settings(), seed=0)
    assert outcome.temperature == pytest.approx(2.5, rel=0.15)
    assert outcome.summary["ece_after"] < outcome.summary["ece_before"]
    assert set(outcome.reliability["stage"]) == {"before", "after"}


def test_tree_space_keeps_l2_regularization():
    space = get_settings().tuning.tree_space
    assert min(space["reg_lambda"]) > 0


def test_search_spaces_and_grids():
    settings = get_settings()
    full, quick = search_space(settings, quick=False), search_space(settings, quick=True)
    configs = sum(
        len(g["model__estimator__C"]) * len(g["model__power"])
        for g in linear_grid(full.linear_C, full.linear_powers, full.ngram_ranges)
    )
    assert configs == 30
    assert quick.tree_trials < full.tree_trials
    assert set(tree_distributions(full.tree_space)) >= {
        "model__power",
        "model__estimator__num_leaves",
    }
    assert json.dumps(full.to_dict())


def test_tuned_params_round_trip():
    settings = get_settings()
    payload = json.loads(
        (settings.paths.reports / "tuning" / "best_params.json").read_text(encoding="utf-8")
    )
    tuned = TunedParams.from_dict(payload, settings)
    assert TunedParams.from_dict(json.loads(json.dumps(tuned.to_dict())), settings) == tuned
