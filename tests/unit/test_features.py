from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.base import clone

from ledgerclf.exceptions import ConfigError
from ledgerclf.features import (
    DEFAULT_TRANSFORMERS,
    EXPLANATION_GROUPS,
    AmountFlags,
    ColumnConcat,
    InvoiceFormatFeatures,
    PeriodEncoder,
    SignedLogAmount,
    TextNormalizer,
    TrainingMedianImputer,
    build_feature_pipeline,
    feature_groups,
    normalize_text,
    readable_feature_name,
)
from tests.conftest import process


@pytest.fixture(scope="module")
def fitted(training, settings):
    pipeline = build_feature_pipeline(settings.feature)
    matrix = pipeline.fit_transform(training.frame)
    return pipeline, matrix


def test_text_normalizer_lowercases_and_marks_missing():
    out = TextNormalizer().fit_transform(pd.Series(["Word1 Word2", None, "  ", np.nan]))
    assert out.tolist() == ["word1 word2", "__missing__", "__missing__", "__missing__"]
    two_d = TextNormalizer().fit_transform(pd.DataFrame({"a": ["X"], "b": [None]}))
    assert two_d.tolist() == [["x", "__missing__"]]


def test_normalize_text_handles_numbers():
    assert normalize_text(5) == "5"


def test_column_concat_joins_with_separator():
    out = ColumnConcat(" || ").fit_transform(np.array([["a", "b"], ["c", "d"]], dtype=object))
    assert out.tolist() == [["a || b"], ["c || d"]]


def test_training_median_imputer_uses_fit_median_only():
    imputer = TrainingMedianImputer().fit(np.array([[1.0], [3.0], [np.nan]]))
    assert imputer.transform(np.array([[np.nan], [7.0]])).tolist() == [[2.0], [7.0]]
    empty = TrainingMedianImputer().fit(np.array([[np.nan], [np.nan]]))
    assert empty.transform(np.array([[np.nan]])).tolist() == [[0.0]]


def test_signed_log_keeps_sign():
    out = SignedLogAmount().fit_transform(np.array([[-99.0], [0.0], [99.0]]))
    assert out[:, 0] == pytest.approx([-np.log(100), 0.0, np.log(100)])


@pytest.mark.parametrize(
    ("amount", "negative", "round_"),
    [
        (1680.0, 0, 1),
        (-790.66, 1, 0),
        (1500.0000001, 0, 1),
        (0.0, 0, 0),
        (-50.0, 1, 1),
        (75.5, 0, 0),
    ],
)
def test_amount_flags(amount, negative, round_):
    flags = AmountFlags(10.0).fit_transform(np.array([[amount]]))
    assert flags.tolist() == [[negative, round_]]


def test_period_encoder_clips_to_training_range():
    encoder = PeriodEncoder(clip=True).fit(np.array([[24200.0], [24227.0]]))
    assert encoder.transform(np.array([[2030 * 12 + 1.0], [10.0]])).tolist() == [
        [24227.0],
        [24200.0],
    ]
    unclipped = PeriodEncoder(clip=False).fit(np.array([[1.0], [2.0]]))
    assert unclipped.transform(np.array([[5.0]])).tolist() == [[5.0]]


def test_invoice_format_features_flag_scientific_style():
    out = InvoiceFormatFeatures().fit_transform(
        np.array([["4.80P+11"], ["AB-12"], [None]], dtype=object)
    )
    assert out.shape == (3, 5)
    assert out[0, 4] == 1.0 and out[1, 2] == 1.0


def test_pipeline_output_is_sparse_and_finite(fitted):
    _, matrix = fitted
    assert sp.issparse(matrix) and matrix.format == "csr"
    assert np.isfinite(matrix.data).all()


def test_pipeline_transforms_holdout_with_unseen_values(fitted, settings):
    pipeline, matrix = fitted
    holdout, _, _ = process(settings.paths.holdout / "holdout_features.xlsx", "holdout.xlsx")
    out = pipeline.transform(holdout)
    assert out.shape == (590, matrix.shape[1])
    assert np.isfinite(out.data).all()


def test_pipeline_handles_missing_and_extreme_values(fitted, training):
    pipeline, matrix = fitted
    odd = training.frame.head(4).copy()
    odd.loc[0, "period"] = 2030 * 12 + 1
    odd.loc[1, "amount"] = np.nan
    odd.loc[2, "Col1"] = None
    odd.loc[3, "amount"] = 1e15
    out = pipeline.transform(odd)
    assert np.isfinite(out.data).all()
    names = list(pipeline.get_feature_names_out())
    period = names.index("period__period")
    assert out[0, period] == pytest.approx(matrix[:, period].max())


def test_every_feature_maps_to_exactly_one_group(fitted):
    pipeline, matrix = fitted
    groups = feature_groups(pipeline.get_feature_names_out())
    indices = sorted(i for members in groups.values() for i in members)
    assert indices == list(range(matrix.shape[1]))
    assert set(groups) == set(EXPLANATION_GROUPS)


def test_fit_uses_only_given_rows(training, settings):
    small = build_feature_pipeline(settings.feature).fit(training.frame.head(100))
    vocabulary = small.named_transformers_["col1_tokens"].named_steps["tfidf"].vocabulary_
    head_tokens = {t.lower() for v in training.frame.head(100)["Col1"] for t in v.split()}
    assert set(vocabulary) == head_tokens


def test_readable_feature_names(fitted):
    pipeline, _ = fitted
    readable = {readable_feature_name(n) for n in pipeline.get_feature_names_out()}
    assert "token:word563" in readable
    assert {"amount_signed_log", "is_negative", "is_round", "period", "doc:nodoc"} <= readable
    assert any(r.startswith("pair:") and " || " in r for r in readable)
    assert any(r.startswith("value:") for r in readable)


def test_ablation_subsets_and_additions(training, settings):
    subset = [n for n in DEFAULT_TRANSFORMERS if n != "col4_tokens"]
    reduced = build_feature_pipeline(settings.feature, subset).fit(training.frame)
    assert "Col4" not in feature_groups(reduced.get_feature_names_out())
    extended = build_feature_pipeline(settings.feature, [*DEFAULT_TRANSFORMERS, "col2_format"])
    extended.fit(training.frame)
    assert "Col2" in feature_groups(extended.get_feature_names_out())
    with pytest.raises(ConfigError, match="Unknown transformers"):
        build_feature_pipeline(settings.feature, ["nope"])


def test_pipeline_clones_cleanly(settings):
    assert clone(build_feature_pipeline(settings.feature)).get_params()
