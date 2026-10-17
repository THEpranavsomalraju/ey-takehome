"""Feature pipeline: the single source of feature logic for training and inference.

Every transformer fits inside the pipeline on training folds only, and the
same fitted pipeline transforms uploads, so training and inference can never
diverge. The output is one sparse CSR matrix shared by every model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ledgerclf.config import FeatureParams
from ledgerclf.exceptions import ConfigError
from ledgerclf.schema import (
    AMOUNT,
    COL_ACCOUNT,
    COL_AMOUNT,
    COL_DESCRIPTION,
    COL_DOC_TYPE,
    COL_INVOICE,
    COL_PERIOD,
    COL_VENDOR,
    MISSING_TOKEN,
    PERIOD,
)

GROUP_VENDOR = COL_VENDOR
GROUP_PAIR = f"{COL_VENDOR}+{COL_ACCOUNT}"
GROUP_AMOUNT = COL_AMOUNT
GROUP_DESCRIPTION = COL_DESCRIPTION
GROUP_PERIOD = COL_PERIOD
GROUP_ACCOUNT = COL_ACCOUNT
GROUP_DOC_TYPE = COL_DOC_TYPE
GROUP_INVOICE = COL_INVOICE
EXPLANATION_GROUPS: tuple[str, ...] = (
    GROUP_VENDOR,
    GROUP_PAIR,
    GROUP_AMOUNT,
    GROUP_DESCRIPTION,
    GROUP_PERIOD,
    GROUP_ACCOUNT,
    GROUP_DOC_TYPE,
)

# Transformer name -> (source group, readable feature prefix). The order is the
# column order of the feature matrix.
TRANSFORMERS: dict[str, tuple[str, str]] = {
    "col1_tokens": (GROUP_VENDOR, "token"),
    "col1_value": (GROUP_VENDOR, "value"),
    "col4_tokens": (GROUP_DESCRIPTION, "token"),
    "col6_tokens": (GROUP_ACCOUNT, "token"),
    "col6_value": (GROUP_ACCOUNT, "value"),
    "pair": (GROUP_PAIR, "pair"),
    "col7": (GROUP_DOC_TYPE, "doc"),
    "amount": (GROUP_AMOUNT, ""),
    "amount_flags": (GROUP_AMOUNT, ""),
    "period": (GROUP_PERIOD, ""),
    "col2_format": (GROUP_INVOICE, "invoice"),
}
DEFAULT_TRANSFORMERS: tuple[str, ...] = tuple(n for n in TRANSFORMERS if n != "col2_format")
# Name sklearn's OneHotEncoder gives the bucket of values seen fewer than min_frequency times.
INFREQUENT_CATEGORY = "infrequent_sklearn"
_TOKEN_COLUMNS = {
    "col1_tokens": COL_VENDOR,
    "col4_tokens": COL_DESCRIPTION,
    "col6_tokens": COL_ACCOUNT,
}
_VALUE_COLUMNS = {"col1_value": COL_VENDOR, "col6_value": COL_ACCOUNT}
_SCIENTIFIC = re.compile(r"^\d\.\d+[a-z][+-]\d+$")


def _as_frame(X: Any) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _width(X: Any) -> int:
    shape = np.shape(X)
    return int(shape[1]) if len(shape) > 1 else 1


class TextNormalizer(BaseEstimator, TransformerMixin):
    """Lowercase text and map missing or empty values to ``__missing__``.

    Accepts a Series (returns a 1-D array for vectorizers) or a DataFrame
    (returns a 2-D array for encoders). Stateless.
    """

    def fit(self, X: Any, y: Any = None) -> TextNormalizer:
        """Remember input feature names; nothing is learned."""
        self.feature_names_in_ = np.asarray(_as_frame(X).columns, dtype=object)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Return lowercased strings with ``__missing__`` for missing values."""
        if isinstance(X, pd.Series):
            return np.asarray(X.map(normalize_text).to_numpy(), dtype=object)
        frame = _as_frame(X)
        return np.column_stack([frame[c].map(normalize_text).to_numpy() for c in frame.columns])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the input column names unchanged."""
        return self.feature_names_in_


def normalize_text(value: object) -> str:
    """Return the model form of one text cell: lowercase, ``__missing__`` when empty."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return MISSING_TOKEN
    text = str(value).strip().lower()
    return text if text else MISSING_TOKEN


class ColumnConcat(BaseEstimator, TransformerMixin):
    """Join several text columns into one key, for the vendor-account pair."""

    def __init__(self, separator: str = FeatureParams().pair_separator) -> None:
        self.separator = separator

    def fit(self, X: Any, y: Any = None) -> ColumnConcat:
        """Record the input width; nothing else is learned."""
        self.n_features_in_ = _width(X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Return a single-column 2-D array of joined keys."""
        array = np.asarray(X, dtype=object)
        joined = [self.separator.join(str(v) for v in row) for row in array]
        return np.asarray(joined, dtype=object).reshape(-1, 1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the single output name."""
        return np.asarray(["pair"], dtype=object)


class TrainingMedianImputer(BaseEstimator, TransformerMixin):
    """Fill NaN with the median of the training fold.

    Lives inside the pipeline so that every CV fold imputes with its own
    training median and inference uses the final training median.
    """

    def fit(self, X: Any, y: Any = None) -> TrainingMedianImputer:
        """Learn one median per column (0.0 when a column has no values)."""
        array = np.asarray(X, dtype=float)
        with np.errstate(all="ignore"):
            medians = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in array.T])
        self.medians_ = medians
        self.feature_names_in_ = np.asarray(_as_frame(X).columns, dtype=object)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Return the input with NaN replaced by the training medians."""
        array = np.asarray(X, dtype=float).copy()
        missing = ~np.isfinite(array)
        array[missing] = np.take(self.medians_, np.nonzero(missing)[1])
        return array

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the input column names unchanged."""
        return self.feature_names_in_


def signed_log(values: Any) -> np.ndarray:
    """Return ``sign(x) * log1p(|x|)``: compresses large amounts and keeps credits negative."""
    array = np.asarray(values, dtype=float)
    return np.asarray(np.sign(array) * np.log1p(np.abs(array)))


class SignedLogAmount(BaseEstimator, TransformerMixin):
    """Compress heavy-tailed amounts with ``sign(x) * log1p(|x|)``, keeping credits negative."""

    def fit(self, X: Any, y: Any = None) -> SignedLogAmount:
        """Record the input width; nothing else is learned."""
        self.n_features_in_ = _width(X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Return the signed log of every value."""
        return signed_log(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the output feature name."""
        return np.asarray(["amount_signed_log"], dtype=object)


CENTS_PER_UNIT = 100
CENTS_EXACT_LIMIT = 2.0**53 / CENTS_PER_UNIT  # above this, float64 cannot hold every cent


class AmountFlags(BaseEstimator, TransformerMixin):
    """Binary flags: negative amount (credit) and round amount (multiple of the modulus).

    Round amounts recur in rare classes (1,680 and 1,500 in Category_3 and
    Category_4). Rounding to cents first avoids float noise; float arithmetic
    keeps huge amounts exact, and amounts that round to zero cents are not round.
    """

    def __init__(self, modulus: float = 10.0) -> None:
        self.modulus = modulus

    def fit(self, X: Any, y: Any = None) -> AmountFlags:
        """Record the input width; nothing else is learned."""
        self.n_features_in_ = _width(X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Return ``[is_negative, is_round]`` for every row."""
        amount = np.asarray(X, dtype=float)[:, 0]
        magnitude = np.abs(amount)
        exact = magnitude < CENTS_EXACT_LIMIT
        with np.errstate(over="ignore", invalid="ignore"):
            cents = np.round(np.where(exact, magnitude, 0.0) * CENTS_PER_UNIT)
            in_cents = (np.fmod(cents, round(self.modulus * CENTS_PER_UNIT)) == 0) & (cents != 0)
            in_units = np.fmod(magnitude, self.modulus) == 0
        is_round = np.where(exact, in_cents, in_units)
        return np.column_stack([amount < 0, is_round]).astype(float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the two flag names."""
        return np.asarray(["is_negative", "is_round"], dtype=object)


class PeriodEncoder(BaseEstimator, TransformerMixin):
    """Period index (``year * 12 + month``) clipped to the training range.

    Clipping stops a linear model from extrapolating to dates it never saw;
    the ``DATE_OUT_OF_RANGE`` flag tells the user when clipping happened.
    """

    def __init__(self, clip: bool = True) -> None:
        self.clip = clip

    def fit(self, X: Any, y: Any = None) -> PeriodEncoder:
        """Learn the training minimum and maximum period."""
        array = np.asarray(X, dtype=float)
        self.min_ = float(np.nanmin(array))
        self.max_ = float(np.nanmax(array))
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Return the (clipped) period index."""
        array = np.asarray(X, dtype=float)
        return np.clip(array, self.min_, self.max_) if self.clip else array

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the output feature name."""
        return np.asarray(["period"], dtype=object)


class InvoiceFormatFeatures(BaseEstimator, TransformerMixin):
    """Col2 format features, used only by the ablation that confirms dropping Col2."""

    def fit(self, X: Any, y: Any = None) -> InvoiceFormatFeatures:
        """Record the input width; nothing else is learned."""
        self.n_features_in_ = _width(X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Return length, letter count, hyphen flag, digit share, scientific-style flag."""
        values = [normalize_text(v) for v in np.asarray(X, dtype=object)[:, 0]]
        rows = []
        for text in values:
            length = max(len(text), 1)
            rows.append(
                [
                    np.log1p(len(text)),
                    sum(c.isalpha() for c in text) / length,
                    float("-" in text),
                    sum(c.isdigit() for c in text) / length,
                    float(bool(_SCIENTIFIC.match(text))),
                ]
            )
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the five feature names."""
        names = ["log_length", "letter_share", "has_hyphen", "digit_share", "scientific_style"]
        return np.asarray(names, dtype=object)


def _tokens_pipeline(params: FeatureParams) -> Pipeline:
    vectorizer = TfidfVectorizer(
        token_pattern=params.token_pattern,
        lowercase=False,
        ngram_range=params.ngram_range,
        sublinear_tf=params.sublinear_tf,
        min_df=params.min_df,
        dtype=np.float64,
    )
    return Pipeline([("normalize", TextNormalizer()), ("tfidf", vectorizer)])


def _one_hot(min_frequency: int | None) -> OneHotEncoder:
    if min_frequency is None:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=min_frequency, sparse_output=True
    )


def _transformer(name: str, params: FeatureParams) -> tuple[str, Any, str | list[str]]:
    if name in _TOKEN_COLUMNS:
        return name, _tokens_pipeline(params), _TOKEN_COLUMNS[name]
    if name in _VALUE_COLUMNS:
        steps = [
            ("normalize", TextNormalizer()),
            ("onehot", _one_hot(params.whole_value_min_frequency)),
        ]
        return name, Pipeline(steps), [_VALUE_COLUMNS[name]]
    builders: dict[str, tuple[list[tuple[str, Any]], list[str]]] = {
        "pair": (
            [
                ("normalize", TextNormalizer()),
                ("concat", ColumnConcat(params.pair_separator)),
                ("onehot", _one_hot(params.pair_min_frequency)),
            ],
            [COL_VENDOR, COL_ACCOUNT],
        ),
        "col7": ([("normalize", TextNormalizer()), ("onehot", _one_hot(None))], [COL_DOC_TYPE]),
        "amount": (
            [
                ("impute", TrainingMedianImputer()),
                ("signed_log", SignedLogAmount()),
                ("scale", StandardScaler()),
            ],
            [AMOUNT],
        ),
        "amount_flags": (
            [
                ("impute", TrainingMedianImputer()),
                ("flags", AmountFlags(params.round_amount_modulus)),
            ],
            [AMOUNT],
        ),
        "period": (
            [
                ("impute", TrainingMedianImputer()),
                ("encode", PeriodEncoder(clip=True)),
                ("scale", StandardScaler()),
            ],
            [PERIOD],
        ),
        "col2_format": ([("format", InvoiceFormatFeatures())], [COL_INVOICE]),
    }
    steps, columns = builders[name]
    return name, Pipeline(steps), columns


def build_feature_pipeline(
    params: FeatureParams, transformers: Sequence[str] = DEFAULT_TRANSFORMERS
) -> ColumnTransformer:
    """Build the feature ColumnTransformer.

    Args:
        params: Feature parameters from config (or the tuned champion).
        transformers: Transformer names to include, in matrix order. The
            default set is the production feature set; ablations pass subsets
            or add ``col2_format``.

    Returns:
        An unfitted ColumnTransformer producing one sparse CSR matrix.
    """
    unknown = sorted(set(transformers) - set(TRANSFORMERS))
    if unknown:
        raise ConfigError(
            f"Unknown transformers: {', '.join(unknown)}. Use names from features.TRANSFORMERS.",
            code="UNKNOWN_TRANSFORMER",
        )
    ordered = [name for name in TRANSFORMERS if name in transformers]
    return ColumnTransformer(
        [_transformer(name, params) for name in ordered],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def feature_groups(feature_names: Sequence[str]) -> dict[str, list[int]]:
    """Map each source group to the indices of its output features.

    Args:
        feature_names: ``get_feature_names_out()`` of the fitted pipeline.

    Returns:
        Dict from group name (Col1, Col1+Col6, Col3, Col4, Col5, Col6, Col7,
        and Col2 when present) to feature indices. Every feature belongs to
        exactly one group.
    """
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(feature_names):
        transformer = str(name).split("__", 1)[0]
        group = TRANSFORMERS[transformer][0]
        groups.setdefault(group, []).append(index)
    return groups


def readable_feature_name(feature_name: str) -> str:
    """Turn a pipeline feature name into a short readable name for explanations.

    Examples: ``col6_tokens__word575`` to ``token:word575``;
    ``pair__pair_a || b`` to ``pair:a || b``; ``amount__amount_signed_log``
    to ``amount_signed_log``.
    """
    transformer, _, rest = str(feature_name).partition("__")
    prefix = TRANSFORMERS[transformer][1]
    if transformer in (*_VALUE_COLUMNS, "pair", "col7"):
        # One-hot names carry the input column as a prefix: "Col1_word16 word17".
        rest = rest.split("_", 1)[1] if "_" in rest else rest
        if rest == INFREQUENT_CATEGORY:
            rest = "(rare or unseen value)"
    return f"{prefix}:{rest}" if prefix else rest
