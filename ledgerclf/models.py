"""Model factories, class-weight wrapper, log-odds blend with exact contributions, baselines.

Every factory returns an unfitted estimator that takes the coerced frame
(``schema.coerce_values`` output) and contains its own feature pipeline, so
cross-validation refits every transformer on each training fold.

LightGBM is imported only inside functions, so importing this module works on
machines where LightGBM fails to load and the fallback model still loads.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.metaestimators import available_if

from ledgerclf.config import (
    BlendParams,
    ExplainParams,
    FeatureParams,
    LinearParams,
    TreeParams,
)
from ledgerclf.exceptions import ConfigError
from ledgerclf.features import DEFAULT_TRANSFORMERS, build_feature_pipeline, normalize_text
from ledgerclf.labels import CANONICAL_CLASSES, MAJORITY_CLASS
from ledgerclf.schema import COL_ACCOUNT, COL_VENDOR

ABSENT_CLASS_LOGIT = ExplainParams().absent_class_logit
TEXT_AND_DOC_TRANSFORMERS = (
    "col1_tokens",
    "col1_value",
    "col4_tokens",
    "col6_tokens",
    "col6_value",
    "pair",
    "col7",
)


@contextmanager
def quiet_feature_name_check() -> Iterator[None]:
    """Silence sklearn's feature-name warning on LightGBM predictions.

    LightGBM 4.6 reports generic feature names (``Column_0`` ...) after fitting
    on a sparse matrix, so sklearn warns on every prediction from a sparse
    matrix. The matrix always comes from the same fitted pipeline, so the
    warning carries no information. ``catch_warnings`` is not thread-safe, so
    the blend's prediction and contribution paths call the LightGBM booster
    directly (no sklearn name check); only benchmark wrappers use this.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        yield


def class_weights(y: np.ndarray, power: float) -> dict[str, float]:
    """Return ``(n_max / n_c) ** power`` for every class present in ``y``.

    Power 0 is unweighted; power 1 matches sklearn ``balanced`` up to scale.

    Args:
        y: Training labels of the current fit.
        power: Class weight power.

    Returns:
        Dict from class label to weight, keys sorted.
    """
    labels, counts = np.unique(np.asarray(y, dtype=object), return_counts=True)
    largest = counts.max()
    return {
        str(label): float((largest / count) ** power)
        for label, count in zip(labels, counts, strict=True)
    }


class PoweredClassWeight(BaseEstimator, ClassifierMixin):
    """Wrap a classifier and set ``class_weight`` from the labels of each fit.

    Computing the weights inside ``fit`` means every CV fold weights classes
    by its own training labels.

    Args:
        estimator: A classifier with a ``class_weight`` parameter.
        power: Class weight power (0 = unweighted).
    """

    def __init__(self, estimator: Any = None, power: float = 0.0) -> None:
        self.estimator = estimator
        self.power = power

    def fit(self, X: Any, y: Any) -> PoweredClassWeight:
        """Fit a clone of the estimator with powered class weights."""
        y_array = np.asarray(y, dtype=object)
        weights = None if self.power == 0 else class_weights(y_array, self.power)
        self.estimator_ = clone(self.estimator).set_params(class_weight=weights)
        self.estimator_.fit(X, y_array)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict labels with the fitted estimator."""
        with quiet_feature_name_check():
            return np.asarray(self.estimator_.predict(X), dtype=object)

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probabilities in ``classes_`` order."""
        with quiet_feature_name_check():
            return np.asarray(self.estimator_.predict_proba(X))

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X: Any) -> np.ndarray:
        """Return raw scores in ``classes_`` order."""
        return np.asarray(self.estimator_.decision_function(X))


def to_canonical(values: np.ndarray, classes: np.ndarray, fill: float) -> np.ndarray:
    """Place per-class columns in canonical class order, filling absent classes.

    A fold without Category_5 trains a model with five classes; every
    probability or logit matrix still gets all six canonical columns.

    Args:
        values: Matrix with one column per entry of ``classes``.
        classes: Class labels of the columns.
        fill: Value for canonical classes missing from ``classes``.

    Returns:
        Matrix of shape (n_rows, 6) in canonical order.
    """
    out = np.full((values.shape[0], len(CANONICAL_CLASSES)), fill, dtype=float)
    index = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
    for column, cls in enumerate(classes):
        out[:, index[str(cls)]] = values[:, column]
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    """Return row-wise softmax, stable for very negative absent-class logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum(axis=1, keepdims=True))


@dataclass(frozen=True)
class Contributions:
    """Exact additive decomposition of blended logits.

    ``values[i, k, j]`` is feature j's contribution to class k's logit for row
    i, and ``bias[i, k]`` the intercept part, so that
    ``values[i, k].sum() + bias[i, k] == logits[i, k]``.
    """

    values: np.ndarray
    bias: np.ndarray
    logits: np.ndarray


class LogOddsBlendClassifier(BaseEstimator, ClassifierMixin):
    """Blend a linear model and LightGBM in logit space.

    ``z = (w * z_linear + (1 - w) * z_tree) / T`` and ``p = softmax(z)``.
    Averaging logits instead of probabilities keeps every prediction an exact
    sum of per-feature contributions, which the explanation agent reports.
    With ``tree=None`` the model is a temperature-scaled logistic regression:
    the fallback when LightGBM cannot load, with the same explanation path
    and no LightGBM object in its pickle.

    Args:
        linear: Unfitted ``PoweredClassWeight(LogisticRegression)``.
        tree: Unfitted ``PoweredClassWeight(LGBMClassifier)``, or None.
        linear_weight: Weight ``w`` of the linear logits.
        temperature: Calibration temperature ``T`` (1.0 until calibrated).
    """

    def __init__(
        self,
        linear: Any = None,
        tree: Any = None,
        linear_weight: float = 0.5,
        temperature: float = 1.0,
    ) -> None:
        self.linear = linear
        self.tree = tree
        self.linear_weight = linear_weight
        self.temperature = temperature

    def fit(self, X: Any, y: Any) -> LogOddsBlendClassifier:
        """Fit both members on the same feature matrix."""
        y_array = np.asarray(y, dtype=object)
        self.linear_ = clone(self.linear).fit(X, y_array)
        self.tree_ = None if self.tree is None else clone(self.tree).fit(X, y_array)
        if self.tree_ is not None and [str(c) for c in self.linear_.classes_] != [
            str(c) for c in self.tree_.classes_
        ]:
            raise ConfigError(
                "Blend members learned different class orders; refit the model with: "
                "python -m ledgerclf train",
                code="BLEND_CLASS_ORDER",
            )
        self.classes_ = np.asarray(CANONICAL_CLASSES, dtype=object)
        return self

    @property
    def effective_weight(self) -> float:
        """Return the linear weight actually used (1.0 when there is no tree member)."""
        return 1.0 if self.tree is None else float(self.linear_weight)

    def _member_logits(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        classes = self.linear_.classes_
        linear = to_canonical(self.linear_.decision_function(X), classes, ABSENT_CLASS_LOGIT)
        if self.tree_ is None:
            return linear, np.zeros_like(linear)
        raw = self.tree_.estimator_.booster_.predict(X, raw_score=True)
        tree = to_canonical(np.asarray(raw), classes, ABSENT_CLASS_LOGIT)
        return linear, tree

    def decision_function(self, X: Any) -> np.ndarray:
        """Return blended logits divided by the temperature, canonical class order."""
        linear, tree = self._member_logits(X)
        weight = self.effective_weight
        return (weight * linear + (1.0 - weight) * tree) / self.temperature

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return calibrated probabilities for all six canonical classes."""
        return softmax(self.decision_function(X))

    def predict(self, X: Any) -> np.ndarray:
        """Return the most likely canonical class per row."""
        return np.asarray(self.classes_[np.argmax(self.decision_function(X), axis=1)], dtype=object)

    def contributions(self, X: Any) -> Contributions:
        """Decompose blended logits into exact per-feature contributions.

        Linear part: ``W[k, j] * x_j`` with bias ``b[k]``. Tree part: LightGBM
        SHAP values (``pred_contrib``) with their expected-value column. Both
        are mixed with the blend weight and divided by the temperature.

        Args:
            X: Feature matrix (output of the fitted feature pipeline).

        Returns:
            Contributions with ``values`` (rows, 6, features), ``bias`` (rows, 6)
            and ``logits`` (rows, 6); absent classes carry zero contributions and
            the absent-class logit as bias.
        """
        matrix = sp.csr_matrix(X)
        linear_values, linear_bias = _linear_contributions(self.linear_.estimator_, matrix)
        if self.tree_ is None:
            tree_values, tree_bias = np.zeros_like(linear_values), np.zeros_like(linear_bias)
        else:
            tree_values, tree_bias = _tree_contributions(self.tree_.estimator_, matrix)
        weight, temperature = self.effective_weight, self.temperature
        values = (weight * linear_values + (1 - weight) * tree_values) / temperature
        bias = (weight * linear_bias + (1 - weight) * tree_bias) / temperature
        return Contributions(values=values, bias=bias, logits=self.decision_function(matrix))


def _expand(
    values: np.ndarray, bias: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map per-class contribution arrays to canonical order (absent classes: zero, absent bias)."""
    rows, _, features = values.shape
    full_values = np.zeros((rows, len(CANONICAL_CLASSES), features))
    full_bias = np.full((rows, len(CANONICAL_CLASSES)), ABSENT_CLASS_LOGIT)
    index = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
    for position, cls in enumerate(classes):
        full_values[:, index[str(cls)], :] = values[:, position, :]
        full_bias[:, index[str(cls)]] = bias[:, position]
    return full_values, full_bias


def _linear_contributions(
    model: LogisticRegression, matrix: sp.csr_matrix
) -> tuple[np.ndarray, np.ndarray]:
    dense = matrix.toarray()
    values = dense[:, np.newaxis, :] * model.coef_[np.newaxis, :, :]
    bias = np.repeat(model.intercept_[np.newaxis, :], dense.shape[0], axis=0)
    return _expand(values, bias, model.classes_)


def _tree_contributions(model: Any, matrix: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    raw = model.booster_.predict(matrix, pred_contrib=True)
    classes, features = model.classes_, matrix.shape[1]
    if isinstance(raw, list):
        per_class = [np.asarray(sp.csr_matrix(block).toarray()) for block in raw]
        stacked = np.stack(per_class, axis=1)
    else:
        stacked = np.asarray(raw).reshape(matrix.shape[0], len(classes), features + 1)
    return _expand(stacked[:, :, :features], stacked[:, :, features], classes)


class ProbabilityBlendClassifier(BaseEstimator, ClassifierMixin):
    """Average member probabilities (comparison only; not additive in logits)."""

    def __init__(self, linear: Any = None, tree: Any = None) -> None:
        self.linear = linear
        self.tree = tree

    def fit(self, X: Any, y: Any) -> ProbabilityBlendClassifier:
        """Fit both members on the same feature matrix."""
        y_array = np.asarray(y, dtype=object)
        self.linear_ = clone(self.linear).fit(X, y_array)
        self.tree_ = clone(self.tree).fit(X, y_array)
        self.classes_ = np.asarray(CANONICAL_CLASSES, dtype=object)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return the mean of the two members' canonical probabilities."""
        linear = to_canonical(self.linear_.predict_proba(X), self.linear_.classes_, 0.0)
        tree = to_canonical(self.tree_.predict_proba(X), self.tree_.classes_, 0.0)
        return (linear + tree) / 2

    def predict(self, X: Any) -> np.ndarray:
        """Return the class with the highest averaged probability."""
        return np.asarray(self.classes_[np.argmax(self.predict_proba(X), axis=1)], dtype=object)


class VendorAccountLookup(BaseEstimator, ClassifierMixin):
    """Baseline: majority label per (vendor, account), then per vendor, then global."""

    def fit(self, X: pd.DataFrame, y: Any) -> VendorAccountLookup:
        """Learn majority labels per pair and per vendor from the training rows."""
        frame = self._keys(X).assign(label=np.asarray(y, dtype=object))
        self.pair_ = self._majority(frame, ["vendor", "account"])
        self.vendor_ = self._majority(frame, ["vendor"])
        self.global_ = str(pd.Series(frame["label"]).value_counts().idxmax())
        self.classes_ = np.asarray(sorted(set(frame["label"])), dtype=object)
        return self

    @staticmethod
    def _keys(X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "vendor": X[COL_VENDOR].map(normalize_text).to_numpy(),
                "account": X[COL_ACCOUNT].map(normalize_text).to_numpy(),
            }
        )

    @staticmethod
    def _majority(frame: pd.DataFrame, keys: list[str]) -> dict[tuple[str, ...], str]:
        counts = frame.groupby([*keys, "label"]).size().reset_index(name="n")
        counts = counts.sort_values(
            [*keys, "n", "label"], ascending=[True] * len(keys) + [False, True]
        )
        best = counts.drop_duplicates(keys)
        return {tuple(row[k] for k in keys): row["label"] for _, row in best.iterrows()}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with the most specific known key."""
        keys = self._keys(X)
        labels = [
            self.pair_.get((v, a), self.vendor_.get((v,), self.global_))
            for v, a in zip(keys["vendor"], keys["account"], strict=True)
        ]
        return np.asarray(labels, dtype=object)


def make_logistic(linear: LinearParams, features: FeatureParams | None = None) -> Pipeline:
    """Return features plus a powered-weight multinomial logistic regression."""
    model = LogisticRegression(C=linear.C, max_iter=linear.max_iter, solver=linear.solver)
    return Pipeline(
        [
            ("features", build_feature_pipeline(features or FeatureParams())),
            ("model", PoweredClassWeight(model, linear.class_weight_power)),
        ]
    )


def lightgbm_classifier(tree: TreeParams, random_state: int) -> Any:
    """Return an unfitted, deterministic, single-threaded LGBMClassifier (lazy import)."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="multiclass",
        n_estimators=tree.n_estimators,
        learning_rate=tree.learning_rate,
        num_leaves=tree.num_leaves,
        max_depth=tree.max_depth,
        min_child_samples=tree.min_child_samples,
        reg_lambda=tree.reg_lambda,
        colsample_bytree=tree.colsample_bytree,
        subsample=tree.subsample,
        subsample_freq=tree.subsample_freq,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        random_state=random_state,
        verbose=-1,
    )


def make_lightgbm(
    tree: TreeParams, random_state: int, features: FeatureParams | None = None
) -> Pipeline:
    """Return features plus a powered-weight LightGBM classifier."""
    return Pipeline(
        [
            ("features", build_feature_pipeline(features or FeatureParams())),
            (
                "model",
                PoweredClassWeight(
                    lightgbm_classifier(tree, random_state), tree.class_weight_power
                ),
            ),
        ]
    )


def make_blend(
    linear: LinearParams,
    tree: TreeParams,
    blend: BlendParams,
    random_state: int,
    features: FeatureParams | None = None,
) -> Pipeline:
    """Return features plus the log-odds blend of logistic regression and LightGBM."""
    model = LogOddsBlendClassifier(
        linear=PoweredClassWeight(
            LogisticRegression(C=linear.C, max_iter=linear.max_iter, solver=linear.solver),
            linear.class_weight_power,
        ),
        tree=PoweredClassWeight(lightgbm_classifier(tree, random_state), tree.class_weight_power),
        linear_weight=blend.linear_weight,
        temperature=blend.temperature,
    )
    return Pipeline(
        [("features", build_feature_pipeline(features or FeatureParams())), ("model", model)]
    )


def make_calibrated_logistic(
    linear: LinearParams, temperature: float, features: FeatureParams | None = None
) -> Pipeline:
    """Return features plus a temperature-scaled logistic regression (the fallback model)."""
    model = LogOddsBlendClassifier(
        linear=PoweredClassWeight(
            LogisticRegression(C=linear.C, max_iter=linear.max_iter, solver=linear.solver),
            linear.class_weight_power,
        ),
        tree=None,
        linear_weight=1.0,
        temperature=temperature,
    )
    return Pipeline(
        [("features", build_feature_pipeline(features or FeatureParams())), ("model", model)]
    )


def make_probability_blend(linear: LinearParams, tree: TreeParams, random_state: int) -> Pipeline:
    """Return features plus the probability-averaging blend (comparison only)."""
    model = ProbabilityBlendClassifier(
        linear=PoweredClassWeight(
            LogisticRegression(C=linear.C, max_iter=linear.max_iter, solver=linear.solver),
            linear.class_weight_power,
        ),
        tree=PoweredClassWeight(lightgbm_classifier(tree, random_state), tree.class_weight_power),
    )
    return Pipeline([("features", build_feature_pipeline(FeatureParams())), ("model", model)])


def make_dummy() -> DummyClassifier:
    """Return the majority-class baseline (always Category_1)."""
    return DummyClassifier(strategy="constant", constant=MAJORITY_CLASS)


def make_lookup() -> VendorAccountLookup:
    """Return the vendor-account majority lookup baseline."""
    return VendorAccountLookup()


def make_complement_nb(alpha: float) -> Pipeline:
    """Return Complement Naive Bayes on text and document-flag features only."""
    features = build_feature_pipeline(FeatureParams(), TEXT_AND_DOC_TRANSFORMERS)
    return Pipeline([("features", features), ("model", ComplementNB(alpha=alpha))])


def make_linear_svc(C: float, power: float, random_state: int, max_iter: int) -> Pipeline:
    """Return a linear SVM (predictions from ``decision_function``; no probabilities)."""
    model = PoweredClassWeight(LinearSVC(C=C, random_state=random_state, max_iter=max_iter), power)
    return Pipeline([("features", build_feature_pipeline(FeatureParams())), ("model", model)])


def logistic_with_features(
    linear: LinearParams,
    transformers: tuple[str, ...] = DEFAULT_TRANSFORMERS,
    features: FeatureParams | None = None,
) -> Pipeline:
    """Return logistic regression on a chosen feature subset (ablations)."""
    model = LogisticRegression(C=linear.C, max_iter=linear.max_iter, solver=linear.solver)
    return Pipeline(
        [
            ("features", build_feature_pipeline(features or FeatureParams(), transformers)),
            ("model", PoweredClassWeight(model, linear.class_weight_power)),
        ]
    )
