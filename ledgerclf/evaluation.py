"""Metrics, the grouped CV runner with out-of-fold capture, and the validation schemes.

Every scheme runs on training rows only. Each fold refits the whole model
pipeline (features included) on its training rows, so no statistic from a
test fold reaches the model. Every CV is grouped, so duplicate rows never
straddle a fold boundary.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from ledgerclf.calibration import brier_score, expected_calibration_error
from ledgerclf.config import CalibrationParams
from ledgerclf.labels import CANONICAL_CLASSES, FIVE_CLASSES, MAJORITY_CLASS, RARE_CLASSES
from ledgerclf.models import ABSENT_CLASS_LOGIT, to_canonical
from ledgerclf.reference import pair_key
from ledgerclf.schema import COL_ACCOUNT, COL_VENDOR
from ledgerclf.splitting import grouped_cv, random_cv, small_class_warning_suppressed

logger = logging.getLogger(__name__)

_PROBABILITY_FLOOR = CalibrationParams().probability_floor


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None
) -> dict[str, float]:
    """Compute the classification and probability metrics.

    Args:
        y_true: True canonical labels.
        y_pred: Predicted canonical labels.
        proba: Optional probabilities in canonical order (adds log loss, Brier, ECE).

    Returns:
        Dict with accuracy, macro_f1_5, macro_f1_6, balanced_accuracy, per-class
        recall, and the probability metrics when ``proba`` is given.
    """
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # classes absent from a fold trigger sklearn warnings
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1_5": f1_score(
                y_true, y_pred, labels=list(FIVE_CLASSES), average="macro", zero_division=0
            ),
            "macro_f1_6": f1_score(
                y_true, y_pred, labels=list(CANONICAL_CLASSES), average="macro", zero_division=0
            ),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }
        _, recall, _, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(CANONICAL_CLASSES), zero_division=0
        )
    for cls, value, count in zip(CANONICAL_CLASSES, recall, support, strict=True):
        metrics[f"recall_{cls}"] = float(value) if count else np.nan
    if proba is not None:
        metrics.update(probability_metrics(proba, y_true))
    return {k: float(v) for k, v in metrics.items()}


def probability_metrics(proba: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Return log loss (all six classes), Brier score, and ECE."""
    clipped = np.clip(proba, _PROBABILITY_FLOOR, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return {
        "log_loss": log_loss(y_true, clipped, labels=list(CANONICAL_CLASSES)),
        "brier": brier_score(proba, y_true),
        "ece": expected_calibration_error(proba, y_true, CalibrationParams().n_bins),
    }


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return precision, recall, F1, and support per canonical class."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(CANONICAL_CLASSES), zero_division=0
        )
    return pd.DataFrame(
        {
            "class": CANONICAL_CLASSES,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def confusion_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return the confusion matrix (rows true, columns predicted) in canonical order."""
    table = pd.crosstab(
        pd.Categorical(y_true, categories=CANONICAL_CLASSES),
        pd.Categorical(y_pred, categories=CANONICAL_CLASSES),
        dropna=False,
    )
    table.index.name, table.columns.name = "true_label", None
    return table.reset_index()


@dataclass(frozen=True)
class Split:
    """One train/test split of training-row positions."""

    seed: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def grouped_splits(
    y: np.ndarray, groups: np.ndarray, seeds: Sequence[int], n_splits: int
) -> list[Split]:
    """Return primary grouped, stratified folds for every seed."""
    placeholder = np.zeros(len(y))
    splits: list[Split] = []
    for seed in seeds:
        with small_class_warning_suppressed():
            folds = list(grouped_cv(seed, n_splits).split(placeholder, y, groups))
        splits.extend(Split(seed, f, tr, te) for f, (tr, te) in enumerate(folds))
    return splits


def random_splits(y: np.ndarray, seed: int, n_splits: int) -> list[Split]:
    """Return ungrouped stratified folds (leakage demonstration only)."""
    with small_class_warning_suppressed():
        folds = list(random_cv(seed, n_splits).split(np.zeros(len(y)), y))
    return [Split(seed, f, tr, te) for f, (tr, te) in enumerate(folds)]


@dataclass
class FoldOutput:
    """Predictions of one fitted fold."""

    split: Split
    pred: np.ndarray
    proba: np.ndarray | None
    logits: np.ndarray | None
    fit_seconds: float


@dataclass
class CVResult:
    """Cross-validation result with out-of-fold predictions per seed.

    Accuracy mean and std come from the individual folds (3 seeds x 5 folds,
    15 folds in all). The other metrics are computed on
    the pooled out-of-fold predictions of each seed, then averaged over
    seeds, because rare classes are too thin inside a single fold.
    """

    model_name: str
    per_fold: pd.DataFrame
    per_seed: pd.DataFrame
    summary: dict[str, float]
    oof_pred: dict[int, np.ndarray] = field(default_factory=dict)
    oof_proba: dict[int, np.ndarray] = field(default_factory=dict)
    oof_logits: dict[int, np.ndarray] = field(default_factory=dict)
    fold_of_row: dict[int, np.ndarray] = field(default_factory=dict)


def _fit_fold(estimator: Any, frame: pd.DataFrame, y: np.ndarray, split: Split) -> FoldOutput:
    model = clone(estimator)
    started = time.perf_counter()
    model.fit(frame.iloc[split.train_idx], y[split.train_idx])
    seconds = time.perf_counter() - started
    test = frame.iloc[split.test_idx]
    pred = np.asarray(model.predict(test), dtype=object)
    proba = logits = None
    if hasattr(model, "predict_proba"):
        proba = to_canonical(np.asarray(model.predict_proba(test)), model.classes_, 0.0)
    if hasattr(model, "decision_function") and proba is not None:
        logits = to_canonical(
            np.asarray(model.decision_function(test)), model.classes_, ABSENT_CLASS_LOGIT
        )
    return FoldOutput(split, pred, proba, logits, seconds)


def run_cv(
    name: str,
    estimator: Any,
    frame: pd.DataFrame,
    y: np.ndarray,
    splits: Sequence[Split],
    n_jobs: int = 1,
) -> CVResult:
    """Fit and score an estimator on every split, capturing out-of-fold outputs.

    Args:
        name: Model name for reports.
        estimator: Unfitted estimator taking the coerced frame.
        frame: Training rows (coerced frame).
        y: Training labels.
        splits: Splits over row positions of ``frame``.
        n_jobs: Parallel processes over folds (each model stays single-threaded).

    Returns:
        The CV result.
    """
    outputs = Parallel(n_jobs=n_jobs)(delayed(_fit_fold)(estimator, frame, y, s) for s in splits)
    fold_rows = [
        {"seed": o.split.seed, "fold": o.split.fold, "fit_seconds": o.fit_seconds}
        | compute_metrics(y[o.split.test_idx], o.pred)
        for o in outputs
    ]
    result = CVResult(name, pd.DataFrame(fold_rows), pd.DataFrame(), {})
    _pool_out_of_fold(result, outputs, len(y))
    seed_rows = [
        {"seed": seed} | compute_metrics(y, result.oof_pred[seed], result.oof_proba.get(seed))
        for seed in sorted(result.oof_pred)
    ]
    result.per_seed = pd.DataFrame(seed_rows)
    result.summary = _summarize(result.per_fold, result.per_seed)
    return result


def _pool_out_of_fold(result: CVResult, outputs: list[FoldOutput], n_rows: int) -> None:
    for output in outputs:
        seed, test = output.split.seed, output.split.test_idx
        result.oof_pred.setdefault(seed, np.empty(n_rows, dtype=object))[test] = output.pred
        result.fold_of_row.setdefault(seed, np.full(n_rows, -1))[test] = output.split.fold
        if output.proba is not None:
            result.oof_proba.setdefault(seed, np.zeros((n_rows, len(CANONICAL_CLASSES))))[test] = (
                output.proba
            )
        if output.logits is not None:
            result.oof_logits.setdefault(seed, np.zeros((n_rows, len(CANONICAL_CLASSES))))[test] = (
                output.logits
            )


def _summarize(per_fold: pd.DataFrame, per_seed: pd.DataFrame) -> dict[str, float]:
    summary = {
        "accuracy_mean": float(per_fold["accuracy"].mean()),
        "accuracy_std": float(per_fold["accuracy"].std(ddof=1)) if len(per_fold) > 1 else 0.0,
        "fit_seconds": float(per_fold["fit_seconds"].mean()),
        "folds": len(per_fold),
    }
    for column in per_seed.columns:
        if column == "seed":
            continue
        summary[f"{column}_seed_mean"] = float(per_seed[column].mean())
        summary[f"{column}_seed_std"] = (
            float(per_seed[column].std(ddof=1)) if len(per_seed) > 1 else 0.0
        )
    return summary


@dataclass
class HoldOutResult:
    """Single train/test evaluation (out-of-time) with predictions."""

    metrics: dict[str, float]
    pred: np.ndarray
    proba: np.ndarray | None
    rare_alarms: int
    train_rows: int
    test_rows: int


def run_train_test(
    estimator: Any, frame: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray
) -> HoldOutResult:
    """Fit on the masked rows, score on the rest (out-of-time scheme).

    Args:
        estimator: Unfitted estimator.
        frame: Training rows.
        y: Labels.
        train_mask: True for rows used to fit.

    Returns:
        Metrics, predictions, and the count of Category_1 rows predicted as a
        rare class (false rare-class alarms).
    """
    split = Split(0, 0, np.flatnonzero(train_mask), np.flatnonzero(~train_mask))
    output = _fit_fold(estimator, frame, y, split)
    y_test = y[split.test_idx]
    alarms = int(((y_test == MAJORITY_CLASS) & np.isin(output.pred, RARE_CLASSES)).sum())
    return HoldOutResult(
        compute_metrics(y_test, output.pred, output.proba),
        output.pred,
        output.proba,
        alarms,
        len(split.train_idx),
        len(split.test_idx),
    )


def pair_seen_mask(frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    """Return True for test rows whose vendor-account pair appears in the training rows."""
    pairs = pd.Series(
        [pair_key(v, a) for v, a in zip(frame[COL_VENDOR], frame[COL_ACCOUNT], strict=True)]
    )
    known = set(pairs.iloc[train_idx])
    return np.asarray(pairs.iloc[test_idx].isin(known))


def accuracy_by_pair_seen(
    result: CVResult, frame: pd.DataFrame, y: np.ndarray, splits: Sequence[Split], seed: int
) -> pd.DataFrame:
    """Return out-of-fold accuracy split by seen versus unseen vendor-account pair."""
    seen = np.zeros(len(y), dtype=bool)
    for split in (s for s in splits if s.seed == seed):
        seen[split.test_idx] = pair_seen_mask(frame, split.train_idx, split.test_idx)
    correct = result.oof_pred[seed] == y
    rows = []
    for label, mask in (("seen pair", seen), ("unseen pair", ~seen)):
        rows.append(
            {
                "model": result.model_name,
                "pair": label,
                "rows": int(mask.sum()),
                "accuracy": float(correct[mask].mean()),
            }
        )
    return pd.DataFrame(rows)
