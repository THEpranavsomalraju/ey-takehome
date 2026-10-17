"""Probability calibration: temperature scaling, ECE, Brier, reliability and threshold tables.

Temperature scaling (finishing item D) divides the blended logits by one
scalar T fitted on out-of-fold logits. It keeps the argmax (accuracy is
unchanged) and keeps logits additive, so explanations stay exact; isotonic
or sigmoid calibration would break that (docs/04 section 9).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ledgerclf.config import CalibrationParams
from ledgerclf.labels import CANONICAL_CLASSES
from ledgerclf.models import softmax

_PROBABILITY_FLOOR = CalibrationParams().probability_floor


def one_hot(y: np.ndarray, classes: Sequence[str] = CANONICAL_CLASSES) -> np.ndarray:
    """Return a one-hot matrix of labels in the given class order."""
    index = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(y), len(classes)))
    out[np.arange(len(y)), [index[str(label)] for label in y]] = 1.0
    return out


def negative_log_likelihood(proba: np.ndarray, y: np.ndarray) -> float:
    """Return the mean negative log probability of the true class."""
    picked = (proba * one_hot(y)).sum(axis=1)
    return float(-np.mean(np.log(np.clip(picked, _PROBABILITY_FLOOR, 1.0))))


def fit_temperature(logits: np.ndarray, y: np.ndarray, bounds: tuple[float, float]) -> float:
    """Find the temperature minimizing the negative log likelihood of out-of-fold logits.

    Args:
        logits: Out-of-fold logits before temperature, canonical class order.
        y: True labels.
        bounds: Search interval for T.

    Returns:
        The fitted temperature.
    """
    result = minimize_scalar(
        lambda t: negative_log_likelihood(softmax(logits / t), y),
        bounds=bounds,
        method="bounded",
    )
    return float(result.x)


def brier_score(proba: np.ndarray, y: np.ndarray) -> float:
    """Return the multiclass Brier score: mean over rows of sum_k (p_k - y_k)^2."""
    return float(np.mean(((proba - one_hot(y)) ** 2).sum(axis=1)))


def reliability_table(proba: np.ndarray, y: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Return confidence bins with rows, mean confidence, and accuracy.

    Args:
        proba: Probabilities, canonical class order.
        y: True labels.
        n_bins: Number of equal-width bins on the maximum probability.

    Returns:
        One row per bin (empty bins included with zero rows).
    """
    confidence = proba.max(axis=1)
    correct = np.asarray(CANONICAL_CLASSES, dtype=object)[proba.argmax(axis=1)] == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        members = bins == b
        rows.append(
            {
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "rows": int(members.sum()),
                "mean_confidence": float(confidence[members].mean()) if members.any() else np.nan,
                "accuracy": float(correct[members].mean()) if members.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Return ECE: sum over bins of bin share times |accuracy - confidence|."""
    table = reliability_table(proba, y, n_bins)
    filled = table[table["rows"] > 0]
    weights = filled["rows"] / len(y)
    return float((weights * (filled["accuracy"] - filled["mean_confidence"]).abs()).sum())


def threshold_table(proba: np.ndarray, y: np.ndarray, thresholds: Sequence[float]) -> pd.DataFrame:
    """Return coverage and accuracy when rows below a confidence threshold go to review.

    Args:
        proba: Calibrated probabilities, canonical class order.
        y: True labels.
        thresholds: Candidate confidence thresholds.

    Returns:
        One row per threshold: coverage, accuracy kept, accuracy flagged, rows flagged.
    """
    confidence = proba.max(axis=1)
    correct = np.asarray(CANONICAL_CLASSES, dtype=object)[proba.argmax(axis=1)] == y
    rows = []
    for threshold in thresholds:
        kept = confidence >= threshold
        rows.append(
            {
                "threshold": threshold,
                "coverage": float(kept.mean()),
                "accuracy_kept": float(correct[kept].mean()) if kept.any() else np.nan,
                "accuracy_flagged": float(correct[~kept].mean()) if (~kept).any() else np.nan,
                "rows_flagged": int((~kept).sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(table: pd.DataFrame, target_kept_accuracy: float, default: float) -> float:
    """Return the lowest threshold whose kept accuracy reaches the target (D17 rule).

    Args:
        table: Output of ``threshold_table``.
        target_kept_accuracy: Required accuracy on rows above the threshold.
        default: Threshold used when no candidate reaches the target.

    Returns:
        The chosen threshold.
    """
    passing = table[table["accuracy_kept"] >= target_kept_accuracy]
    return float(passing["threshold"].min()) if not passing.empty else default
