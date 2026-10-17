"""Training reference statistics: what the model has seen.

Built from training rows only (or from the training rows of one CV fold for
out-of-fold flag analysis). Flags compare uploads against these statistics,
and explanations quote them as evidence. They never feed model features.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ledgerclf.config import ExplainParams, FeatureParams
from ledgerclf.features import GROUP_PAIR, normalize_text
from ledgerclf.labels import CANONICAL_CLASSES
from ledgerclf.schema import (
    AMOUNT,
    COL_ACCOUNT,
    COL_DESCRIPTION,
    COL_DOC_TYPE,
    COL_VENDOR,
    MISSING_TOKEN,
    PERIOD,
    TOKEN_COLUMNS,
)

PAIR_SEPARATOR = FeatureParams().pair_separator
PAIR_KEY = GROUP_PAIR
VALUE_COLUMNS = (COL_VENDOR, COL_ACCOUNT, COL_DOC_TYPE, COL_DESCRIPTION)

ClassCounts = dict[str, dict[str, dict[str, int]]]


def pair_key(vendor: object, account: object) -> str:
    """Return the normalized vendor-account pair key."""
    return normalize_text(vendor) + PAIR_SEPARATOR + normalize_text(account)


def _tokens(value: object) -> list[str]:
    text = normalize_text(value)
    return [] if text == MISSING_TOKEN else text.split()


@dataclass(frozen=True)
class ReferenceStats:
    """Known values, ranges, and class counts of the training rows.

    Value and token keys are in model form (lowercased, ``__missing__`` for blanks).
    """

    known_vendors: frozenset[str]
    known_accounts: frozenset[str]
    known_pairs: frozenset[str]
    known_doc_types: frozenset[str]
    known_tokens: dict[str, frozenset[str]]
    period_min: int
    period_max: int
    amount_min: float
    amount_max: float
    amount_median: float
    period_median: int
    amount_quantiles: tuple[float, ...]
    value_class_counts: ClassCounts
    token_class_counts: ClassCounts
    class_counts: dict[str, int]
    training_rows: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict with sorted keys and sorted sets."""
        return {
            "known_vendors": sorted(self.known_vendors),
            "known_accounts": sorted(self.known_accounts),
            "known_pairs": sorted(self.known_pairs),
            "known_doc_types": sorted(self.known_doc_types),
            "known_tokens": {k: sorted(v) for k, v in sorted(self.known_tokens.items())},
            "period_min": self.period_min,
            "period_max": self.period_max,
            "amount_min": self.amount_min,
            "amount_max": self.amount_max,
            "amount_median": self.amount_median,
            "period_median": self.period_median,
            "amount_quantiles": list(self.amount_quantiles),
            "value_class_counts": self.value_class_counts,
            "token_class_counts": self.token_class_counts,
            "class_counts": self.class_counts,
            "training_rows": self.training_rows,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReferenceStats:
        """Rebuild the statistics from ``to_dict`` output."""
        return cls(
            known_vendors=frozenset(payload["known_vendors"]),
            known_accounts=frozenset(payload["known_accounts"]),
            known_pairs=frozenset(payload["known_pairs"]),
            known_doc_types=frozenset(payload["known_doc_types"]),
            known_tokens={k: frozenset(v) for k, v in payload["known_tokens"].items()},
            period_min=int(payload["period_min"]),
            period_max=int(payload["period_max"]),
            amount_min=float(payload["amount_min"]),
            amount_max=float(payload["amount_max"]),
            amount_median=float(payload["amount_median"]),
            period_median=int(payload["period_median"]),
            amount_quantiles=tuple(float(q) for q in payload["amount_quantiles"]),
            value_class_counts=payload["value_class_counts"],
            token_class_counts=payload["token_class_counts"],
            class_counts={k: int(v) for k, v in payload["class_counts"].items()},
            training_rows=int(payload["training_rows"]),
        )

    def amount_percentile(self, amount: float) -> float:
        """Return the share of training amounts at or below ``amount`` (0 to 1)."""
        quantiles = np.asarray(self.amount_quantiles)
        return float(np.interp(amount, quantiles, np.linspace(0.0, 1.0, len(quantiles))))


def _counts(keys: list[str], y: np.ndarray) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for key, label in zip(keys, y, strict=True):
        table[key][str(label)] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(table.items())}


def _token_counts(values: pd.Series, y: np.ndarray) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for value, label in zip(values, y, strict=True):
        for token in set(_tokens(value)):
            table[token][str(label)] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(table.items())}


def build_reference(frame: pd.DataFrame, y: np.ndarray) -> ReferenceStats:
    """Build reference statistics from training rows.

    Args:
        frame: Coerced training rows.
        y: Their canonical labels.

    Returns:
        The reference statistics.
    """
    y = np.asarray(y, dtype=object)
    values = {c: [normalize_text(v) for v in frame[c]] for c in VALUE_COLUMNS}
    pairs = [pair_key(v, a) for v, a in zip(frame[COL_VENDOR], frame[COL_ACCOUNT], strict=True)]
    amounts = frame[AMOUNT].dropna().to_numpy(dtype=float)
    periods = frame[PERIOD].dropna().to_numpy(dtype=float)
    points = np.linspace(0, 100, ExplainParams().amount_quantile_points)
    value_counts = {c: _counts(values[c], y) for c in VALUE_COLUMNS}
    value_counts[PAIR_KEY] = _counts(pairs, y)
    return ReferenceStats(
        known_vendors=frozenset(values[COL_VENDOR]),
        known_accounts=frozenset(values[COL_ACCOUNT]),
        known_pairs=frozenset(pairs),
        known_doc_types=frozenset(values[COL_DOC_TYPE]),
        known_tokens={c: frozenset(t for v in frame[c] for t in _tokens(v)) for c in TOKEN_COLUMNS},
        period_min=int(periods.min()),
        period_max=int(periods.max()),
        amount_min=float(amounts.min()),
        amount_max=float(amounts.max()),
        amount_median=float(np.median(amounts)),
        period_median=int(np.median(periods)),
        amount_quantiles=tuple(float(q) for q in np.percentile(amounts, points)),
        value_class_counts=value_counts,
        token_class_counts={c: _token_counts(frame[c], y) for c in TOKEN_COLUMNS},
        class_counts={c: int((y == c).sum()) for c in CANONICAL_CLASSES},
        training_rows=len(y),
    )
