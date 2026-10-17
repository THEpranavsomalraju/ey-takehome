"""Row-level risk flags and the review policy (docs/04 section 10).

Flags mark risk; they never withhold a prediction (CLAUDE.md rule 8). A row
needs human review when it carries any flag whose severity is critical.
Severities are set by out-of-fold evidence in ``select`` (DECISIONS.md D18).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from ledgerclf.exceptions import ConfigError
from ledgerclf.features import normalize_text
from ledgerclf.reference import ReferenceStats, pair_key
from ledgerclf.schema import (
    AMOUNT,
    AMOUNT_WAS_MISSING,
    COL_ACCOUNT,
    COL_DOC_TYPE,
    COL_VENDOR,
    MISSING_TOKEN,
    PERIOD,
    PERIOD_WAS_MISSING,
    TEXT_TRUNCATED,
    TOKEN_COLUMNS,
)


class FlagSeverity(StrEnum):
    """How a flag affects review routing."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


LOW_CONFIDENCE = "LOW_CONFIDENCE"
ROW_STRUCTURE_SUSPECT = "ROW_STRUCTURE_SUSPECT"
# File-level issues whose rows may hold shifted or merged values.
ROW_STRUCTURE_CODES = ("ROW_HAS_EXTRA_FIELDS", "UNBALANCED_QUOTE")
FLAG_MESSAGES: dict[str, str] = {
    "UNSEEN_VENDOR": "New vendor: this vendor (Col1) is not in the historical invoices.",
    "UNSEEN_ACCOUNT": "New GL account: this account (Col6) is not in the historical invoices.",
    "UNSEEN_VENDOR_ACCOUNT_PAIR": (
        "New combination: this vendor and GL account were never used together before."
    ),
    "UNSEEN_DOC_TYPE": "New document flag: this Col7 value is not in the historical invoices.",
    "ALL_TOKENS_UNSEEN": (
        "Unfamiliar text: no word in the vendor, description, or account appears in the "
        "historical invoices."
    ),
    "TEXT_MISSING": "Missing details: the vendor (Col1) or GL account (Col6) is blank.",
    "AMOUNT_IMPUTED": (
        "Missing amount: the amount (Col3) was blank or unreadable, so a typical amount "
        "(the historical median) was used."
    ),
    "AMOUNT_OUT_OF_RANGE": "Unusual amount: the amount (Col3) is outside the historical range.",
    "DATE_IMPUTED": (
        "Missing date: the date (Col5) was blank or unreadable, so a typical month "
        "(the historical median) was used."
    ),
    "DATE_OUT_OF_RANGE": (
        "Date outside history: the posting date (Col5) is outside the months the model "
        "learned from, so the nearest known month was used."
    ),
    "TEXT_TRUNCATED": "Long text: a text value was over the length limit and was shortened.",
    ROW_STRUCTURE_SUSPECT: (
        "Possible file problem: this line had extra fields or an unclosed quote, so values "
        "may sit in the wrong columns."
    ),
    LOW_CONFIDENCE: (
        "Low confidence: the model's confidence is below the review threshold, so a person "
        "should confirm the category."
    ),
}
DEFAULT_SEVERITIES: dict[str, FlagSeverity] = {
    "UNSEEN_VENDOR": FlagSeverity.CRITICAL,
    "UNSEEN_ACCOUNT": FlagSeverity.WARNING,
    "UNSEEN_VENDOR_ACCOUNT_PAIR": FlagSeverity.WARNING,
    "UNSEEN_DOC_TYPE": FlagSeverity.WARNING,
    "ALL_TOKENS_UNSEEN": FlagSeverity.CRITICAL,
    "TEXT_MISSING": FlagSeverity.CRITICAL,
    "AMOUNT_IMPUTED": FlagSeverity.CRITICAL,
    "AMOUNT_OUT_OF_RANGE": FlagSeverity.WARNING,
    "DATE_IMPUTED": FlagSeverity.CRITICAL,
    "DATE_OUT_OF_RANGE": FlagSeverity.WARNING,
    "TEXT_TRUNCATED": FlagSeverity.INFO,
    ROW_STRUCTURE_SUSPECT: FlagSeverity.CRITICAL,
    LOW_CONFIDENCE: FlagSeverity.CRITICAL,
}
# Flags computed from the row values and the reference statistics.
ROW_FLAGS: tuple[str, ...] = tuple(
    code for code in FLAG_MESSAGES if code not in (LOW_CONFIDENCE, ROW_STRUCTURE_SUSPECT)
)


@dataclass(frozen=True)
class ReviewPolicy:
    """Flag severities plus the confidence threshold for ``LOW_CONFIDENCE``."""

    threshold: float
    severities: dict[str, FlagSeverity] = field(default_factory=lambda: dict(DEFAULT_SEVERITIES))

    def critical_codes(self) -> list[str]:
        """Return the flag codes that require review, sorted."""
        return sorted(code for code, s in self.severities.items() if s is FlagSeverity.CRITICAL)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dict."""
        return {
            "threshold": self.threshold,
            "severities": {k: str(v) for k, v in sorted(self.severities.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ReviewPolicy:
        """Rebuild a policy from ``to_dict`` output; unknown or missing flags keep defaults.

        Raises:
            ConfigError: When the payload is malformed.
        """
        severities = payload.get("severities")
        threshold = payload.get("threshold")
        if not isinstance(severities, dict) or not isinstance(threshold, int | float):
            raise ConfigError(
                "The review policy in champion.json is malformed. Run: python -m ledgerclf select",
                code="POLICY_MALFORMED",
            )
        loaded = {str(k): FlagSeverity(str(v)) for k, v in severities.items()}
        return cls(threshold=float(threshold), severities={**DEFAULT_SEVERITIES, **loaded})


def _has_known_token(value: object, known: frozenset[str]) -> bool:
    text = normalize_text(value)
    return text != MISSING_TOKEN and any(token in known for token in text.split())


def _all_tokens_unseen(frame: pd.DataFrame, reference: ReferenceStats) -> np.ndarray:
    seen_any = np.zeros(len(frame), dtype=bool)
    for column in TOKEN_COLUMNS:
        known = reference.known_tokens[column]
        seen_any |= np.array([_has_known_token(v, known) for v in frame[column]], dtype=bool)
    return ~seen_any


def compute_row_flags(frame: pd.DataFrame, reference: ReferenceStats) -> pd.DataFrame:
    """Return one boolean column per row flag (all flags except ``LOW_CONFIDENCE``).

    Args:
        frame: Coerced rows.
        reference: Training reference statistics.

    Returns:
        Boolean frame with columns ``ROW_FLAGS``, aligned with ``frame``.
    """
    vendor = frame[COL_VENDOR].map(normalize_text)
    account = frame[COL_ACCOUNT].map(normalize_text)
    doc = frame[COL_DOC_TYPE].map(normalize_text)
    pairs = [pair_key(v, a) for v, a in zip(frame[COL_VENDOR], frame[COL_ACCOUNT], strict=True)]
    vendor_known = vendor.isin(reference.known_vendors).to_numpy()
    account_known = account.isin(reference.known_accounts).to_numpy()
    pair_known = np.isin(np.asarray(pairs, dtype=object), sorted(reference.known_pairs))
    amount = frame[AMOUNT].to_numpy(dtype=float)
    period = frame[PERIOD].to_numpy(dtype=float)
    text_missing = (vendor == MISSING_TOKEN).to_numpy() | (account == MISSING_TOKEN).to_numpy()
    with np.errstate(invalid="ignore"):
        flags = {
            "UNSEEN_VENDOR": ~vendor_known & (vendor != MISSING_TOKEN).to_numpy(),
            "UNSEEN_ACCOUNT": ~account_known & (account != MISSING_TOKEN).to_numpy(),
            "UNSEEN_VENDOR_ACCOUNT_PAIR": ~pair_known & vendor_known & account_known,
            "UNSEEN_DOC_TYPE": ~doc.isin(reference.known_doc_types).to_numpy(),
            "ALL_TOKENS_UNSEEN": _all_tokens_unseen(frame, reference),
            "TEXT_MISSING": text_missing,
            "AMOUNT_IMPUTED": frame[AMOUNT_WAS_MISSING].to_numpy(dtype=bool),
            "AMOUNT_OUT_OF_RANGE": (amount < reference.amount_min)
            | (amount > reference.amount_max),
            "DATE_IMPUTED": frame[PERIOD_WAS_MISSING].to_numpy(dtype=bool),
            "DATE_OUT_OF_RANGE": (period < reference.period_min) | (period > reference.period_max),
            "TEXT_TRUNCATED": frame[TEXT_TRUNCATED].to_numpy(dtype=bool),
        }
    return pd.DataFrame({code: np.asarray(flags[code], dtype=bool) for code in ROW_FLAGS})


def apply_review_policy(
    flags: pd.DataFrame, confidence: np.ndarray, policy: ReviewPolicy
) -> tuple[pd.DataFrame, np.ndarray]:
    """Add ``LOW_CONFIDENCE`` and decide which rows need review.

    Args:
        flags: Output of ``compute_row_flags``.
        confidence: Calibrated probability of the predicted class per row.
        policy: Severities and threshold.

    Returns:
        Tuple of (flags including ``LOW_CONFIDENCE``, boolean review_required).
    """
    full = flags.copy()
    full[LOW_CONFIDENCE] = np.asarray(confidence) < policy.threshold
    critical = [code for code in policy.critical_codes() if code in full.columns]
    review = full[critical].any(axis=1).to_numpy() if critical else np.zeros(len(full), dtype=bool)
    return full, review


def flag_strings(flags: pd.DataFrame) -> list[str]:
    """Return the pipe-separated flag codes of every row (empty string when none)."""
    codes = list(flags.columns)
    matrix = flags.to_numpy(dtype=bool)
    return ["|".join(code for code, on in zip(codes, row, strict=True) if on) for row in matrix]
