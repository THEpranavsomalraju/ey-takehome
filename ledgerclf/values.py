"""Cell-level parsers for text, amounts, and posting periods (docs/05 section 2.5).

Every parser is a pure function of one cell. Parsers never raise on bad
values: they return None so that the caller can flag and impute.
Column-level decisions (day-first inference) live in ``schema.coerce_values``.
"""

from __future__ import annotations

import datetime as dt
import math
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

MISSING_MARKERS = frozenset({"", "nan", "none", "null", "n/a", "na", "#n/a", "-", "nat"})
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_WHITESPACE = re.compile(r"\s+")

_CURRENCY = re.compile(r"^(?:usd|eur|gbp|\$|€|£)\s*|\s*(?:usd|eur|gbp|\$|€|£)$")
_PLAIN_NUMBER = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$", re.ASCII)
_EUROPEAN = re.compile(r"^[-+]?\d{1,3}(?:\.\d{3})+(?:,\d+)?$", re.ASCII)
_GROUPED = re.compile(r"^[-+]?\d{1,3}(?P<sep>[,' ])\d{3}(?:(?P=sep)\d{3})*(?:\.\d+)?$", re.ASCII)

EXCEL_SERIAL_RANGE = (20_000, 80_000)
_EXCEL_ORIGIN = dt.datetime(1899, 12, 30)
VALID_YEARS = (1900, 2200)
_ISO_DATE = re.compile(r"^(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?(?:[ T].*)?$", re.ASCII)
_COMPACT_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})$", re.ASCII)
_NUMERIC_DATE = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})(?:[ T].*)?$", re.ASCII)
_MONTHS = {
    name: index
    for index, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for name in names
}


def is_missing(value: object) -> bool:
    """Return True for None, NaN, NaT, and pandas missing scalars."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, float | np.floating):
        return bool(np.isnan(value))
    return False


def clean_text(value: object, max_chars: int) -> tuple[str | None, bool]:
    """Convert any cell to a cleaned display string.

    Integer-valued floats lose their ``.0`` so that ``10719.0`` and
    ``10719`` match. Text is NFKC-normalized, invisible characters are removed,
    whitespace collapses, and missing markers become None.

    Args:
        value: Raw cell value.
        max_chars: Maximum characters kept; longer text is truncated.

    Returns:
        Tuple of (cleaned text or None, True when truncated).
    """
    if is_missing(value):
        return None, False
    text = _to_text(value)
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE)
    text = _WHITESPACE.sub(" ", _CONTROL.sub(" ", text)).strip()
    if text.lower() in MISSING_MARKERS:
        return None, False
    if len(text) > max_chars:
        return text[:max_chars], True
    return text, False


def _to_text(value: object) -> str:
    if isinstance(value, bool | np.bool_):
        return "True" if value else "False"
    if isinstance(value, float | np.floating) and math.isfinite(value) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, pd.Timestamp | dt.datetime):
        return value.isoformat(sep=" ")
    return str(value)


def parse_amount(value: object) -> float | None:
    """Parse an amount cell into a finite float.

    Accepts numbers and strings with currency symbols or codes, thousands
    separators, parentheses or a trailing minus for negatives, and European
    ``1.234,56`` notation. Ambiguous or malformed text returns None.

    Args:
        value: Raw cell value.

    Returns:
        The amount, or None when missing, infinite, or unparseable.
    """
    if is_missing(value) or isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, int | float | np.integer | np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if not isinstance(value, str):
        return None
    return _parse_amount_text(unicodedata.normalize("NFKC", value))


def _parse_amount_text(text: str) -> float | None:
    text = text.strip().replace("−", "-").lower()
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative, text = True, text[1:-1].strip()
    text = _CURRENCY.sub("", text).strip()
    if text.endswith("-") and not text.startswith("-"):
        negative, text = True, text[:-1].strip()
    number = _number_from_text(text)
    if number is None or (negative and number < 0) or not math.isfinite(number):
        return None
    return -number if negative else number


def _number_from_text(text: str) -> float | None:
    if _EUROPEAN.match(text):
        return float(text.replace(".", "").replace(",", "."))
    grouped = _GROUPED.match(text)
    if grouped:
        return float(text.replace(grouped.group("sep"), ""))
    if _PLAIN_NUMBER.match(text):
        return float(text)
    return None


@dataclass(frozen=True)
class ParsedDate:
    """A parsed calendar date reduced to its monthly period."""

    year: int
    month: int

    @property
    def period_index(self) -> int:
        """Return ``year * 12 + month``, the model's period feature."""
        return self.year * 12 + self.month

    @property
    def label(self) -> str:
        """Return the display form ``YYYY-MM``."""
        return f"{self.year:04d}-{self.month:02d}"


def period_label(period_index: int) -> str:
    """Convert a period index back to ``YYYY-MM``."""
    year, month = divmod(period_index - 1, 12)
    return f"{year:04d}-{month + 1:02d}"


def numeric_date_first_part(value: object) -> int | None:
    """Return the first number of an ``a/b/yyyy`` string, else None.

    Used by column-level day-first inference.
    """
    if not isinstance(value, str):
        return None
    match = _NUMERIC_DATE.match(value.strip())
    return int(match.group(1)) if match else None


def is_ambiguous_numeric_date(value: object) -> bool:
    """Return True for ``a/b/yyyy`` strings where both parts could be the month."""
    if not isinstance(value, str):
        return False
    match = _NUMERIC_DATE.match(value.strip())
    if match is None:
        return False
    first, second = int(match.group(1)), int(match.group(2))
    return first <= 12 and second <= 12 and first != second


def parse_date(value: object, day_first: bool) -> ParsedDate | None:
    """Parse a date cell into a monthly period.

    Args:
        value: Raw cell value (datetime, Excel serial number, or text).
        day_first: Read ``a/b/yyyy`` as day/month/year when True.

    Returns:
        The parsed period, or None when missing or unparseable.
    """
    if is_missing(value) or isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, pd.Timestamp | dt.datetime | dt.date | np.datetime64):
        stamp = pd.Timestamp(value)
        return _checked(stamp.year, stamp.month, stamp.day)
    if isinstance(value, int | float | np.integer | np.floating):
        return _from_serial(float(value))
    if isinstance(value, str):
        return _parse_date_text(unicodedata.normalize("NFKC", value).strip(), day_first)
    return None


def _from_serial(number: float) -> ParsedDate | None:
    low, high = EXCEL_SERIAL_RANGE
    if not math.isfinite(number) or not low <= number <= high:
        return None
    stamp = _EXCEL_ORIGIN + dt.timedelta(days=int(number))
    return _checked(stamp.year, stamp.month, stamp.day)


def _parse_date_text(text: str, day_first: bool) -> ParsedDate | None:
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text, re.ASCII):
        compact = _COMPACT_DATE.match(text)
        if compact:
            return _checked(*(int(part) for part in compact.groups()))
        return _from_serial(float(text))
    iso = _ISO_DATE.match(text)
    if iso:
        year, month, day = iso.groups()
        return _checked(int(year), int(month), int(day) if day else 1)
    numeric = _NUMERIC_DATE.match(text)
    if numeric:
        first, second, year = (int(part) for part in numeric.groups())
        day, month = (first, second) if day_first else (second, first)
        return _checked(year, month, day)
    return _parse_month_name(text)


def _parse_month_name(text: str) -> ParsedDate | None:
    tokens = [t for t in re.split(r"[\s,./\-]+", text.lower()) if t]
    months = [_MONTHS[t] for t in tokens if t in _MONTHS]
    years = [int(t) for t in tokens if re.fullmatch(r"\d{4}", t, re.ASCII)]
    days = [int(t) for t in tokens if re.fullmatch(r"\d{1,2}", t, re.ASCII)]
    leftovers = [
        t for t in tokens if t not in _MONTHS and not re.fullmatch(r"\d{1,4}", t, re.ASCII)
    ]
    if len(months) != 1 or len(years) != 1 or len(days) > 1 or leftovers:
        return None
    return _checked(years[0], months[0], days[0] if days else 1)


def _checked(year: int, month: int, day: int) -> ParsedDate | None:
    if not VALID_YEARS[0] <= year <= VALID_YEARS[1]:
        return None
    try:
        dt.date(year, month, day)
    except ValueError:
        return None
    return ParsedDate(year=year, month=month)
