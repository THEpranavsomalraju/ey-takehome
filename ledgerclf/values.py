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

from ledgerclf.config import EXCEL_SERIAL_RANGE, MONTHS_PER_YEAR, VALID_YEARS

MISSING_MARKERS = frozenset({"", "nan", "none", "null", "n/a", "na", "#n/a", "-", "nat"})
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_WHITESPACE = re.compile(r"\s+")

_CURRENCY = re.compile(r"^(?:usd|eur|gbp|\$|\u20ac|\u00a3)\s*|\s*(?:usd|eur|gbp|\$|\u20ac|\u00a3)$")
_PLAIN_NUMBER = re.compile(r"^(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$", re.ASCII)
# European notation only when unambiguous: a decimal comma, or two or more dot groups.
# "1.234" and "12.500" stay dot-decimal numbers (DECISIONS.md D27).
_EUROPEAN = re.compile(r"^\d{1,3}(?:\.\d{3})+,\d+$|^\d{1,3}(?:\.\d{3}){2,}$", re.ASCII)
_GROUPED = re.compile(r"^\d{1,3}(?P<sep>[,' ])\d{3}(?:(?P=sep)\d{3})*(?:\.\d+)?$", re.ASCII)

_EXCEL_ORIGIN = dt.datetime(1899, 12, 30)
_ISO_DATE = re.compile(r"^(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?(?:[ T].*)?$", re.ASCII)
_COMPACT_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})$", re.ASCII)
_COMPACT_DATE_RANGE = (19_000_101, 22_001_231)
_NUMERIC_DATE = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})(?:[ T].*)?$", re.ASCII)
_MONTH_NAMES = (
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
)
_MONTHS = {name: index for index, names in enumerate(_MONTH_NAMES, start=1) for name in names}


def is_missing(value: object) -> bool:
    """Return True for None, NaN, NaT, and pandas missing scalars.

    Args:
        value: Any cell value.

    Returns:
        True when the value represents a missing cell.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, float | np.floating):
        return bool(np.isnan(value))
    return False


def strip_invisible(text: str) -> str:
    """NFKC-normalize text and remove invisible format characters.

    Removes every Unicode "format" (Cf) character: zero-width spaces and
    joiners, byte-order marks, and direction marks that right-to-left locale
    exports insert. ASCII text is returned unchanged without a scan.

    Args:
        text: Any string.

    Returns:
        The normalized string without format characters.
    """
    if text.isascii():
        return text
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def clean_text(value: object, max_chars: int | None = None) -> tuple[str | None, bool]:
    """Convert any cell to a cleaned display string.

    Integer-valued floats lose their ``.0`` so that ``10719.0`` and
    ``10719`` match. Text is NFKC-normalized, invisible characters are removed,
    whitespace collapses, and missing markers become None.

    Args:
        value: Raw cell value.
        max_chars: Maximum characters kept; longer text is truncated. None keeps all.

    Returns:
        Tuple of (cleaned text or None, True when truncated).
    """
    if is_missing(value):
        return None, False
    text = strip_invisible(_to_text(value))
    text = _WHITESPACE.sub(" ", _CONTROL.sub(" ", text)).strip()
    if text.lower() in MISSING_MARKERS:
        return None, False
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars], True
    return text, False


def is_blank_cell(value: object) -> bool:
    """Return True when a cell is missing or holds only whitespace or a missing marker.

    Args:
        value: Raw cell value.

    Returns:
        True for blank cells.
    """
    return clean_text(value)[0] is None


def _to_text(value: object) -> str:
    if isinstance(value, bool | np.bool_):
        return "True" if value else "False"
    if (
        isinstance(value, float | np.floating)
        and math.isfinite(value)
        and float(value).is_integer()
    ):
        return str(int(value))
    if isinstance(value, pd.Timestamp | dt.datetime):
        return value.isoformat(sep=" ")
    return str(value)


def parse_amount(value: object) -> float | None:
    """Parse an amount cell into a finite float.

    Accepts numbers and strings with currency symbols or codes on either side
    of the sign, thousands separators, parentheses or a trailing minus for
    negatives, and unambiguous European ``1.234,56`` notation. Malformed text
    returns None instead of a guess.

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
    return _parse_amount_text(strip_invisible(value))


def _parse_amount_text(text: str) -> float | None:
    text = text.strip().replace("\u2212", "-").lower()
    signs: list[str] = []
    if text.startswith("(") and text.endswith(")"):
        signs.append("-")
        text = text[1:-1].strip()
    text = _strip_sign(text, signs)  # "-$5" -> "$5"
    text = _CURRENCY.sub("", text).strip()  # "$5" -> "5", "5 usd" -> "5"
    text = _strip_sign(text, signs)  # "$-5" -> "-5" -> "5"
    if text.endswith("-"):
        signs.append("-")
        text = text[:-1].strip()
    if len(signs) > 1:  # "(-5)", "--5", "+-5": contradictory signs are rejected, not guessed
        return None
    number = _number_from_text(text)
    if number is None:
        return None
    return -number if "-" in signs else number


def _strip_sign(text: str, signs: list[str]) -> str:
    """Remove one leading sign, recording it in ``signs``."""
    if text[:1] in ("-", "+"):
        signs.append(text[0])
        return text[1:].lstrip()
    return text


def _number_from_text(text: str) -> float | None:
    if _EUROPEAN.match(text):
        number = float(text.replace(".", "").replace(",", "."))
    elif grouped := _GROUPED.match(text):
        number = float(text.replace(grouped.group("sep"), ""))
    elif _PLAIN_NUMBER.match(text):
        number = float(text)
    else:
        return None
    return number if math.isfinite(number) else None


def period_index(year: int, month: int) -> int:
    """Return the model's period feature, ``year * 12 + month``.

    Args:
        year: Calendar year.
        month: Month number, 1 to 12.

    Returns:
        The integer period index.
    """
    return year * MONTHS_PER_YEAR + month


def period_label(index: int) -> str:
    """Convert a period index back to ``YYYY-MM``.

    Args:
        index: Value produced by ``period_index``.

    Returns:
        The display label.
    """
    year, month = divmod(index - 1, MONTHS_PER_YEAR)
    return f"{year:04d}-{month + 1:02d}"


@dataclass(frozen=True)
class ParsedDate:
    """A parsed calendar date reduced to its monthly period."""

    year: int
    month: int

    @property
    def period_index(self) -> int:
        """Return ``year * 12 + month``, the model's period feature."""
        return period_index(self.year, self.month)

    @property
    def label(self) -> str:
        """Return the display form ``YYYY-MM``."""
        return f"{self.year:04d}-{self.month:02d}"


def date_order_evidence(value: object) -> str | None:
    """Tell whether an ``a/b/yyyy`` string proves the column's day and month order.

    Only a valid date counts: ``31/08/2018`` proves day-first, ``08/31/2018``
    proves month-first, and a typo such as ``45/08/2018`` proves nothing.

    Args:
        value: Raw cell value.

    Returns:
        ``"day_first"``, ``"month_first"``, or None when the value proves neither.
    """
    if not isinstance(value, str):
        return None
    match = _NUMERIC_DATE.match(strip_invisible(value).strip())
    if match is None:
        return None
    first, second, year = (int(part) for part in match.groups())
    if first > MONTHS_PER_YEAR and _checked(year, second, first) is not None:
        return "day_first"
    if second > MONTHS_PER_YEAR and _checked(year, first, second) is not None:
        return "month_first"
    return None


def is_ambiguous_numeric_date(value: object) -> bool:
    """Return True for ``a/b/yyyy`` strings where both parts could be the month.

    Args:
        value: Raw cell value.

    Returns:
        True when the day and month order cannot be told from this value alone.
    """
    if not isinstance(value, str):
        return False
    match = _NUMERIC_DATE.match(strip_invisible(value).strip())
    if match is None:
        return False
    first, second = int(match.group(1)), int(match.group(2))
    return first <= MONTHS_PER_YEAR and second <= MONTHS_PER_YEAR and first != second


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
        return _from_number(float(value))
    if isinstance(value, str):
        return _parse_date_text(strip_invisible(value).strip(), day_first)
    return None


def _from_number(number: float) -> ParsedDate | None:
    """Read a numeric cell as an Excel serial date or a ``yyyymmdd`` integer."""
    low, high = _COMPACT_DATE_RANGE
    if math.isfinite(number) and number.is_integer() and low <= number <= high:
        compact = _COMPACT_DATE.match(str(int(number)))
        if compact:
            return _checked(*(int(part) for part in compact.groups()))
    return _from_serial(number)


def _from_serial(number: float) -> ParsedDate | None:
    low, high = EXCEL_SERIAL_RANGE
    if not math.isfinite(number) or not low <= number <= high:
        return None
    stamp = _EXCEL_ORIGIN + dt.timedelta(days=int(number))
    return _checked(stamp.year, stamp.month, stamp.day)


def _parse_date_text(text: str, day_first: bool) -> ParsedDate | None:
    if not text:
        return None
    compact = _COMPACT_DATE.match(text)
    if compact:
        return _checked(*(int(part) for part in compact.groups()))
    iso = _ISO_DATE.match(text)  # before the serial check, so "2018.08" reads as a month
    if iso:
        year, month, day = iso.groups()
        return _checked(int(year), int(month), int(day) if day else 1)
    if re.fullmatch(r"\d+(?:\.\d+)?", text, re.ASCII):
        return _from_serial(float(text))
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
