"""Input schema: column names, column resolution, value coercion, validation report.

Every source (training file, holdout file, upload) passes through
``resolve_columns`` then ``coerce_values``, so training and inference see
identical cleaning (docs/05 section 1, principle 5).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from ledgerclf.config import Limits
from ledgerclf.exceptions import SchemaError
from ledgerclf.values import (
    clean_text,
    is_ambiguous_numeric_date,
    numeric_date_first_part,
    parse_amount,
    parse_date,
)

logger = logging.getLogger(__name__)

COL_VENDOR = "Col1"
COL_INVOICE = "Col2"
COL_AMOUNT = "Col3"
COL_DESCRIPTION = "Col4"
COL_PERIOD = "Col5"
COL_ACCOUNT = "Col6"
COL_DOC_TYPE = "Col7"

INPUT_COLUMNS: tuple[str, ...] = (
    COL_VENDOR,
    COL_INVOICE,
    COL_AMOUNT,
    COL_DESCRIPTION,
    COL_PERIOD,
    COL_ACCOUNT,
    COL_DOC_TYPE,
)
MODEL_INPUT_COLUMNS: tuple[str, ...] = tuple(c for c in INPUT_COLUMNS if c != COL_INVOICE)
REQUIRED_COLUMNS: tuple[str, ...] = MODEL_INPUT_COLUMNS
OPTIONAL_COLUMNS: tuple[str, ...] = (COL_INVOICE,)
TEXT_COLUMNS: tuple[str, ...] = (COL_VENDOR, COL_INVOICE, COL_DESCRIPTION, COL_ACCOUNT, COL_DOC_TYPE)
TOKEN_COLUMNS: tuple[str, ...] = (COL_VENDOR, COL_DESCRIPTION, COL_ACCOUNT)
LABEL_COLUMN = "ClassificationLabel"
MISSING_TOKEN = "__missing__"

ROW_NUMBER = "row_number"
SOURCE_ROW = "source_row"
AMOUNT = "amount"
PERIOD = "period"
AMOUNT_WAS_MISSING = "amount_was_missing"
PERIOD_WAS_MISSING = "period_was_missing"
TEXT_TRUNCATED = "text_truncated"

KNOWN_DOC_TYPES: dict[str, str] = {"nodoc": "NoDoc", "doc1": "Doc1", "doc2": "Doc2", "doc3": "Doc3"}

_HEADER_ALIASES: dict[str, str] = {
    **{f"col{i}": name for i, name in enumerate(INPUT_COLUMNS, start=1)},
    **dict(zip("abcdefg", INPUT_COLUMNS, strict=True)),
    **{alias: LABEL_COLUMN for alias in ("classificationlabel", "label", "class", "target", "h")},
}
_HEADER_STRIP = re.compile(r"[\s_\-.]+")
_INDEX_ARTIFACT = re.compile(r"^(unnamed\d*|index|)$")


class Severity(StrEnum):
    """Severity of a file-level validation issue (docs/05 section 2.6)."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True)
class ValidationIssue:
    """One file-level finding, with a plain-English message."""

    severity: Severity
    code: str
    message: str
    column: str | None = None
    row_numbers: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dict."""
        return {
            "severity": str(self.severity),
            "code": self.code,
            "message": self.message,
            "column": self.column,
            "row_numbers": [int(r) for r in self.row_numbers],
        }


@dataclass
class ValidationReport:
    """Every issue found while reading, resolving, and coercing one file."""

    issues: list[ValidationIssue] = field(default_factory=list)
    rows_read: int = 0
    rows_dropped_empty: int = 0
    rows_predicted: int = 0
    column_mapping: dict[str, str] = field(default_factory=dict)

    def has_errors(self) -> bool:
        """Return True when any issue has ERROR severity."""
        return any(issue.severity is Severity.ERROR for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dict."""
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "rows_read": int(self.rows_read),
            "rows_dropped_empty": int(self.rows_dropped_empty),
            "rows_predicted": int(self.rows_predicted),
            "column_mapping": {str(k): str(v) for k, v in self.column_mapping.items()},
        }


def issue(
    severity: Severity,
    code: str,
    message: str,
    column: str | None = None,
    rows: Iterable[int] = (),
) -> ValidationIssue:
    """Build a ValidationIssue with sorted, de-duplicated row numbers."""
    return ValidationIssue(severity, code, message, column, tuple(sorted({int(r) for r in rows})))


def normalize_header(value: object) -> str:
    """Normalize a header cell for matching: lowercase, no spaces or separators."""
    text, _ = clean_text(value, max_chars=200)
    if text is None:
        return ""
    return _HEADER_STRIP.sub("", unicodedata.normalize("NFKC", text).lower())


def canonical_header(value: object) -> str | None:
    """Return the canonical column name for a header cell, or None."""
    return _HEADER_ALIASES.get(normalize_header(value))


def find_header_row(raw: pd.DataFrame, limits: Limits) -> int | None:
    """Find the header row within the first rows of a raw sheet.

    Args:
        raw: Sheet read with ``header=None``.
        limits: Provides the scan depth and the minimum number of matches.

    Returns:
        The 0-based row index of the first row where enough cells match
        known column names, or None when no row qualifies.
    """
    for position in range(min(limits.header_scan_rows, len(raw))):
        names = {canonical_header(v) for v in raw.iloc[position].tolist()} - {None}
        if len(names & set(INPUT_COLUMNS)) >= limits.header_min_matches:
            return position
    return None


def resolve_columns(
    raw: pd.DataFrame, limits: Limits
) -> tuple[pd.DataFrame, dict[str, str], list[ValidationIssue]]:
    """Map a raw sheet to canonical column names.

    Args:
        raw: Sheet read with ``header=None`` and ``dtype=object``.
        limits: Input limits (header scan depth).

    Returns:
        Tuple of (frame with ``source_row`` plus canonical columns, mapping
        from source header to canonical name, issues).

    Raises:
        SchemaError: When required columns are missing or the sheet is empty.
    """
    if raw.shape[1] == 0 or raw.shape[0] == 0:
        raise SchemaError(
            "The file holds no data. Upload a file with a header row and at least one data row.",
            code="NO_DATA_ROWS",
        )
    header_row = find_header_row(raw, limits)
    if header_row is None:
        return _resolve_by_position(_trim_trailing_blank_columns(raw))
    return _resolve_by_header(raw, header_row)


def _is_blank(values: pd.Series) -> bool:
    return bool(values.map(lambda v: clean_text(v, 10)[0] is None).all())


def _trim_trailing_blank_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop blank columns at the right edge, such as those from a trailing delimiter."""
    width = raw.shape[1]
    while width > 0 and _is_blank(raw.iloc[:, width - 1]):
        width -= 1
    return raw.iloc[:, :width]


def _resolve_by_header(
    raw: pd.DataFrame, header_row: int
) -> tuple[pd.DataFrame, dict[str, str], list[ValidationIssue]]:
    headers = raw.iloc[header_row].tolist()
    body = raw.iloc[header_row + 1 :]
    issues: list[ValidationIssue] = []
    chosen: dict[str, int] = {}
    extras: list[str] = []
    mapping: dict[str, str] = {}
    for position, header in enumerate(headers):
        name = canonical_header(header)
        label = clean_text(header, 200)[0] or ""
        if name is None:
            values = body.iloc[:, position]
            if not (label == "" and _is_blank(values)) and not _is_index_artifact(label, values):
                extras.append(label or f"(unnamed column {position + 1})")
            continue
        if name in chosen:
            issues.append(
                issue(
                    Severity.WARNING,
                    "DUPLICATE_COLUMN",
                    f"Column '{name}' appears more than once. The first occurrence was used.",
                    column=name,
                )
            )
            continue
        chosen[name] = position
        mapping[label] = name
    if extras:
        issues.append(
            issue(
                Severity.WARNING,
                "EXTRA_COLUMNS_IGNORED",
                "Ignored columns not used by the model: " + ", ".join(extras) + ".",
            )
        )
    frame = _assemble(body, chosen, first_source_row=header_row + 2)
    issues.extend(_check_required(frame, [str(h) for h in headers]))
    return _with_optional(frame, issues), mapping, issues


def _resolve_by_position(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], list[ValidationIssue]]:
    width = raw.shape[1]
    if width == 0:
        raise SchemaError(
            "The file holds no data. Upload a file with a header row and at least one data row.",
            code="NO_DATA_ROWS",
        )
    if width not in (len(INPUT_COLUMNS), len(INPUT_COLUMNS) + 1):
        first_row = ", ".join(str(clean_text(v, 40)[0]) for v in raw.iloc[0].tolist())
        raise SchemaError(
            "Could not find a header row naming the input columns Col1 to Col7 "
            f"(found in the first row: {first_row}). Add a header row with Col1 to Col7 "
            "(or A to G), or upload a file with exactly 7 input columns in that order.",
            code="MISSING_REQUIRED_COLUMNS",
        )
    names = [*INPUT_COLUMNS, LABEL_COLUMN][:width]
    chosen = {name: position for position, name in enumerate(names)}
    frame = _assemble(raw, chosen, first_source_row=1)
    mapping = {f"position {p + 1}": n for n, p in chosen.items()}
    warning = issue(
        Severity.WARNING,
        "POSITIONAL_MAPPING",
        f"No header row found. Columns were mapped by position: first {len(INPUT_COLUMNS)} "
        "to Col1 to Col7" + (", the last to the label" if width > len(INPUT_COLUMNS) else "") + ".",
    )
    return frame, mapping, [warning]


def _is_index_artifact(label: str, values: pd.Series) -> bool:
    if not _INDEX_ARTIFACT.match(normalize_header(label)):
        return False
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.isna().any() or len(numbers) == 0:
        return False
    start = numbers.iloc[0]
    return bool((numbers.to_numpy() == np.arange(start, start + len(numbers))).all())


def _assemble(body: pd.DataFrame, chosen: dict[str, int], first_source_row: int) -> pd.DataFrame:
    data = {name: body.iloc[:, position].to_numpy(dtype=object) for name, position in chosen.items()}
    frame = pd.DataFrame(data, dtype=object)
    frame.insert(0, SOURCE_ROW, np.arange(first_source_row, first_source_row + len(body)))
    return frame


def _check_required(frame: pd.DataFrame, headers: list[str]) -> list[ValidationIssue]:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        found = [h for h in headers if clean_text(h, 200)[0] is not None]
        raise SchemaError(
            f"Missing required columns: {', '.join(missing)}. "
            f"Found columns: {', '.join(found) or 'none'}. Rename the columns to match "
            "Col1 to Col7, or upload a file with all seven input columns.",
            code="MISSING_REQUIRED_COLUMNS",
        )
    return []


def _with_optional(frame: pd.DataFrame, issues: list[ValidationIssue]) -> pd.DataFrame:
    if COL_INVOICE not in frame.columns:
        frame = frame.copy()
        frame.insert(1, COL_INVOICE, None)
        issues.append(
            issue(
                Severity.WARNING,
                "OPTIONAL_COLUMN_MISSING",
                "Column Col2 (invoice number) is missing. Predictions do not use Col2, "
                "so results are unaffected.",
                column=COL_INVOICE,
            )
        )
    return frame


def coerce_values(
    frame: pd.DataFrame, limits: Limits
) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    """Clean every value of a resolved frame into typed columns.

    Coercion never imputes: missing or unparseable amounts and dates become
    NaN with a ``*_was_missing`` indicator, and the pipeline's training-fold
    imputer fills them identically in training and inference.

    Args:
        frame: Output of ``resolve_columns``.
        limits: Input limits (text length).

    Returns:
        Tuple of (typed frame, issues). The frame holds ``row_number``,
        ``source_row``, display strings for Col1 to Col7 (and the raw label
        when present), ``amount``, ``period``, and indicator columns.

    Raises:
        SchemaError: When no non-empty data rows remain.
    """
    issues: list[ValidationIssue] = []
    out = pd.DataFrame({SOURCE_ROW: frame[SOURCE_ROW].to_numpy(dtype=np.int64)})
    truncated = np.zeros(len(frame), dtype=bool)
    for column in TEXT_COLUMNS:
        cleaned = [clean_text(v, limits.max_text_chars) for v in frame[column].tolist()]
        out[column] = pd.Series([c[0] for c in cleaned], dtype=object)
        truncated |= np.array([c[1] for c in cleaned], dtype=bool)
    out[COL_DOC_TYPE] = out[COL_DOC_TYPE].map(_canonical_doc_type).astype(object)
    out[AMOUNT] = _coerce_amounts(frame[COL_AMOUNT])
    out[PERIOD], out[COL_PERIOD], date_issues = _coerce_periods(frame[COL_PERIOD])
    if LABEL_COLUMN in frame.columns:
        out[LABEL_COLUMN] = pd.Series([clean_text(v, 200)[0] for v in frame[LABEL_COLUMN]], dtype=object)
    out[COL_AMOUNT] = _amount_display(frame[COL_AMOUNT], out[AMOUNT])
    out[TEXT_TRUNCATED] = truncated
    out, empty_rows = _drop_empty_rows(out, frame)
    out[AMOUNT_WAS_MISSING] = out[AMOUNT].isna().to_numpy()
    out[PERIOD_WAS_MISSING] = out[PERIOD].isna().to_numpy()
    issues.extend(_row_issues(out, empty_rows, date_issues))
    return _ordered(out), issues


def _canonical_doc_type(value: object) -> object:
    if value is None:
        return None
    return KNOWN_DOC_TYPES.get(str(value).replace(" ", "").lower(), value)


def _coerce_amounts(values: pd.Series) -> pd.Series:
    parsed = [parse_amount(v) for v in values.tolist()]
    return pd.Series([np.nan if p is None else p for p in parsed], dtype=np.float64)


def _amount_display(raw: pd.Series, parsed: pd.Series) -> pd.Series:
    display = []
    for raw_value, number in zip(raw.tolist(), parsed.tolist(), strict=True):
        if not np.isnan(number):
            display.append(f"{number:.2f}")
        else:
            display.append(clean_text(raw_value, 100)[0])
    return pd.Series(display, dtype=object)


def _coerce_periods(values: pd.Series) -> tuple[pd.Series, pd.Series, list[ValidationIssue]]:
    raw = values.tolist()
    first_parts = [p for p in (numeric_date_first_part(v) for v in raw) if p is not None]
    day_first = any(part > 12 for part in first_parts)
    parsed = [parse_date(v, day_first=day_first) for v in raw]
    periods = pd.Series([np.nan if p is None else float(p.period_index) for p in parsed])
    display = [
        p.label if p is not None else clean_text(v, 100)[0] for p, v in zip(parsed, raw, strict=True)
    ]
    issues = []
    if not day_first and any(is_ambiguous_numeric_date(v) for v in raw):
        issues.append(
            issue(
                Severity.INFO,
                "DATE_ORDER_ASSUMED_MONTH_FIRST",
                "Dates such as 03/04/2018 were read month first (March 4). No date in Col5 "
                "has a first part above 12, so the order could not be confirmed.",
                column=COL_PERIOD,
            )
        )
    return periods.astype(np.float64), pd.Series(display, dtype=object), issues


def _drop_empty_rows(out: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    input_cols = [c for c in INPUT_COLUMNS if c not in (COL_AMOUNT, COL_PERIOD)]
    text_empty = out[input_cols].isna().all(axis=1).to_numpy()
    raw_amount_empty = frame[COL_AMOUNT].map(lambda v: clean_text(v, 10)[0] is None).to_numpy()
    raw_period_empty = frame[COL_PERIOD].map(lambda v: clean_text(v, 10)[0] is None).to_numpy()
    empty = text_empty & raw_amount_empty & raw_period_empty
    dropped = out.loc[empty, SOURCE_ROW].tolist()
    kept = out.loc[~empty].reset_index(drop=True)
    if kept.empty:
        raise SchemaError(
            "The file has a header but no data rows. Add at least one row of Col1 to Col7 values.",
            code="NO_DATA_ROWS",
        )
    kept.insert(0, ROW_NUMBER, np.arange(1, len(kept) + 1))
    return kept, dropped


def _row_issues(
    out: pd.DataFrame, empty_rows: list[int], date_issues: list[ValidationIssue]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if empty_rows:
        issues.append(
            issue(
                Severity.WARNING,
                "EMPTY_ROWS_DROPPED",
                f"Dropped {len(empty_rows)} empty row(s). Spreadsheet rows: "
                + _row_list(empty_rows)
                + ".",
                rows=empty_rows,
            )
        )
    bad_amounts = out.loc[out[AMOUNT].isna(), SOURCE_ROW].tolist()
    if bad_amounts:
        issues.append(
            issue(
                Severity.WARNING,
                "AMOUNT_UNPARSEABLE",
                f"{len(bad_amounts)} amount(s) in Col3 were missing or unreadable and were "
                "replaced by the training median. Rows: " + _row_list(bad_amounts) + ".",
                column=COL_AMOUNT,
                rows=bad_amounts,
            )
        )
    bad_dates = out.loc[out[PERIOD].isna(), SOURCE_ROW].tolist()
    if bad_dates:
        issues.append(
            issue(
                Severity.WARNING,
                "DATE_UNPARSEABLE",
                f"{len(bad_dates)} date(s) in Col5 were missing or unreadable and were replaced "
                "by the training median period. Rows: " + _row_list(bad_dates) + ".",
                column=COL_PERIOD,
                rows=bad_dates,
            )
        )
    truncated = out.loc[out[TEXT_TRUNCATED], SOURCE_ROW].tolist()
    if truncated:
        issues.append(
            issue(
                Severity.INFO,
                "TEXT_TRUNCATED",
                f"{len(truncated)} row(s) had text longer than the limit; it was truncated.",
                rows=truncated,
            )
        )
    return [*issues, *date_issues]


def _row_list(rows: list[int], limit: int = 10) -> str:
    shown = ", ".join(str(r) for r in rows[:limit])
    return shown + (f" and {len(rows) - limit} more" if len(rows) > limit else "")


def _ordered(out: pd.DataFrame) -> pd.DataFrame:
    leading = [ROW_NUMBER, SOURCE_ROW, *INPUT_COLUMNS]
    trailing = [AMOUNT, PERIOD, AMOUNT_WAS_MISSING, PERIOD_WAS_MISSING, TEXT_TRUNCATED]
    label = [LABEL_COLUMN] if LABEL_COLUMN in out.columns else []
    return out[leading + label + trailing].reset_index(drop=True)
