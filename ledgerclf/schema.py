"""Input schema: column names, column resolution, value coercion, validation report.

Every source (training file, holdout file, upload) passes through
``resolve_columns`` then ``coerce_values``, so training and inference see
identical cleaning.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from ledgerclf.config import Limits
from ledgerclf.exceptions import SchemaError
from ledgerclf.labels import normalize_label
from ledgerclf.values import (
    clean_text,
    date_order_evidence,
    is_ambiguous_numeric_date,
    is_blank_cell,
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
TEXT_COLUMNS: tuple[str, ...] = (
    COL_VENDOR,
    COL_INVOICE,
    COL_DESCRIPTION,
    COL_ACCOUNT,
    COL_DOC_TYPE,
)
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

# Common business names for each column, in normalized form (lowercase, no separators).
_BUSINESS_NAMES: dict[str, tuple[str, ...]] = {
    COL_VENDOR: ("vendor", "vendorname", "supplier", "suppliername", "payee"),
    COL_INVOICE: (
        "invoice",
        "invoiceno",
        "invoicenumber",
        "invoicenum",
        "invoiceid",
        "invno",
        "invoice#",
        "inv#",
    ),
    COL_AMOUNT: ("amount", "amt", "invoiceamount", "lineamount", "netamount", "value"),
    COL_DESCRIPTION: ("description", "desc", "linedescription", "itemdescription", "memo"),
    COL_PERIOD: ("date", "postingdate", "postdate", "period", "postingperiod", "invoicedate"),
    COL_ACCOUNT: ("glaccount", "gl", "glcode", "account", "accountname", "ledgeraccount"),
    COL_DOC_TYPE: ("doctype", "documenttype", "docflag", "documentflag", "doc", "document"),
}
_HEADER_ALIASES: dict[str, str] = {
    **{f"col{i}": name for i, name in enumerate(INPUT_COLUMNS, start=1)},
    **dict(zip("abcdefg", INPUT_COLUMNS, strict=True)),
    **{alias: column for column, aliases in _BUSINESS_NAMES.items() for alias in aliases},
    **{
        alias: LABEL_COLUMN
        for alias in ("classificationlabel", "label", "class", "category", "target", "h")
    },
}
_HEADER_STRIP = re.compile(r"[\s_\-.]+")
_INDEX_ARTIFACT = re.compile(r"^(unnamed:?\d*|index|)$")


class Severity(StrEnum):
    """Severity of a file-level validation issue."""

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
        return any(found.severity is Severity.ERROR for found in self.issues)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dict."""
        return {
            "issues": [found.to_dict() for found in self.issues],
            "rows_read": int(self.rows_read),
            "rows_dropped_empty": int(self.rows_dropped_empty),
            "rows_predicted": int(self.rows_predicted),
            "column_mapping": {str(k): str(v) for k, v in self.column_mapping.items()},
        }


def make_issue(
    severity: Severity,
    code: str,
    message: str,
    column: str | None = None,
    rows: Iterable[int] = (),
) -> ValidationIssue:
    """Build a ValidationIssue with sorted, de-duplicated row numbers.

    Args:
        severity: ERROR, WARNING, or INFO.
        code: Stable machine-readable code.
        message: Plain-English message, including the fix when one exists.
        column: Canonical column the issue concerns, if any.
        rows: Spreadsheet row numbers the issue concerns.

    Returns:
        The issue.
    """
    return ValidationIssue(severity, code, message, column, tuple(sorted({int(r) for r in rows})))


def normalize_header(value: object) -> str:
    """Normalize a header cell for matching: lowercase, no spaces or separators.

    Args:
        value: Raw header cell.

    Returns:
        The normalized text, or an empty string for blank cells.
    """
    text, _ = clean_text(value)
    return "" if text is None else _HEADER_STRIP.sub("", text.lower())


def canonical_header(value: object) -> str | None:
    """Return the canonical column name for a header cell.

    Args:
        value: Raw header cell.

    Returns:
        ``Col1`` to ``Col7`` or the label column name, or None when unknown.
    """
    return _HEADER_ALIASES.get(normalize_header(value))


def find_header_row(raw: pd.DataFrame, limits: Limits) -> int | None:
    """Find the header row within the first rows of a raw sheet.

    Args:
        raw: Sheet read with ``header=None``.
        limits: Provides the scan depth and the minimum number of matches.

    Returns:
        The 0-based row index of the first row where enough cells match
        known column names, or None when no row qualifies. Blank rows do not
        count toward the scan depth, so empty lines above a header are skipped.
    """
    scanned = 0
    for position in range(len(raw)):
        cells = raw.iloc[position].tolist()
        if all(is_blank_cell(v) for v in cells):
            continue
        names = {canonical_header(v) for v in cells} - {None}
        if len(names & set(INPUT_COLUMNS)) >= limits.header_min_matches:
            return position
        scanned += 1
        if scanned >= limits.header_scan_rows:
            break
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
        return _resolve_by_position(_trim_blank_edges(raw), limits)
    return _resolve_by_header(raw, header_row, limits)


def _is_blank(values: pd.Series) -> bool:
    return bool(values.map(is_blank_cell).all())


def _trim_blank_edges(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop blank leading rows and blank leading or trailing columns.

    Exports often start a block at C3 or end each line with a delimiter.
    Interior blank columns stay, so positional mapping never shifts a real
    column that happens to be empty. The row index (the 0-based sheet row)
    is kept, so spreadsheet row numbers stay true.
    """
    columns = [c for c in raw.columns if not _is_blank(raw[c])]
    if not columns:
        return raw.iloc[:, :0]
    first, last = list(raw.columns).index(columns[0]), list(raw.columns).index(columns[-1])
    trimmed = raw.iloc[:, first : last + 1]
    filled = ~trimmed.apply(lambda row: row.map(is_blank_cell).all(), axis=1)
    start = int(np.argmax(filled.to_numpy())) if filled.any() else len(trimmed)
    return trimmed.iloc[start:]


def _resolve_by_header(
    raw: pd.DataFrame, header_row: int, limits: Limits
) -> tuple[pd.DataFrame, dict[str, str], list[ValidationIssue]]:
    headers = raw.iloc[header_row].tolist()
    body = raw.iloc[header_row + 1 :]
    issues: list[ValidationIssue] = []
    chosen: dict[str, int] = {}
    extras: list[str] = []
    mapping: dict[str, str] = {}
    ragged_rows: set[int] = set()
    for position, header in enumerate(headers):
        name = canonical_header(header)
        label = clean_text(header, limits.header_max_chars)[0] or ""
        if name is None:
            values = body.iloc[:, position]
            if position == 0 and _is_index_artifact(label, values):
                continue
            if label == "" and not _is_blank(values):
                filled = ~values.map(is_blank_cell).to_numpy()
                ragged_rows.update((body.index[filled] + 1).tolist())
            elif label != "" and not _is_index_artifact(label, values):
                extras.append(label)
            continue
        if name in chosen:
            issues.append(
                make_issue(
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
            make_issue(
                Severity.WARNING,
                "EXTRA_COLUMNS_IGNORED",
                "Ignored columns not used by the model: " + _row_list(extras, limits) + ".",
            )
        )
    if ragged_rows:
        issues.append(_ragged_rows_issue(sorted(ragged_rows), limits))
    _require_columns(chosen, headers, limits)
    body, repeated = _drop_repeated_headers(body, chosen, limits)
    if repeated:
        issues.append(
            make_issue(
                Severity.WARNING,
                "REPEATED_HEADER_ROWS",
                "Skipped rows that repeat the header (files joined together): "
                + _row_list(repeated, limits)
                + ".",
                rows=repeated,
            )
        )
    frame = _assemble(body, chosen)
    return _with_optional(frame, issues), mapping, issues


def _ragged_rows_issue(rows: list[int], limits: Limits) -> ValidationIssue:
    return make_issue(
        Severity.WARNING,
        "ROW_HAS_EXTRA_FIELDS",
        "These rows have more fields than the header: "
        + _row_list(rows, limits)
        + ". A value probably contains an unquoted delimiter (for example a vendor written as "
        "Acme, Inc), so the values in these rows may sit in the wrong columns. Quote such "
        "values or remove the extra delimiter, then upload again.",
        rows=rows,
    )


def _drop_repeated_headers(
    body: pd.DataFrame, chosen: dict[str, int], limits: Limits
) -> tuple[pd.DataFrame, list[int]]:
    """Remove body rows that are copies of the header row, returning their row numbers."""
    probe = list(chosen.items())[:2]
    candidates = np.zeros(len(body), dtype=bool)
    for name, position in probe:
        candidates |= body.iloc[:, position].map(canonical_header).eq(name).to_numpy()
    if not candidates.any():
        return body, []
    repeated = np.zeros(len(body), dtype=bool)
    for index in np.flatnonzero(candidates):
        cells = body.iloc[index].tolist()
        matches = sum(canonical_header(cells[p]) == n for n, p in chosen.items())
        repeated[index] = matches >= limits.header_min_matches
    return body.loc[~repeated], (body.index[repeated] + 1).tolist()


def _resolve_by_position(
    raw: pd.DataFrame, limits: Limits
) -> tuple[pd.DataFrame, dict[str, str], list[ValidationIssue]]:
    width = raw.shape[1]
    if width == 0:
        raise SchemaError(
            "The file holds no data. Upload a file with a header row and at least one data row.",
            code="NO_DATA_ROWS",
        )
    if width not in (len(INPUT_COLUMNS), len(INPUT_COLUMNS) + 1):
        cells = [clean_text(v, limits.message_value_chars)[0] for v in raw.iloc[0].tolist()]
        first_row = ", ".join(c for c in cells if c is not None)
        raise SchemaError(
            "Could not find a header row naming the input columns Col1 to Col7 "
            f"(found in the first row: {first_row}). Add a header row with Col1 to Col7 "
            "(or A to G), or upload a file with exactly 7 input columns in that order.",
            code="MISSING_REQUIRED_COLUMNS",
        )
    header_like = _first_row_looks_like_header(raw, _positions(len(INPUT_COLUMNS)))
    first_row = ", ".join(str(clean_text(v, limits.message_value_chars)[0]) for v in raw.iloc[0])
    header_row = int(raw.index[0]) + 1
    if header_like:
        raw = raw.iloc[1:]
    with_label = width > len(INPUT_COLUMNS) and _looks_like_labels(raw.iloc[:, -1], limits)
    chosen = _positions(len(INPUT_COLUMNS) + int(with_label))
    _require_readable_positions(raw, chosen, first_row, limits)
    issues = [_positional_issue(header_row if header_like else None, first_row, width, with_label)]
    frame = _assemble(raw, chosen)
    mapping = {f"position {p + 1}": n for n, p in chosen.items()}
    return frame, mapping, issues


def _positions(count: int) -> dict[str, int]:
    """Columns in their documented order: Col1 to Col7, then the label."""
    return {name: position for position, name in enumerate([*INPUT_COLUMNS, LABEL_COLUMN][:count])}


def _looks_like_labels(values: pd.Series, limits: Limits) -> bool:
    """True when most filled cells of an extra column are category labels."""
    filled = [v for v in values if not is_blank_cell(v)]
    if not filled:
        return False
    labels = sum(normalize_label(v) is not None for v in filled)
    return labels / len(filled) >= limits.label_column_min_share


def _require_readable_positions(
    raw: pd.DataFrame, chosen: dict[str, int], first_row: str, limits: Limits
) -> None:
    """Refuse a positional mapping that would put the wrong data in the amount and date.

    Raises:
        SchemaError: When most rows have neither a readable amount nor a readable date.
    """
    amount_at, period_at = chosen[COL_AMOUNT], chosen[COL_PERIOD]
    readable = raw.apply(
        lambda row: (
            parse_amount(row.iloc[amount_at]) is not None
            or parse_date(row.iloc[period_at], day_first=False) is not None
        ),
        axis=1,
    )
    if len(raw) and readable.mean() < limits.positional_min_readable:
        raise SchemaError(
            "The column names were not recognized, and in the order given the 3rd column does "
            "not hold amounts and the 5th does not hold dates, so the columns cannot be matched "
            f"(first row: {first_row}). Name the columns Col1 to Col7 (or vendor, invoice "
            "number, amount, description, posting date, GL account, document flag).",
            code="MISSING_REQUIRED_COLUMNS",
        )


def _positional_issue(
    header_row: int | None, first_row: str, width: int, with_label: bool
) -> ValidationIssue:
    order = "Col1 to Col7 in order" + (", then the label" if with_label else "")
    extra = (
        " The extra last column did not hold category labels and was ignored."
        if (width > len(INPUT_COLUMNS) and not with_label)
        else ""
    )
    if header_row is None:
        message = f"No header row found, so columns were read by position ({order}).{extra}"
        return make_issue(Severity.WARNING, "POSITIONAL_MAPPING", message)
    message = (
        f"Row {header_row} looks like a header ({first_row}), but its names were not "
        f"recognized, so it was skipped and the columns were read by position ({order}).{extra} "
        "Name the columns Col1 to Col7 to map them by name instead."
    )
    return make_issue(Severity.WARNING, "HEADER_NOT_RECOGNIZED", message, rows=[header_row])


def _first_row_looks_like_header(raw: pd.DataFrame, chosen: dict[str, int]) -> bool:
    """Return True when row 1 has text where every later row has an amount and a date."""
    if len(raw) < 2:
        return False
    amount_at, period_at = chosen[COL_AMOUNT], chosen[COL_PERIOD]

    def parses(row: pd.Series) -> bool:
        has_amount = parse_amount(row.iloc[amount_at]) is not None
        return has_amount or parse_date(row.iloc[period_at], day_first=False) is not None

    first = raw.iloc[0]
    if parses(first) or any(is_blank_cell(v) for v in first):
        return False
    rest = raw.iloc[1:]
    return bool(rest.apply(parses, axis=1).mean() >= 0.5)


def _is_index_artifact(label: str, values: pd.Series) -> bool:
    if not _INDEX_ARTIFACT.match(normalize_header(label)):
        return False
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.isna().any() or len(numbers) == 0:
        return False
    start = numbers.iloc[0]
    return bool((numbers.to_numpy() == np.arange(start, start + len(numbers))).all())


def _assemble(body: pd.DataFrame, chosen: dict[str, int]) -> pd.DataFrame:
    """Select the chosen columns; ``source_row`` is the 1-based sheet row from the index."""
    data = {
        name: body.iloc[:, position].to_numpy(dtype=object) for name, position in chosen.items()
    }
    frame = pd.DataFrame(data, dtype=object)
    frame.insert(0, SOURCE_ROW, np.asarray(body.index, dtype=np.int64) + 1)
    return frame


def _require_columns(chosen: dict[str, int], headers: list[object], limits: Limits) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in chosen]
    if not missing:
        return
    cleaned = [clean_text(h, limits.header_max_chars)[0] for h in headers]
    found = [h for h in cleaned if h is not None]
    raise SchemaError(
        f"Missing required columns: {', '.join(missing)}. "
        f"Found columns: {', '.join(found) or 'none'}. Rename the columns to match "
        "Col1 to Col7, or upload a file with all seven input columns." + _lookalike_hint(found),
        code="MISSING_REQUIRED_COLUMNS",
    )


def _lookalike_hint(headers: list[str]) -> str:
    """Explain headers that contain non-ASCII characters resembling Latin letters."""
    hints = []
    for header in headers:
        odd = [ch for ch in header if not ch.isascii()]
        if odd:
            codes = ", ".join(f"'{ch}' (U+{ord(ch):04X})" for ch in odd)
            hints.append(f"Header '{header}' contains the non-ASCII character {codes}")
    if not hints:
        return ""
    return " " + "; ".join(hints) + ", which only looks like a Latin letter. Retype it."


def _with_optional(frame: pd.DataFrame, issues: list[ValidationIssue]) -> pd.DataFrame:
    if COL_INVOICE not in frame.columns:
        frame = frame.copy()
        frame.insert(1, COL_INVOICE, None)
        issues.append(
            make_issue(
                Severity.WARNING,
                "OPTIONAL_COLUMN_MISSING",
                "Column Col2 (invoice number) is missing. Predictions do not use Col2, "
                "so results are unaffected.",
                column=COL_INVOICE,
            )
        )
    return frame


def coerce_values(
    frame: pd.DataFrame, limits: Limits, day_first: bool | None = None
) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    """Clean every value of a resolved frame into typed columns.

    Coercion never imputes: missing or unparseable amounts and dates become
    NaN with a ``*_was_missing`` indicator, and the pipeline's training-fold
    imputer fills them identically in training and inference.

    Args:
        frame: Output of ``resolve_columns``.
        limits: Input limits (text length).
        day_first: Date order to use for Col5; None decides it from the column itself
            (an edited single row reuses the decision made for its whole upload).

    Returns:
        Tuple of (typed frame, issues). The frame holds ``row_number``,
        ``source_row``, display strings for Col1 to Col7 (and the raw label
        when present), ``amount``, ``period``, and indicator columns.

    Raises:
        SchemaError: When no non-empty data rows remain.
    """
    out = pd.DataFrame({SOURCE_ROW: frame[SOURCE_ROW].to_numpy(dtype=np.int64)})
    truncated_columns: dict[str, np.ndarray] = {}
    for column in TEXT_COLUMNS:
        cleaned = [clean_text(v, limits.max_text_chars) for v in frame[column].tolist()]
        out[column] = pd.Series([c[0] for c in cleaned], dtype=object)
        truncated_columns[column] = np.array([c[1] for c in cleaned], dtype=bool)
    out[COL_DOC_TYPE] = out[COL_DOC_TYPE].map(_canonical_doc_type).astype(object)
    out[AMOUNT] = _coerce_amounts(frame[COL_AMOUNT])
    out[PERIOD], out[COL_PERIOD], date_issues = _coerce_periods(
        frame[COL_PERIOD], limits, day_first
    )
    if LABEL_COLUMN in frame.columns:
        out[LABEL_COLUMN] = pd.Series(
            [clean_text(v, limits.header_max_chars)[0] for v in frame[LABEL_COLUMN]], dtype=object
        )
    out[COL_AMOUNT] = _amount_display(frame[COL_AMOUNT], out[AMOUNT], limits)
    out[TEXT_TRUNCATED] = np.any(list(truncated_columns.values()), axis=0)
    truncated_columns = {c: m for c, m in truncated_columns.items() if m.any()}
    out, empty_rows = _drop_empty_rows(out, frame)
    out[AMOUNT_WAS_MISSING] = out[AMOUNT].isna().to_numpy()
    out[PERIOD_WAS_MISSING] = out[PERIOD].isna().to_numpy()
    issues = _row_issues(out, empty_rows, limits)
    issues.extend(_truncation_issues(out, sorted(truncated_columns), limits))
    return _ordered(out), [*issues, *date_issues]


def _canonical_doc_type(value: object) -> object:
    if value is None:
        return None
    return KNOWN_DOC_TYPES.get(str(value).replace(" ", "").lower(), value)


def _coerce_amounts(values: pd.Series) -> pd.Series:
    parsed = [parse_amount(v) for v in values.tolist()]
    return pd.Series([np.nan if p is None else p for p in parsed], dtype=np.float64)


def _amount_display(raw: pd.Series, parsed: pd.Series, limits: Limits) -> pd.Series:
    display: list[str | None] = []
    for raw_value, number in zip(raw.tolist(), parsed.tolist(), strict=True):
        if not np.isnan(number):
            display.append(f"{number:.2f}")
        else:
            display.append(clean_text(raw_value, limits.max_text_chars)[0])
    return pd.Series(display, dtype=object)


def column_day_first(values: Iterable[object]) -> bool:
    """Decide the date order of a whole Col5 column: day first when more dates prove it.

    Args:
        values: Raw Col5 cells.

    Returns:
        True when day-first evidence (such as 31/08/2018) outweighs month-first evidence.
    """
    evidence = [date_order_evidence(v) for v in values]
    return evidence.count("day_first") > evidence.count("month_first")


def _coerce_periods(
    values: pd.Series, limits: Limits, day_first: bool | None
) -> tuple[pd.Series, pd.Series, list[ValidationIssue]]:
    raw = values.tolist()
    evidence = [date_order_evidence(v) for v in raw]
    day_votes, month_votes = evidence.count("day_first"), evidence.count("month_first")
    if day_first is None:
        day_first = column_day_first(raw)
    parsed = [parse_date(v, day_first=day_first) for v in raw]
    periods = pd.Series([np.nan if p is None else float(p.period_index) for p in parsed])
    display = [
        p.label if p is not None else clean_text(v, limits.max_text_chars)[0]
        for p, v in zip(parsed, raw, strict=True)
    ]
    issues: list[ValidationIssue] = []
    if day_votes and month_votes:
        issues.append(
            make_issue(
                Severity.WARNING,
                "DATE_ORDER_MIXED",
                f"Col5 mixes day-first dates ({day_votes} rows such as 31/08/2018) and "
                f"month-first dates ({month_votes} rows such as 08/31/2018). The column was read "
                f"{'day' if day_first else 'month'} first; check dates such as 03/04/2018.",
                column=COL_PERIOD,
            )
        )
    elif not day_first and any(is_ambiguous_numeric_date(v) for v in raw):
        issues.append(
            make_issue(
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
    raw_amount_empty = frame[COL_AMOUNT].map(is_blank_cell).to_numpy()
    raw_period_empty = frame[COL_PERIOD].map(is_blank_cell).to_numpy()
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


def _row_issues(out: pd.DataFrame, empty_rows: list[int], limits: Limits) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if empty_rows:
        issues.append(
            make_issue(
                Severity.WARNING,
                "EMPTY_ROWS_DROPPED",
                f"Dropped {len(empty_rows)} empty row(s). Spreadsheet rows: "
                + _row_list(empty_rows, limits)
                + ".",
                rows=empty_rows,
            )
        )
    bad_amounts = out.loc[out[AMOUNT].isna(), SOURCE_ROW].tolist()
    if bad_amounts:
        issues.append(
            make_issue(
                Severity.WARNING,
                "AMOUNT_UNPARSEABLE",
                f"{len(bad_amounts)} amount(s) in Col3 were missing or unreadable and were "
                "replaced by the training median. Rows: " + _row_list(bad_amounts, limits) + ".",
                column=COL_AMOUNT,
                rows=bad_amounts,
            )
        )
    bad_dates = out.loc[out[PERIOD].isna(), SOURCE_ROW].tolist()
    if bad_dates:
        issues.append(
            make_issue(
                Severity.WARNING,
                "DATE_UNPARSEABLE",
                f"{len(bad_dates)} date(s) in Col5 were missing or unreadable and were replaced "
                "by the training median period. Rows: " + _row_list(bad_dates, limits) + ".",
                column=COL_PERIOD,
                rows=bad_dates,
            )
        )
    return issues


def _truncation_issues(
    out: pd.DataFrame, columns: list[str], limits: Limits
) -> list[ValidationIssue]:
    truncated = out.loc[out[TEXT_TRUNCATED], SOURCE_ROW].tolist()
    if not truncated:
        return []
    message = (
        f"{len(truncated)} row(s) had text longer than {limits.max_text_chars} characters in "
        f"{', '.join(columns)}; only the first {limits.max_text_chars} characters were used. "
        "Rows: " + _row_list(truncated, limits) + "."
    )
    return [make_issue(Severity.INFO, "TEXT_TRUNCATED", message, rows=truncated)]


def _row_list(rows: Sequence[object], limits: Limits) -> str:
    shown_count = limits.issue_rows_shown
    shown = ", ".join(str(r) for r in rows[:shown_count])
    extra = len(rows) - shown_count
    return shown + (f" and {extra} more" if extra > 0 else "")


def _ordered(out: pd.DataFrame) -> pd.DataFrame:
    leading = [ROW_NUMBER, SOURCE_ROW, *INPUT_COLUMNS]
    trailing = [AMOUNT, PERIOD, AMOUNT_WAS_MISSING, PERIOD_WAS_MISSING, TEXT_TRUNCATED]
    label = [LABEL_COLUMN] if LABEL_COLUMN in out.columns else []
    return out[leading + label + trailing].reset_index(drop=True)
