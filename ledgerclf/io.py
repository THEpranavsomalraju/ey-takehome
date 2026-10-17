"""File reading for every supported format (Excel and CSV), plus file utilities.

The format comes from the file bytes first and the extension second, so a
CSV saved with an ``.xlsx`` name still reads. Every cell comes back as a raw
object; cleaning happens later in ``schema``. Blank lines stay in the frame
as empty rows so that every row keeps its true spreadsheet row number.
"""

from __future__ import annotations

import csv
import hashlib
import io as std_io
import logging
import re
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xlrd

from ledgerclf.config import UTF16_NUL_SHARE, UTF16_SAMPLE_BYTES, UTF16_TEXT_SHARE, Limits
from ledgerclf.exceptions import FileReadError, SchemaError
from ledgerclf.schema import (
    INPUT_COLUMNS,
    REQUIRED_COLUMNS,
    Severity,
    ValidationIssue,
    canonical_header,
    find_header_row,
    make_issue,
)
from ledgerclf.values import is_blank_cell

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".xls", ".xlsx", ".xlsm", ".csv", ".txt", ".tsv")
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZIP_MAGIC = b"PK\x03\x04"
_ENCRYPTED_STREAM = "EncryptedPackage".encode("utf-16-le")
_KNOWN_BINARY_MAGIC = (b"%PDF", b"\x89PNG", b"\xff\xd8\xff", b"GIF8")
_HTML_MARKERS = (b"<html", b"<table", b"<!doctype html")
_XLSB_PART = "xl/workbook.bin"
_DELIMITERS = (",", ";", "\t", "|")
# Share of NUL bytes above which content is binary rather than text with stray NULs.
_BINARY_NUL_SHARE = 0.1
_TEXT_ENCODINGS = ("utf-8", "cp1252")
_FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
_ZIP_CREATE_SYSTEM = 3  # zip "made by" field: Unix, on every platform
_FIXED_STAMP = b"2026-01-01T00:00:00Z"
_CORE_TIMESTAMPS = re.compile(rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*(</dcterms:)")


@dataclass(frozen=True)
class ReadInfo:
    """How a file was read: detected format, sheet, encoding, delimiter, header row."""

    file_name: str
    detected_format: str
    sheet_name: str | None
    encoding: str | None
    delimiter: str | None
    header_row_index: int | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dict."""
        return {
            "file_name": self.file_name,
            "detected_format": self.detected_format,
            "sheet_name": self.sheet_name,
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "header_row_index": self.header_row_index,
        }


def read_table(
    source: Path | str | bytes, file_name: str, limits: Limits
) -> tuple[pd.DataFrame, ReadInfo, list[ValidationIssue]]:
    """Read any supported file into a raw frame of object cells.

    Args:
        source: A path on disk (``Path`` or ``str``) or the uploaded bytes.
        file_name: Original file name, used for the extension and messages.
        limits: Size and row limits.

    Returns:
        Tuple of (raw frame read with ``header=None``, read info, issues).

    Raises:
        FileReadError: For missing, empty, oversized, unsupported, unreadable,
            or password-protected files, and for files above the row limit.
        SchemaError: When every sheet of a workbook is empty.
    """
    data = _load_bytes(source, file_name, limits)
    detected, issues = _detect_format(data, file_name, limits)
    sheet = encoding = delimiter = None
    if detected == "csv":
        raw, encoding, delimiter, csv_issues = _read_csv(data, limits)
        issues.extend(csv_issues)
    else:
        raw, sheet, sheet_issues = _read_excel(data, detected, file_name, limits)
        issues.extend(sheet_issues)
    header_row = find_header_row(raw, limits)
    _check_row_limit(raw, limits)
    info = ReadInfo(
        file_name=file_name,
        detected_format=detected,
        sheet_name=sheet,
        encoding=encoding,
        delimiter=delimiter,
        header_row_index=header_row,
    )
    return raw, info, issues


def _load_bytes(source: Path | str | bytes, file_name: str, limits: Limits) -> bytes:
    if isinstance(source, str):
        source = Path(source)
    if not isinstance(source, Path):
        data = bytes(source)
        _check_size(len(data), file_name, limits)
        return data
    try:
        _check_size(source.stat().st_size, file_name, limits)
        return source.read_bytes()
    except FileNotFoundError as error:
        raise FileReadError(
            f"The file '{file_name}' was not found at {source}. Check the path and try again.",
            code="FILE_NOT_FOUND",
        ) from error
    except PermissionError as error:
        raise FileReadError(
            f"The file '{file_name}' cannot be opened: permission denied. Close it in other "
            "programs (for example Excel) or copy it to a folder you can read, then try again.",
            code="FILE_PERMISSION_DENIED",
        ) from error
    except OSError as error:
        raise FileReadError(
            f"'{file_name}' could not be read as a file ({type(error).__name__}). Choose a "
            "single Excel or CSV file, not a folder or a device.",
            code="FILE_UNREADABLE",
        ) from error


def _check_size(size: int, file_name: str, limits: Limits) -> None:
    if size == 0:
        raise FileReadError(
            f"The file '{file_name}' is empty (0 bytes). Upload a file holding Col1 to Col7.",
            code="FILE_EMPTY",
        )
    if size > limits.max_file_bytes:
        raise FileReadError(
            f"The file '{file_name}' is {size / 1e6:.1f} MB, above the "
            f"{limits.max_file_bytes / 1e6:.0f} MB limit. Split the file into smaller parts.",
            code="FILE_TOO_LARGE",
        )


def _check_row_limit(raw: pd.DataFrame, limits: Limits) -> None:
    """Count non-empty rows only: Excel writes formatted empty rows as ``,,,,,,`` lines."""
    if len(raw) <= limits.max_rows:
        return
    data_rows = int((~_blank_mask(raw)).sum()) - 1  # minus the header row
    if data_rows > limits.max_rows:
        raise FileReadError(
            f"The file has {data_rows:,} data rows, above the {limits.max_rows:,} row limit. "
            "Split the file into smaller parts.",
            code="TOO_MANY_ROWS",
        )


def _blank_mask(raw: pd.DataFrame) -> np.ndarray:
    """Return True for rows whose every cell is missing or whitespace."""
    values = raw.to_numpy(dtype=object)
    missing = pd.isna(values)
    text_blank = np.char.strip(np.where(missing, "", values).astype(str)) == ""
    return np.asarray((missing | text_blank).all(axis=1))


def _unsupported(file_name: str, detail: str = "") -> FileReadError:
    return FileReadError(
        f"The file '{file_name}' is not a supported type{detail}. Supported types: "
        + ", ".join(SUPPORTED_EXTENSIONS)
        + ". Save the data as Excel (.xlsx) or CSV and upload again.",
        code="UNSUPPORTED_FORMAT",
    )


def _detect_format(
    data: bytes, file_name: str, limits: Limits
) -> tuple[str, list[ValidationIssue]]:
    extension = Path(file_name).suffix.lower()
    if data.startswith(_OLE_MAGIC):
        if _ENCRYPTED_STREAM in data:
            raise FileReadError(
                f"The workbook '{file_name}' is password protected. Remove the password "
                "in Excel (File > Info > Protect Workbook) and upload again.",
                code="PASSWORD_PROTECTED",
            )
        return "xls", _mismatch(extension, "xls", file_name)
    if data.startswith(_ZIP_MAGIC):
        archive = _xlsx_archive_kind(data)
        if archive == "xlsb":
            raise _unsupported(file_name, " (binary Excel workbook, .xlsb)")
        if archive != "xlsx":
            raise _unsupported(file_name)
        return "xlsx", _mismatch(extension, "xlsx", file_name)
    if _looks_like_html(data):
        raise _unsupported(file_name, " (a web page in HTML format, not a spreadsheet)")
    if _looks_binary(data, limits):
        raise _unsupported(file_name)
    if extension and extension not in SUPPORTED_EXTENSIONS:
        raise _unsupported(file_name)
    return "csv", _mismatch(extension, "csv", file_name)


def _looks_like_html(data: bytes) -> bool:
    head = data[:1024].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    return head.startswith(b"<") and any(marker in head for marker in _HTML_MARKERS)


def _looks_binary(data: bytes, limits: Limits) -> bool:
    """Return True for known binary formats and NUL-heavy content that is not UTF-16 text.

    A few NUL bytes (padding from some exporters) do not make a CSV binary;
    they are removed later with a warning.
    """
    if data.startswith(_KNOWN_BINARY_MAGIC):
        return True
    head = data[: limits.binary_sniff_bytes]
    nul_share = head.count(0) / max(len(head), 1)
    return nul_share > _BINARY_NUL_SHARE and _utf16_variant(data) is None


def _utf16_variant(data: bytes) -> str | None:
    """Return the UTF-16 codec for BOM-marked or BOM-less UTF-16 text, else None."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    sample = data[:UTF16_SAMPLE_BYTES]
    if len(sample) < 2:
        return None
    even_zero = sample[0::2].count(0) / len(sample[0::2])
    odd_zero = sample[1::2].count(0) / len(sample[1::2])
    if odd_zero > UTF16_NUL_SHARE and even_zero < UTF16_TEXT_SHARE:
        return "utf-16-le"
    if even_zero > UTF16_NUL_SHARE and odd_zero < UTF16_TEXT_SHARE:
        return "utf-16-be"
    return None


def _xlsx_archive_kind(data: bytes) -> str:
    """Return "xlsx", "xlsb", or "other" for a zip container."""
    try:
        with zipfile.ZipFile(std_io.BytesIO(data)) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return "xlsx"  # A truncated workbook: let the Excel reader report it as unreadable.
    if _XLSB_PART in names:
        return "xlsb"
    is_workbook = "[Content_Types].xml" in names and any(n.startswith("xl/") for n in names)
    return "xlsx" if is_workbook else "other"


def _mismatch(extension: str, detected: str, file_name: str) -> list[ValidationIssue]:
    matches = {
        "xls": (".xls",),
        "xlsx": (".xlsx", ".xlsm"),
        "csv": (".csv", ".txt", ".tsv", ""),
    }
    if extension in matches[detected]:
        return []
    return [
        make_issue(
            Severity.WARNING,
            "EXTENSION_MISMATCH",
            f"The file '{file_name}' has extension '{extension}' but its content is "
            f"{detected.upper()}. It was read as {detected.upper()}.",
        )
    ]


def _read_excel(
    data: bytes, detected: str, file_name: str, limits: Limits
) -> tuple[pd.DataFrame, str, list[ValidationIssue]]:
    engine: Literal["xlrd", "openpyxl"] = "xlrd" if detected == "xls" else "openpyxl"
    sheets, hidden, issues = _parse_sheets(data, engine, file_name)
    chosen = _choose_sheet(sheets, hidden, limits)
    if len(sheets) > 1:
        state = " (a hidden sheet: no visible sheet holds the data)" if chosen in hidden else ""
        issues.append(
            make_issue(
                Severity.INFO,
                "SHEET_SELECTED",
                f"The workbook has {len(sheets)} sheets. Used sheet '{chosen}'{state}.",
            )
        )
    return sheets[chosen], chosen, issues


def _parse_sheets(
    data: bytes, engine: Literal["xlrd", "openpyxl"], file_name: str
) -> tuple[dict[str, pd.DataFrame], set[str], list[ValidationIssue]]:
    unreadable = FileReadError(
        f"The file '{file_name}' could not be opened as an Excel workbook. It may be damaged "
        "or incomplete. Open it in Excel, save it again, and upload the new copy.",
        code="FILE_UNREADABLE",
    )
    engine_kwargs = {"logfile": _DebugLog()} if engine == "xlrd" else {}
    try:
        book = pd.ExcelFile(std_io.BytesIO(data), engine=engine, engine_kwargs=engine_kwargs)
    except Exception as error:  # boundary: parsers raise many error types on damaged files
        logger.debug("Excel open failed for %s", file_name, exc_info=True)
        if "encrypt" in str(error).lower():
            raise FileReadError(
                f"The workbook '{file_name}' is password protected. Remove the password in "
                "Excel (File > Info > Protect Workbook) and upload again.",
                code="PASSWORD_PROTECTED",
            ) from error
        raise unreadable from error
    sheets: dict[str, pd.DataFrame] = {}
    issues: list[ValidationIssue] = []
    hidden = _hidden_sheets(book)
    with book, warnings.catch_warnings():
        # openpyxl warns about unsupported features (data validation, styles); harmless here.
        warnings.simplefilter("ignore", UserWarning)
        for name in book.sheet_names:
            try:
                sheets[str(name)] = _without_trailing_blank_rows(
                    book.parse(name, header=None, dtype=object)
                )
            except Exception:  # boundary: one damaged sheet must not hide the others
                logger.debug("Skipping unreadable sheet %s", name, exc_info=True)
                issues.append(
                    make_issue(
                        Severity.WARNING,
                        "SHEET_UNREADABLE",
                        f"Sheet '{name}' could not be read and was skipped.",
                    )
                )
    if not sheets:
        raise unreadable
    return sheets, hidden, issues


class _DebugLog:
    """File-like sink that routes xlrd's console messages to the debug log (no prints)."""

    def write(self, text: str) -> int:
        """Log one chunk of xlrd output."""
        if text.strip():
            logger.debug("xlrd: %s", text.strip())
        return len(text)

    def flush(self) -> None:
        """Nothing is buffered."""


def _without_trailing_blank_rows(sheet: pd.DataFrame) -> pd.DataFrame:
    """Drop blank rows after the last filled row (a stray space far down the sheet)."""
    has_value = np.flatnonzero(sheet.notna().any(axis=1).to_numpy())
    for position in has_value[::-1]:
        if not sheet.iloc[position].map(is_blank_cell).all():
            return sheet.iloc[: position + 1]
    return sheet.iloc[:0]


def _hidden_sheets(book: pd.ExcelFile) -> set[str]:
    """Return names of hidden sheets (openpyxl ``sheet_state``, xlrd ``visibility``)."""
    hidden: set[str] = set()
    native = book.book
    for name in book.sheet_names:
        try:
            sheet = native[name] if hasattr(native, "sheetnames") else native.sheet_by_name(name)
        except (KeyError, IndexError, xlrd.XLRDError):
            continue
        state = getattr(sheet, "sheet_state", None)
        if state in ("hidden", "veryHidden") or getattr(sheet, "visibility", 0) != 0:
            hidden.add(str(name))
    return hidden


def _choose_sheet(sheets: dict[str, pd.DataFrame], hidden: set[str], limits: Limits) -> str:
    """Pick the first visible sheet holding the required header and data rows below it.

    Order of preference: visible sheets with header and data, hidden sheets with
    header and data, then the first non-empty sheet (column resolution then
    reports what is missing). A header-only template sheet never wins.
    """
    ordered = sorted(sheets, key=lambda name: name in hidden)  # stable: visible first
    first_non_empty = None
    for name in ordered:
        frame = sheets[name]
        if frame.dropna(how="all").empty:
            continue
        first_non_empty = first_non_empty or name
        header_row = find_header_row(frame, limits)
        if header_row is None or not _header_has_required(frame.iloc[header_row].tolist()):
            continue
        if not _blank_mask(frame.iloc[header_row + 1 :]).all():
            return name
    if first_non_empty is None:
        raise SchemaError(
            "Every sheet in the workbook is empty. Upload a workbook holding Col1 to Col7.",
            code="NO_DATA_ROWS",
        )
    return first_non_empty


def _header_has_required(cells: list[object]) -> bool:
    names = {canonical_header(c) for c in cells}
    return all(column in names for column in REQUIRED_COLUMNS)


def _read_csv(data: bytes, limits: Limits) -> tuple[pd.DataFrame, str, str, list[ValidationIssue]]:
    text, encoding, issues = _decode(data)
    if "\x00" in text:
        text = text.replace("\x00", "")
        issues.append(
            make_issue(
                Severity.WARNING,
                "NUL_BYTES_REMOVED",
                "Removed NUL characters from the file. Check that the file is a plain-text CSV.",
            )
        )
    delimiter = _sniff_delimiter(text, limits)
    rows, quote_issues = _parse_csv_records(text, delimiter, limits)
    issues.extend(quote_issues)
    while rows and not any(field.strip() for field in rows[-1]):
        rows.pop()  # trailing blank or delimiter-only lines carry no data
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        raise FileReadError(
            "The file holds no readable rows. Upload a CSV with a header row and data rows.",
            code="FILE_EMPTY",
        )
    # Interior blank lines stay as empty rows so that source row numbers stay true.
    padded = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(padded, dtype=object), encoding, delimiter, issues


def _parse_csv_records(
    text: str, delimiter: str, limits: Limits
) -> tuple[list[list[str]], list[ValidationIssue]]:
    """Parse CSV records, recovering from a quote that is never closed.

    A field holding both a line break and the delimiter means an opening
    quote swallowed the following rows. The file is then re-read line by
    line, so every physical row survives, and a warning names the row.
    """
    # Long text cells are truncated later (TEXT_TRUNCATED); the csv module's
    # default 128 KB field cap would otherwise raise on them.
    csv.field_size_limit(max(csv.field_size_limit(), limits.max_file_bytes))
    try:
        rows = list(csv.reader(std_io.StringIO(text, newline=""), delimiter=delimiter))
    except csv.Error as error:
        raise FileReadError(
            f"The CSV file could not be parsed ({error}). Open it in Excel and save it again "
            "as CSV, or upload the original Excel file.",
            code="FILE_UNREADABLE",
        ) from error
    swallowed = _swallowing_record(rows, delimiter)
    if swallowed is None:
        return rows, []
    lines = text.splitlines()
    rows = [next(csv.reader([line], delimiter=delimiter), []) for line in lines]
    warning = make_issue(
        Severity.WARNING,
        "UNBALANCED_QUOTE",
        f"A quote character opened in spreadsheet row {swallowed} is never closed, so the "
        "file was read one line per row instead. Values in that row after the quote may be "
        "merged; fix the quoting in the file to remove this warning.",
        rows=[swallowed],
    )
    return rows, [warning]


def _swallowing_record(rows: list[list[str]], delimiter: str) -> int | None:
    """Return the 1-based physical line of a record whose quoted field swallowed rows."""
    line = 1
    for row in rows:
        for field in row:
            if "\n" in field and delimiter in field:
                return line
        line += 1 + sum(field.count("\n") for field in row)
    return None


def _decode(data: bytes) -> tuple[str, str, list[ValidationIssue]]:
    if data.startswith(b"\xef\xbb\xbf"):
        try:
            return data[3:].decode("utf-8"), "utf-8-sig", []
        except UnicodeDecodeError:
            warning = make_issue(
                Severity.WARNING,
                "ENCODING_FALLBACK",
                "The file starts with a UTF-8 marker but its text is not UTF-8. It was read "
                "as Windows-1252; check accented characters in the results.",
            )
            return data[3:].decode("cp1252", errors="replace"), "cp1252", [warning]
    utf16 = _utf16_variant(data)
    if utf16 is not None:
        return _decode_utf16(data, utf16)
    for encoding in _TEXT_ENCODINGS:
        try:
            return data.decode(encoding), encoding, []
        except UnicodeDecodeError:
            continue
    warning = make_issue(
        Severity.WARNING,
        "ENCODING_FALLBACK",
        "The file is not UTF-8 or Windows-1252 text. It was read as Latin-1; check "
        "accented characters in the results.",
    )
    return data.decode("latin-1"), "latin-1", [warning]


def _decode_utf16(data: bytes, codec: str) -> tuple[str, str, list[ValidationIssue]]:
    """Decode UTF-16, ignoring an odd trailing byte and warning about any damaged character."""
    whole = data[: len(data) - len(data) % 2]
    text = whole.decode(codec, errors="replace").lstrip("\ufeff")
    if len(whole) == len(data) and "\ufffd" not in text:
        return text, codec, []
    warning = make_issue(
        Severity.WARNING,
        "ENCODING_FALLBACK",
        "Parts of this UTF-16 file are damaged (an incomplete or invalid character). They were "
        "skipped or shown as a replacement mark; check the affected values.",
    )
    return text.rstrip("\ufffd"), codec, [warning]


def _sniff_delimiter(text: str, limits: Limits) -> str:
    """Pick the delimiter giving the most consistent column count over the first lines.

    A deterministic scorer is used instead of ``csv.Sniffer``, which misreads
    files where one delimiter also appears inside quoted text.
    """
    sample = text[: limits.csv_sniff_bytes]
    best, best_score = ",", (0.0, 0)
    for delimiter in _DELIMITERS:
        rows = list(csv.reader(std_io.StringIO(sample, newline=""), delimiter=delimiter))
        widths = [len(r) for r in rows[: limits.csv_sniff_lines] if r]
        if not widths:
            continue
        mode = max(sorted(set(widths)), key=widths.count)
        consistency = widths.count(mode) / len(widths)
        score = (consistency if mode > 1 else 0.0, min(mode, len(INPUT_COLUMNS) + 1))
        if score > best_score:
            best, best_score = delimiter, score
    return best


def write_xlsx_deterministic(frame: pd.DataFrame, path: Path, sheet_name: str = "data") -> None:
    """Write an xlsx whose bytes depend only on its content.

    openpyxl stamps the save time into ``docProps/core.xml`` and every zip
    entry; both are pinned so that reruns produce byte-identical files.
    Entries are stored uncompressed: compressed bytes depend on the zlib
    build (Python 3.14 on Windows ships zlib-ng), stored bytes do not.

    Args:
        frame: Table to write (no index).
        path: Destination path.
        sheet_name: Worksheet name.
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name)
    with zipfile.ZipFile(path) as source:
        entries = [(info.filename, source.read(info.filename)) for info in source.infolist()]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as target:
        for name, payload in entries:
            if name == "docProps/core.xml":
                payload = _CORE_TIMESTAMPS.sub(rb"\g<1>" + _FIXED_STAMP + rb"\g<2>", payload)
            info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = _ZIP_CREATE_SYSTEM  # otherwise 0 on Windows, 3 elsewhere
            target.writestr(info, payload)


def sha256_text_file(path: Path) -> str:
    """Return the hex SHA-256 of a text file, reading Windows line endings as Unix ones.

    Git on Windows and some editors rewrite LF as CRLF without changing the
    content; the hash of a CSV must not depend on that.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file.

    Args:
        path: File to hash.

    Returns:
        The lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
