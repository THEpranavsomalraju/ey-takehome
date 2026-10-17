"""File reading for every supported format (docs/05 section 2.1 to 2.3).

The format comes from the file bytes first and the extension second, so a
CSV saved with an ``.xlsx`` name still reads. Every cell comes back as a raw
object; cleaning happens later in ``schema``.
"""

from __future__ import annotations

import csv
import io as std_io
import logging
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ledgerclf.config import Limits
from ledgerclf.exceptions import FileReadError, SchemaError
from ledgerclf.schema import (
    INPUT_COLUMNS,
    REQUIRED_COLUMNS,
    Severity,
    ValidationIssue,
    canonical_header,
    find_header_row,
    issue,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".xls", ".xlsx", ".xlsm", ".csv", ".txt", ".tsv")
_EXCEL_EXTENSIONS = (".xls", ".xlsx", ".xlsm")
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZIP_MAGIC = b"PK\x03\x04"
_ENCRYPTED_STREAM = "EncryptedPackage".encode("utf-16-le")
_KNOWN_BINARY_MAGIC = (b"%PDF", b"\x89PNG", b"\xff\xd8\xff", b"GIF8")
_DELIMITERS = (",", ";", "\t", "|")
_TEXT_ENCODINGS = ("utf-8", "cp1252")


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
    source: Path | bytes, file_name: str, limits: Limits
) -> tuple[pd.DataFrame, ReadInfo, list[ValidationIssue]]:
    """Read any supported file into a raw frame of object cells.

    Args:
        source: A path on disk or the uploaded bytes.
        file_name: Original file name, used for the extension and messages.
        limits: Size and row limits.

    Returns:
        Tuple of (raw frame read with ``header=None``, read info, issues).

    Raises:
        FileReadError: For empty, oversized, unsupported, unreadable, or
            password-protected files.
    """
    data = _load_bytes(source, file_name, limits)
    detected, issues = _detect_format(data, file_name)
    if detected == "csv":
        raw, encoding, delimiter, csv_issues = _read_csv(data, limits)
        issues.extend(csv_issues)
        info = ReadInfo(file_name, "csv", None, encoding, delimiter, find_header_row(raw, limits))
    else:
        raw, sheet, sheet_issues = _read_excel(data, detected, file_name, limits)
        issues.extend(sheet_issues)
        info = ReadInfo(file_name, detected, sheet, None, None, find_header_row(raw, limits))
    _check_row_limit(raw, limits)
    return raw, info, issues


def _load_bytes(source: Path | bytes, file_name: str, limits: Limits) -> bytes:
    if isinstance(source, Path):
        size = source.stat().st_size
        _check_size(size, file_name, limits)
        data = source.read_bytes()
    else:
        data = bytes(source)
        _check_size(len(data), file_name, limits)
    return data


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
    if len(raw) > limits.max_rows + limits.header_scan_rows:
        raise FileReadError(
            f"The file has {len(raw):,} rows, above the {limits.max_rows:,} row limit. "
            "Split the file into smaller parts.",
            code="TOO_MANY_ROWS",
        )


def _unsupported(file_name: str) -> FileReadError:
    return FileReadError(
        f"The file '{file_name}' is not a supported type. Supported types: "
        + ", ".join(SUPPORTED_EXTENSIONS)
        + ". Save the data as Excel or CSV and upload again.",
        code="UNSUPPORTED_FORMAT",
    )


def _detect_format(data: bytes, file_name: str) -> tuple[str, list[ValidationIssue]]:
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
        if not _is_xlsx_archive(data):
            raise _unsupported(file_name)
        return "xlsx", _mismatch(extension, "xlsx", file_name)
    if _looks_binary(data):
        raise _unsupported(file_name)
    if extension and extension not in SUPPORTED_EXTENSIONS:
        raise _unsupported(file_name)
    return "csv", _mismatch(extension, "csv", file_name)


def _looks_binary(data: bytes) -> bool:
    """Return True for known binary formats and NUL-heavy content without a UTF-16 BOM."""
    if data.startswith(_KNOWN_BINARY_MAGIC):
        return True
    return b"\x00\x00\x00" in data[:4096] and not _has_utf16_bom(data)


def _has_utf16_bom(data: bytes) -> bool:
    return data.startswith((b"\xff\xfe", b"\xfe\xff"))


def _is_xlsx_archive(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(std_io.BytesIO(data)) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return True  # A truncated workbook: let the Excel reader report it as unreadable.
    return "[Content_Types].xml" in names and any(n.startswith("xl/") for n in names)


def _mismatch(extension: str, detected: str, file_name: str) -> list[ValidationIssue]:
    matches = {
        "xls": (".xls",),
        "xlsx": (".xlsx", ".xlsm"),
        "csv": (".csv", ".txt", ".tsv", ""),
    }
    if extension in matches[detected]:
        return []
    return [
        issue(
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
    sheets = _parse_sheets(data, engine, file_name)
    issues: list[ValidationIssue] = []
    chosen = _choose_sheet(sheets, limits)
    if len(sheets) > 1:
        issues.append(
            issue(
                Severity.INFO,
                "SHEET_SELECTED",
                f"The workbook has {len(sheets)} sheets. Used sheet '{chosen}'.",
            )
        )
    return sheets[chosen], chosen, issues


def _parse_sheets(
    data: bytes, engine: Literal["xlrd", "openpyxl"], file_name: str
) -> dict[str, pd.DataFrame]:
    unreadable = FileReadError(
        f"The file '{file_name}' could not be opened as an Excel workbook. It may be damaged "
        "or incomplete. Open it in Excel, save it again, and upload the new copy.",
        code="FILE_UNREADABLE",
    )
    try:
        book = pd.ExcelFile(std_io.BytesIO(data), engine=engine)
    except Exception as error:
        logger.debug("Excel open failed for %s", file_name, exc_info=True)
        raise unreadable from error
    sheets: dict[str, pd.DataFrame] = {}
    with book, warnings.catch_warnings():
        # openpyxl warns about unsupported features (data validation, styles); harmless here.
        warnings.simplefilter("ignore", UserWarning)
        for name in book.sheet_names:
            try:
                sheets[str(name)] = book.parse(name, header=None, dtype=object)
            except Exception:
                logger.debug("Skipping unreadable sheet %s", name, exc_info=True)
    if not sheets:
        raise unreadable
    return sheets


def _choose_sheet(sheets: dict[str, pd.DataFrame], limits: Limits) -> str:
    first_non_empty = None
    for name, frame in sheets.items():
        frame_without_blanks = frame.dropna(how="all")
        if frame_without_blanks.empty:
            continue
        first_non_empty = first_non_empty or name
        header_row = find_header_row(frame, limits)
        if header_row is not None and _header_has_required(frame.iloc[header_row].tolist()):
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
            issue(Severity.WARNING, "NUL_BYTES_REMOVED", "Removed NUL characters from the file.")
        )
    delimiter = _sniff_delimiter(text, limits)
    rows = list(csv.reader(std_io.StringIO(text, newline=""), delimiter=delimiter))
    rows = [row for row in rows if row]  # csv yields [] for blank physical lines only
    width = max((len(row) for row in rows), default=0)
    padded = [row + [""] * (width - len(row)) for row in rows]
    frame = pd.DataFrame(padded, dtype=object)
    if frame.empty:
        raise FileReadError(
            "The file holds no readable rows. Upload a CSV with a header row and data rows.",
            code="FILE_EMPTY",
        )
    return frame, encoding, delimiter, issues


def _decode(data: bytes) -> tuple[str, str, list[ValidationIssue]]:
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace"), "utf-8-sig", []
    if _has_utf16_bom(data):
        return data.decode("utf-16", errors="replace"), "utf-16", []
    for encoding in _TEXT_ENCODINGS:
        try:
            return data.decode(encoding), encoding, []
        except UnicodeDecodeError:
            continue
    warning = issue(
        Severity.WARNING,
        "ENCODING_FALLBACK",
        "The file is not UTF-8 or Windows-1252 text. It was read as Latin-1; check "
        "accented characters in the results.",
    )
    return data.decode("latin-1"), "latin-1", [warning]


def _sniff_delimiter(text: str, limits: Limits) -> str:
    """Pick the delimiter giving the most consistent column count over the first lines.

    A deterministic scorer is used instead of ``csv.Sniffer`` alone, which
    misreads files where one delimiter also appears inside quoted text.
    """
    sample = text[: limits.csv_sniff_bytes]
    best, best_score = ",", (0.0, 0)
    for delimiter in _DELIMITERS:
        rows = list(csv.reader(std_io.StringIO(sample, newline=""), delimiter=delimiter))
        widths = [len(r) for r in rows[: limits.csv_sniff_lines] if r]
        if not widths:
            continue
        mode = max(set(widths), key=widths.count)
        consistency = widths.count(mode) / len(widths)
        score = (consistency if mode > 1 else 0.0, min(mode, len(INPUT_COLUMNS) + 1))
        if score > best_score:
            best, best_score = delimiter, score
    return best
