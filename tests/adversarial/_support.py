"""Shared builders for adversarial tests: the validation path and tiny file factories."""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from pathlib import Path

import openpyxl
import pandas as pd

from ledgerclf.config import get_settings
from ledgerclf.io import read_table
from ledgerclf.schema import ValidationIssue, coerce_values, resolve_columns

HEADER = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7"]
ROW_A = ["Word16 Word17", "10719", "1200", "Word61", "2018-08-31", "Word563 Word575", "NoDoc"]
ROW_B = ["Word19 Word20", "65975", "75", "Word562", "2018-07-31", "Word563", "Doc1"]


def run_path(source: Path | bytes, file_name: str) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    """Run read_table, resolve_columns, and coerce_values. Return (frame, all issues)."""
    limits = get_settings().limits
    raw, _, read_issues = read_table(source, file_name, limits)
    resolved, _, resolve_issues = resolve_columns(raw, limits)
    frame, coerce_issues = coerce_values(resolved, limits)
    return frame, [*read_issues, *resolve_issues, *coerce_issues]


def csv_text(rows: Sequence[Sequence[str]], header: Sequence[str] | None = HEADER) -> str:
    """Join rows with commas and LF, no quoting (callers control the exact bytes)."""
    lines = [] if header is None else [",".join(header)]
    lines += [",".join(row) for row in rows]
    return "\n".join(lines) + "\n"


def with_value(row: Sequence[str], column: int, value: str) -> list[str]:
    """Return a copy of ``row`` with one cell replaced."""
    changed = list(row)
    changed[column] = value
    return changed


def xlsx_from(build: Callable[[openpyxl.Workbook], None]) -> bytes:
    """Build a workbook with ``build`` and return its bytes."""
    book = openpyxl.Workbook()
    build(book)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def codes(issues: Sequence[ValidationIssue]) -> list[str]:
    """Return the issue codes in order."""
    return [i.code for i in issues]
