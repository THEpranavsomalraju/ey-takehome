"""Row fingerprints: recognize uploaded rows the project already knows.

A fingerprint joins a row's inputs in model form (normalized text, amount in
cents, period index), so the same row matches whether it arrives as the
original workbook, a CSV re-save, or a copy pasted into another file. The
service uses it to keep holdout labels unread and to mark in-sample accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ledgerclf.features import normalize_text
from ledgerclf.schema import AMOUNT, PERIOD, TEXT_COLUMNS

FIELD_SEPARATOR = "\x1f"


def _number_texts(values: pd.Series, template: str) -> list[str]:
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return ["" if np.isnan(v) else template.format(v) for v in numbers]


def row_fingerprints(frame: pd.DataFrame) -> np.ndarray:
    """Return one fingerprint string per row of a coerced frame.

    Args:
        frame: Rows as returned by ``schema.coerce_values``.

    Returns:
        An object array of fingerprints in row order.
    """
    columns = [[normalize_text(v) for v in frame[column]] for column in TEXT_COLUMNS]
    columns.append(_number_texts(frame[AMOUNT], "{:.2f}"))
    columns.append(_number_texts(frame[PERIOD], "{:.0f}"))
    return np.asarray(
        [FIELD_SEPARATOR.join(parts) for parts in zip(*columns, strict=True)], dtype=object
    )


def matches(frame: pd.DataFrame, known: frozenset[str]) -> np.ndarray:
    """Return a boolean mask of the rows whose fingerprint is in ``known``."""
    if not known or frame.empty:
        return np.zeros(len(frame), dtype=bool)
    return np.isin(row_fingerprints(frame), sorted(known))
