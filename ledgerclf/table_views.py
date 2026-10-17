"""Dashboard views of a prediction result: filtered table pages and summary insights.

The dashboard never downloads a whole result: it asks for one page at a time,
already searched, filtered, and sorted here, so large uploads stay fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ledgerclf.jsonsafe import to_jsonable
from ledgerclf.labels import CANONICAL_CLASSES
from ledgerclf.predict import PredictionResult
from ledgerclf.schema import INPUT_COLUMNS, ROW_NUMBER

SortOrder = Literal["row", "confidence_asc", "confidence_desc"]
FLAG_SEPARATOR = "|"
EDGE_DECIMALS = 6  # histogram edges without float noise such as 0.15000000000000002


@dataclass(frozen=True)
class RowQuery:
    """What the results table currently shows.

    Attributes:
        search: Case-insensitive text matched against every input column.
        label: Only rows predicted as this category.
        review_only: Only rows that need a person's review.
        flag: Only rows carrying this flag code.
        sort: Row order.
    """

    search: str = ""
    label: str | None = None
    review_only: bool = False
    flag: str | None = None
    sort: SortOrder = "row"


def filtered_rows(predictions: pd.DataFrame, query: RowQuery) -> pd.DataFrame:
    """Return the prediction rows matching ``query``.

    Args:
        predictions: The prediction table of one upload.
        query: Search text, filters, and sort order.

    Returns:
        The matching rows, sorted as requested (ties keep file order).
    """
    keep = np.ones(len(predictions), dtype=bool)
    needle = query.search.strip().lower()
    if needle:
        # "string" dtype keeps missing cells empty (astype(str) would turn them into "None").
        inputs = predictions[list(INPUT_COLUMNS)].astype("string").fillna("")
        text = inputs.agg(" ".join, axis=1).str.lower()
        keep &= text.str.contains(needle, regex=False).to_numpy(dtype=bool)
    if query.label:
        keep &= (predictions["predicted_label"] == query.label).to_numpy()
    if query.review_only:
        keep &= predictions["review_required"].to_numpy(dtype=bool)
    if query.flag:
        keep &= np.array([query.flag in _flag_codes(f) for f in predictions["flags"]], dtype=bool)
    table = predictions[keep]
    if query.sort == "confidence_asc":
        return table.sort_values(["confidence", ROW_NUMBER], kind="stable")
    if query.sort == "confidence_desc":
        return table.sort_values(["confidence", ROW_NUMBER], ascending=[False, True], kind="stable")
    return table


def page(table: pd.DataFrame, offset: int, limit: int, total: int) -> dict[str, Any]:
    """Return one JSON-safe page of a filtered table.

    Args:
        table: Rows after filtering and sorting.
        offset: First row of the page (0-based, clamped to 0).
        limit: Page size (already clamped by the caller).
        total: Rows in the whole result before filtering.

    Returns:
        ``offset``, ``limit``, ``matched`` (rows after filtering), ``total``, ``rows``.
    """
    offset = max(0, offset)
    rows: list[dict[str, Any]] = to_jsonable(
        table.iloc[offset : offset + limit].to_dict(orient="records")
    )
    return {"offset": offset, "limit": limit, "matched": len(table), "total": total, "rows": rows}


def insights(result: PredictionResult, threshold: float, bins: int) -> dict[str, Any]:
    """Return the summary charts of one upload.

    Args:
        result: The prediction result.
        threshold: Review threshold of the active model.
        bins: Number of confidence bins between 0 and 1.

    Returns:
        A confidence histogram split into automatic and review rows, the rows
        per predicted category, and the rows per flag.
    """
    predictions = result.predictions
    confidence = predictions["confidence"].to_numpy(dtype=float)
    review = predictions["review_required"].to_numpy(dtype=bool)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    histogram = [
        {
            "low": round(float(edges[i]), EDGE_DECIMALS),
            "high": round(float(edges[i + 1]), EDGE_DECIMALS),
            "auto": int(((index == i) & ~review).sum()),
            "review": int(((index == i) & review).sum()),
        }
        for i in range(bins)
    ]
    by_class = [
        {
            "label": label,
            "rows": int((predictions["predicted_label"] == label).sum()),
            "review": int(((predictions["predicted_label"] == label) & review).sum()),
        }
        for label in CANONICAL_CLASSES
    ]
    return {
        "threshold": threshold,
        "confidence_histogram": histogram,
        "by_class": by_class,
        "flags": _flag_counts(predictions["flags"]),
        "mean_confidence": float(confidence.mean()) if len(confidence) else None,
    }


def _flag_counts(flags: pd.Series) -> list[dict[str, Any]]:
    """Rows per flag code, most frequent first, ties by code (stable across pandas versions)."""
    counts: dict[str, int] = {}
    for value in flags:
        for code in _flag_codes(value):
            counts[code] = counts.get(code, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"flag": code, "rows": rows} for code, rows in ordered]


def _flag_codes(value: object) -> list[str]:
    return [code for code in str(value or "").split(FLAG_SEPARATOR) if code]
