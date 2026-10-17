"""Label normalization: eleven raw spellings collapse to six canonical classes.

The same function serves training (every label must map) and uploads
(unmapped labels exclude the row from scoring only). See DECISIONS.md D01.
"""

from __future__ import annotations

import re

import pandas as pd

CANONICAL_CLASSES: tuple[str, ...] = (
    "Category_1",
    "Category_2",
    "Category_3",
    "Category_4",
    "Category_5",
    "Category_6",
)
FIVE_CLASSES: tuple[str, ...] = tuple(c for c in CANONICAL_CLASSES if c != "Category_5")
MAJORITY_CLASS = "Category_1"
COMMON_CLASSES: tuple[str, ...] = ("Category_1", "Category_2")
RARE_CLASSES: tuple[str, ...] = tuple(c for c in CANONICAL_CLASSES if c not in COMMON_CLASSES)

_LABEL_PATTERN = re.compile(r"^category(\d+)$", re.ASCII)


def normalize_label(raw: object) -> str | None:
    """Map one raw label to its canonical class.

    Rule: lowercase, strip, remove spaces and underscores, fix the typo
    ``categry``, require ``category<digits>``, map to ``Category_<digits>``.

    Args:
        raw: Any cell value from the label column.

    Returns:
        The canonical class name, or None when the value is missing or does
        not map to one of the six canonical classes.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, float) and pd.isna(raw):
        return None
    text = str(raw).strip().lower().replace(" ", "").replace("_", "")
    text = text.replace("categry", "category")
    match = _LABEL_PATTERN.match(text)
    if match is None:
        return None
    candidate = f"Category_{int(match.group(1))}"
    return candidate if candidate in CANONICAL_CLASSES else None


def normalize_labels(values: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Normalize a label column and report the raw-to-canonical mapping.

    Args:
        values: Raw label values.

    Returns:
        A tuple of (canonical labels as an object Series with None for
        unmapped values, mapping table with columns ``raw_label``,
        ``canonical_label``, ``rows`` sorted by canonical then raw label).
    """
    canonical = values.map(normalize_label).astype(object)
    raw_text = values.map(lambda v: "" if v is None or pd.isna(v) else str(v))
    table = (
        pd.DataFrame({"raw_label": raw_text, "canonical_label": canonical.fillna("")})
        .groupby(["raw_label", "canonical_label"], sort=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["canonical_label", "raw_label"], kind="stable")
        .reset_index(drop=True)
    )
    return canonical, table
