"""Exploratory analysis tables (docs/04 section 15, ``reports/eda``).

Pure functions return DataFrames or dicts. Label-free structure (schema
audit, Col2 type mix and patterns) uses the full raw file; everything that
touches labels uses the training portion only, so the holdout stays unseen.
``run_eda`` loads the data, calls each function, and writes the reports.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from ledgerclf.config import Settings
from ledgerclf.io import read_table
from ledgerclf.labels import CANONICAL_CLASSES, normalize_labels
from ledgerclf.reports import write_json, write_table
from ledgerclf.schema import (
    AMOUNT,
    COL_ACCOUNT,
    COL_AMOUNT,
    COL_DESCRIPTION,
    COL_DOC_TYPE,
    COL_INVOICE,
    COL_PERIOD,
    COL_VENDOR,
    INPUT_COLUMNS,
    LABEL_COLUMN,
    MISSING_TOKEN,
    PERIOD,
    TOKEN_COLUMNS,
    resolve_columns,
)
from ledgerclf.splitting import ROW_ID, TrainingData, load_training_frame
from ledgerclf.values import is_missing, period_label

logger = logging.getLogger(__name__)

PRODUCER = "eda"
TOP_VALUES = 20
HISTOGRAM_BINS = 30
IQR_MULTIPLIER = 1.5
AMOUNT_DECILES = 10
SCIENTIFIC_STYLE = re.compile(r"^\d\.\d+[A-Za-z][+-]\d+$")
INFERRED_ROLES = {
    COL_VENDOR: "Vendor name",
    COL_INVOICE: "Invoice number",
    COL_AMOUNT: "Amount",
    COL_DESCRIPTION: "Line description",
    COL_PERIOD: "Posting period",
    COL_ACCOUNT: "GL account name",
    COL_DOC_TYPE: "Document flag",
    LABEL_COLUMN: "Target",
}


def signed_log(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return ``sign(x) * log1p(abs(x))``, the scale used for amounts."""
    array = np.asarray(values, dtype=float)
    return np.asarray(np.sign(array) * np.log1p(np.abs(array)))


def _text(series: pd.Series) -> pd.Series:
    return series.fillna(MISSING_TOKEN).astype(str)


def _class_table(keys: pd.Series, y: np.ndarray, key_name: str) -> pd.DataFrame:
    table = pd.crosstab(keys.to_numpy(), y).reindex(columns=list(CANONICAL_CLASSES), fill_value=0)
    table.index.name = key_name
    table.columns.name = None
    table["total"] = table.sum(axis=1)
    return table.reset_index()


def schema_summary(resolved: pd.DataFrame) -> dict[str, Any]:
    """Audit every column of the full raw file: Python types, uniques, missing values."""
    columns = {}
    for column in [*INPUT_COLUMNS, LABEL_COLUMN]:
        values = resolved[column]
        present = values[~values.map(is_missing)]
        columns[column] = {
            "python_types": dict(Counter(type(v).__name__ for v in present).most_common()),
            "unique": int(present.astype(str).nunique()),
            "missing": int(len(values) - len(present)),
            "inferred_role": INFERRED_ROLES[column],
        }
    return {
        "rows": len(resolved),
        "columns": [*INPUT_COLUMNS, LABEL_COLUMN],
        "column_details": columns,
        "note": "Roles are working hypotheses from value patterns; "
        "the file names only Col1 to Col7.",
    }


def raw_label_mapping(labels: pd.Series) -> pd.DataFrame:
    """Return the raw-to-canonical label table for the full file."""
    return normalize_labels(labels)[1]


def class_distribution(y: np.ndarray) -> pd.DataFrame:
    """Return rows and share per canonical class."""
    counts = pd.Series(y).value_counts().reindex(list(CANONICAL_CLASSES), fill_value=0)
    return pd.DataFrame(
        {"class": counts.index, "rows": counts.to_numpy(), "share": counts / len(y)}
    )


def missing_values(frame: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Return missing counts per input column, overall and per class."""
    rows = []
    for column in INPUT_COLUMNS:
        missing = _missing_mask(frame, column)
        row = {
            "column": column,
            "missing_rows": int(missing.sum()),
            "missing_share": missing.mean(),
        }
        for cls in CANONICAL_CLASSES:
            row[f"missing_{cls}"] = int((missing & (y == cls)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _missing_mask(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column == COL_AMOUNT:
        return frame[AMOUNT].isna().to_numpy()
    if column == COL_PERIOD:
        return frame[PERIOD].isna().to_numpy()
    return frame[column].isna().to_numpy()


def col2_type_mix(invoices: pd.Series) -> pd.DataFrame:
    """Return Python type counts of raw Col2 values, plus scientific-style strings."""
    present = invoices[~invoices.map(is_missing)]
    counts = Counter(type(v).__name__ for v in present)
    scientific = int(present.map(lambda v: bool(SCIENTIFIC_STYLE.match(str(v)))).sum())
    rows = [{"category": name, "rows": n} for name, n in counts.most_common()]
    rows.append({"category": "scientific-style text (subset of str)", "rows": scientific})
    return pd.DataFrame(rows)


def _shape(value: object) -> str:
    text = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", text))


def col2_patterns(invoices: pd.Series) -> pd.DataFrame:
    """Return the most common Col2 format shapes (digits to 9, letters to A)."""
    present = invoices[~invoices.map(is_missing)]
    shapes = present.map(_shape)
    table = shapes.value_counts().head(TOP_VALUES)
    examples = {shape: str(present[shapes == shape].iloc[0]) for shape in table.index}
    return pd.DataFrame(
        {
            "pattern": table.index,
            "rows": table.to_numpy(),
            "example": [examples[s] for s in table.index],
        }
    )


def amount_summary(amounts: pd.Series) -> pd.DataFrame:
    """Return headline statistics of the amount column."""
    values = amounts.dropna()
    stats = {
        "rows": len(amounts),
        "missing": int(amounts.isna().sum()),
        "min": values.min(),
        "p25": values.quantile(0.25),
        "median": values.median(),
        "mean": values.mean(),
        "p75": values.quantile(0.75),
        "max": values.max(),
        "negative_rows": int((values < 0).sum()),
        "zero_rows": int((values == 0).sum()),
        "skewness": values.skew(),
    }
    return pd.DataFrame({"statistic": list(stats), "value": list(stats.values())})


def amount_by_class(amounts: pd.Series, y: np.ndarray) -> pd.DataFrame:
    """Return count, median, mean, min, max of amounts per class."""
    grouped = pd.DataFrame({"class": y, "amount": amounts.to_numpy()}).groupby("class")["amount"]
    table = grouped.agg(["count", "median", "mean", "min", "max"])
    return table.reindex(list(CANONICAL_CLASSES)).reset_index()


def amount_histogram(amounts: pd.Series, y: np.ndarray) -> pd.DataFrame:
    """Return per-class counts over equal-width bins of the signed-log amount."""
    scaled = signed_log(amounts.fillna(amounts.median()))
    edges = np.linspace(scaled.min(), scaled.max(), HISTOGRAM_BINS + 1)
    table = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:]})
    for cls in CANONICAL_CLASSES:
        table[cls] = np.histogram(scaled[y == cls], bins=edges)[0]
    return table


def amount_outliers(frame: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Return rows outside the IQR fences of the signed-log amount, with their class."""
    scaled = signed_log(frame[AMOUNT])
    q1, q3 = np.nanpercentile(scaled, [25, 75])
    low, high = q1 - IQR_MULTIPLIER * (q3 - q1), q3 + IQR_MULTIPLIER * (q3 - q1)
    mask = (scaled < low) | (scaled > high)
    return pd.DataFrame(
        {
            ROW_ID: frame.loc[mask, ROW_ID].to_numpy(),
            "amount": frame.loc[mask, AMOUNT].to_numpy(),
            "signed_log_amount": scaled[mask],
            "side": np.where(scaled[mask] < low, "low", "high"),
            "fence_low": low,
            "fence_high": high,
            "class": y[mask],
        }
    )


def period_by_class(frame: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Return row counts per monthly period and class, in time order."""
    labels = frame[PERIOD].map(lambda p: "missing" if pd.isna(p) else period_label(int(p)))
    return _class_table(labels, y, "period").sort_values("period").reset_index(drop=True)


def day_of_month(raw_periods: pd.Series) -> pd.DataFrame:
    """Return the day-of-month distribution of raw Col5 dates."""
    days = pd.to_datetime(raw_periods, errors="coerce").dt.day.dropna().astype(int)
    counts = days.value_counts().sort_index()
    return pd.DataFrame({"day": counts.index, "rows": counts.to_numpy()})


def col7_by_class(frame: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Return the document flag by class table."""
    return _class_table(_text(frame[COL_DOC_TYPE]), y, COL_DOC_TYPE)


def _tokens(series: pd.Series) -> list[list[str]]:
    return [str(v).split() for v in series.dropna()]


def text_cardinality(frame: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Return cardinality, vocabulary, token counts, singletons, and purity per text column."""
    rows = []
    for column in TOKEN_COLUMNS:
        values = frame[column]
        tokens = _tokens(values)
        counts = values.dropna().value_counts()
        purity_table = pd.crosstab(_text(values).to_numpy(), y)
        rows.append(
            {
                "column": column,
                "unique_values": int(values.nunique()),
                "missing_rows": int(values.isna().sum()),
                "token_vocabulary": len({t for row in tokens for t in row}),
                "mean_tokens": float(np.mean([len(r) for r in tokens])),
                "max_tokens": max(len(r) for r in tokens),
                "values_seen_once": int((counts == 1).sum()),
                "in_sample_majority_purity": purity_table.max(axis=1).sum() / len(y),
            }
        )
    return pd.DataFrame(rows)


def top_values(frame: pd.DataFrame, y: np.ndarray, column: str) -> pd.DataFrame:
    """Return the most frequent values of a text column with their class counts."""
    table = _class_table(_text(frame[column]), y, "value")
    return table.sort_values(["total", "value"], ascending=[False, True]).head(TOP_VALUES)


def token_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    """Return how many tokens each pair of text columns shares."""
    vocab = {c: {t for row in _tokens(frame[c]) for t in row} for c in TOKEN_COLUMNS}
    rows = []
    for i, first in enumerate(TOKEN_COLUMNS):
        for second in TOKEN_COLUMNS[i + 1 :]:
            shared = vocab[first] & vocab[second]
            example = sorted(shared)[0] if shared else ""
            rows.append(
                {
                    "column_a": first,
                    "column_b": second,
                    "shared_tokens": len(shared),
                    "example": example,
                }
            )
    return pd.DataFrame(rows)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Return the bias-corrected Cramer's V between two categorical series (Bergsma 2013)."""
    table = pd.crosstab(a.to_numpy(), b.to_numpy())
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr, k_corr = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denominator = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2 / denominator)) if denominator > 0 else 0.0


def categorical_views(frame: pd.DataFrame, y: np.ndarray) -> dict[str, pd.Series]:
    """Return the categorical variables used for Cramer's V and mutual information."""
    amounts = frame[AMOUNT].fillna(frame[AMOUNT].median())
    deciles = pd.qcut(amounts.rank(method="first"), AMOUNT_DECILES, labels=False)
    return {
        "Col1 vendor": _text(frame[COL_VENDOR]).str.lower(),
        "Col4 description": _text(frame[COL_DESCRIPTION]).str.lower(),
        "Col6 account": _text(frame[COL_ACCOUNT]).str.lower(),
        "Col1+Col6 pair": _pair_key(frame),
        "Col7 doc type": _text(frame[COL_DOC_TYPE]),
        "Col5 period": frame[PERIOD].fillna(-1).astype(int).astype(str),
        "Col3 amount decile": deciles.astype(str),
        "label": pd.Series(y, index=frame.index, dtype=object),
    }


def _pair_key(frame: pd.DataFrame) -> pd.Series:
    return _text(frame[COL_VENDOR]).str.lower() + " || " + _text(frame[COL_ACCOUNT]).str.lower()


def cramers_v_matrix(frame: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Return the symmetric bias-corrected Cramer's V matrix among the categorical views."""
    views = {k: v for k, v in categorical_views(frame, y).items() if k != "Col4 description"}
    names = list(views)
    matrix = pd.DataFrame(1.0, index=names, columns=names)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            value = cramers_v(views[first], views[second])
            matrix.loc[first, second] = matrix.loc[second, first] = value
    matrix.index.name = "variable"
    return matrix.reset_index()


def mutual_information(frame: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Return mutual information of each column with the label, in nats and normalized."""
    views = categorical_views(frame, y)
    label = views.pop("label")
    rows = [
        {
            "variable": name,
            "categories": int(series.nunique()),
            "mutual_information_nats": mutual_info_score(label, series),
            "normalized_mutual_information": normalized_mutual_info_score(label, series),
        }
        for name, series in views.items()
    ]
    return pd.DataFrame(rows).sort_values("mutual_information_nats", ascending=False)


def _key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].fillna(MISSING_TOKEN).astype(str).agg("\x1f".join, axis=1)


def duplicates_summary(frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """Summarize exact duplicates and label conflicts on the training rows."""
    all_inputs = _key(frame, list(INPUT_COLUMNS))
    group_sizes = pd.Series(groups).value_counts()
    labels_per_group = pd.Series(y).groupby(groups).nunique()
    return {
        "rows": len(frame),
        "duplicate_rows_all_inputs": int(all_inputs.duplicated().sum()),
        "duplicate_rows_group_key": int(len(groups) - len(group_sizes)),
        "groups": len(group_sizes),
        "groups_with_copies": int((group_sizes > 1).sum()),
        "rows_in_groups_with_copies": int(group_sizes[group_sizes > 1].sum()),
        "label_conflict_groups": int((labels_per_group > 1).sum()),
        "group_key": "Col1, Col3, Col4, Col5 (monthly period), Col6, Col7; "
        "Col2 excluded (D02, D22)",
    }


def duplicate_group_sizes(groups: np.ndarray) -> pd.DataFrame:
    """Return how many groups have each size."""
    sizes = pd.Series(groups).value_counts().value_counts().sort_index()
    return pd.DataFrame({"group_size": sizes.index, "groups": sizes.to_numpy()})


def conflict_groups(frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Return group-key groups whose identical rows carry different labels."""
    labels = pd.Series(y)
    mixed = labels.groupby(groups).nunique()
    rows = []
    for group in sorted(mixed[mixed > 1].index):
        members = np.flatnonzero(groups == group)
        counts = labels.iloc[members].value_counts()
        first = frame.iloc[members[0]]
        row = {"group_id": int(group), "rows": len(members)}
        row.update({c: first[c] for c in INPUT_COLUMNS if c != COL_INVOICE})
        row.update({cls: int(counts.get(cls, 0)) for cls in CANONICAL_CLASSES})
        rows.append(row)
    return pd.DataFrame(rows)


def rare_class_rows(frame: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Return every training row of the rare classes (Category_3 to Category_6)."""
    rare = ~np.isin(y, ["Category_1", "Category_2"])
    columns = [ROW_ID, *INPUT_COLUMNS]
    table = frame.loc[rare, columns].copy()
    table.insert(1, "class", y[rare])
    return table.sort_values(["class", COL_VENDOR, ROW_ID]).reset_index(drop=True)


def accuracy_ceiling(frame: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    """Return the best accuracy any model could reach on identical-input conflicts."""
    result: dict[str, Any] = {}
    for name, columns in {
        "text_only": [COL_VENDOR, COL_DESCRIPTION, COL_ACCOUNT],
        "text_amount_doc": [COL_VENDOR, COL_DESCRIPTION, COL_ACCOUNT, COL_AMOUNT, COL_DOC_TYPE],
    }.items():
        key = _key(frame, columns)
        table = pd.crosstab(key.to_numpy(), y)
        mixed = (table > 0).sum(axis=1) > 1
        result[name] = {
            "key_columns": columns,
            "mixed_groups": int(mixed.sum()),
            "rows_in_mixed_groups": int(table[mixed].to_numpy().sum()),
            "ceiling_accuracy": float(table.max(axis=1).sum() / len(y)),
        }
    return result


EDA_CATALOG: dict[str, tuple[str, str]] = {
    "eda/schema_summary.json": (
        "Schema audit (full file)",
        "Python types, uniques, missing values, and inferred role per column of the full raw "
        "file. Label-free structure only.",
    ),
    "eda/raw_label_mapping.csv": (
        "Raw label spellings",
        "Eleven raw label spellings and the six canonical classes they map to (full file).",
    ),
    "eda/class_distribution.csv": (
        "Class distribution",
        "Rows and share per class in the training portion.",
    ),
    "eda/missing_values.csv": (
        "Missing values",
        "Missing values per input column, overall and per class (training portion).",
    ),
    "eda/col2_type_mix.csv": (
        "Col2 type mix (full file)",
        "Python types of raw invoice numbers, including corrupted scientific-notation strings. "
        "Label-free.",
    ),
    "eda/col2_patterns.csv": (
        "Col2 format patterns (full file)",
        "Most common invoice number shapes (digit to 9, letter to A). Label-free.",
    ),
    "eda/amount_summary.csv": (
        "Amount summary",
        "Headline statistics of Col3 on the training portion.",
    ),
    "eda/amount_by_class.csv": (
        "Amount by class",
        "Count, median, mean, min, and max of Col3 per class.",
    ),
    "eda/amount_histogram.csv": (
        "Amount histogram (signed log)",
        "Per-class row counts over 30 equal-width bins of sign(x) * log1p(|x|).",
    ),
    "eda/amount_outliers.csv": (
        "Amount outliers (IQR on signed log)",
        "Rows outside 1.5 x IQR fences of the signed-log amount. Kept in training because "
        "they look like real ledger values (D07).",
    ),
    "eda/period_by_class.csv": (
        "Posting period by class",
        "Rows per monthly period and class.",
    ),
    "eda/day_of_month.csv": (
        "Day of month",
        "Day-of-month distribution of raw Col5 dates: month starts and ends only, so Col5 is "
        "a monthly period.",
    ),
    "eda/col7_by_class.csv": (
        "Document flag by class",
        "Col7 values by class.",
    ),
    "eda/text_cardinality.csv": (
        "Text column cardinality",
        "Unique values, token vocabulary, tokens per row, singletons, and in-sample majority "
        "purity per text column.",
    ),
    "eda/top_values_col1.csv": (
        "Top vendors (Col1)",
        "Most frequent Col1 values with class counts.",
    ),
    "eda/top_values_col4.csv": (
        "Top descriptions (Col4)",
        "Most frequent Col4 values with class counts.",
    ),
    "eda/top_values_col6.csv": (
        "Top accounts (Col6)",
        "Most frequent Col6 values with class counts.",
    ),
    "eda/token_overlap.csv": (
        "Token overlap across columns",
        "Tokens shared between text columns. The same token plays different roles, so each "
        "column gets its own vectorizer.",
    ),
    "eda/cramers_v.csv": (
        "Cramer's V (bias-corrected)",
        "Association between categorical views of the inputs and the label.",
    ),
    "eda/mutual_information.csv": (
        "Mutual information with the label",
        "Mutual information (nats) and normalized mutual information of each input view with "
        "the label.",
    ),
    "eda/duplicates_summary.json": (
        "Duplicates summary",
        "Exact duplicates and label conflicts on the training rows.",
    ),
    "eda/duplicate_group_sizes.csv": (
        "Duplicate group sizes",
        "How many group-key groups have each size.",
    ),
    "eda/conflict_groups.csv": (
        "Label conflict groups",
        "Identical rows (group key) carrying different labels.",
    ),
    "eda/rare_class_rows.csv": (
        "Rare class rows",
        "Every training row of Category_3 to Category_6.",
    ),
    "eda/accuracy_ceiling.json": (
        "Accuracy ceiling",
        "Best possible accuracy given rows with identical inputs and different labels.",
    ),
}

ReportBuilder = Callable[[], pd.DataFrame | dict[str, Any]]


def _builders(
    resolved: pd.DataFrame, labels_raw: pd.Series, training: TrainingData
) -> dict[str, ReportBuilder]:
    frame, y, groups = training.frame, training.y, training.groups
    return {
        "eda/schema_summary.json": lambda: schema_summary(resolved),
        "eda/raw_label_mapping.csv": lambda: raw_label_mapping(labels_raw),
        "eda/class_distribution.csv": lambda: class_distribution(y),
        "eda/missing_values.csv": lambda: missing_values(frame, y),
        "eda/col2_type_mix.csv": lambda: col2_type_mix(resolved[COL_INVOICE]),
        "eda/col2_patterns.csv": lambda: col2_patterns(resolved[COL_INVOICE]),
        "eda/amount_summary.csv": lambda: amount_summary(frame[AMOUNT]),
        "eda/amount_by_class.csv": lambda: amount_by_class(frame[AMOUNT], y),
        "eda/amount_histogram.csv": lambda: amount_histogram(frame[AMOUNT], y),
        "eda/amount_outliers.csv": lambda: amount_outliers(frame, y),
        "eda/period_by_class.csv": lambda: period_by_class(frame, y),
        "eda/day_of_month.csv": lambda: day_of_month(training.raw[COL_PERIOD]),
        "eda/col7_by_class.csv": lambda: col7_by_class(frame, y),
        "eda/text_cardinality.csv": lambda: text_cardinality(frame, y),
        "eda/top_values_col1.csv": lambda: top_values(frame, y, COL_VENDOR),
        "eda/top_values_col4.csv": lambda: top_values(frame, y, COL_DESCRIPTION),
        "eda/top_values_col6.csv": lambda: top_values(frame, y, COL_ACCOUNT),
        "eda/token_overlap.csv": lambda: token_overlap(frame),
        "eda/cramers_v.csv": lambda: cramers_v_matrix(frame, y),
        "eda/mutual_information.csv": lambda: mutual_information(frame, y),
        "eda/duplicates_summary.json": lambda: duplicates_summary(frame, y, groups),
        "eda/duplicate_group_sizes.csv": lambda: duplicate_group_sizes(groups),
        "eda/conflict_groups.csv": lambda: conflict_groups(frame, y, groups),
        "eda/rare_class_rows.csv": lambda: rare_class_rows(frame, y),
        "eda/accuracy_ceiling.json": lambda: accuracy_ceiling(frame, y),
    }


def run_eda(settings: Settings) -> list[str]:
    """Compute every EDA report and write it under ``reports/eda``.

    Args:
        settings: Project settings.

    Returns:
        The report paths written, relative to ``reports/``.
    """
    raw, _, _ = read_table(settings.paths.raw_data, settings.paths.raw_data.name, settings.limits)
    resolved, _, _ = resolve_columns(raw, settings.limits)
    resolved = resolved.reset_index(drop=True)
    labels_raw = resolved[LABEL_COLUMN]
    training = load_training_frame(settings)
    builders = _builders(resolved, labels_raw, training)
    for file, (title, description) in EDA_CATALOG.items():
        result = builders[file]()
        if isinstance(result, pd.DataFrame):
            write_table(settings, file, result, title, PRODUCER, description)
        else:
            write_json(settings, file, result, title, PRODUCER, description)
        logger.info("wrote %s", file)
    return list(EDA_CATALOG)
