"""Predictions for any uploaded file: validation, flags, model, review policy, metrics.

Every non-empty row gets exactly one prediction in original order. Flags
mark risk; they never withhold a prediction. The same validation path serves
training files and uploads.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ledgerclf.artifacts import SAMPLE_PREDICTIONS_FILE, SAMPLE_ROWS_FILE, ModelBundle
from ledgerclf.config import Limits, Settings, get_settings
from ledgerclf.evaluation import compute_metrics, confusion_table, per_class_report
from ledgerclf.flags import (
    ROW_STRUCTURE_CODES,
    ROW_STRUCTURE_SUSPECT,
    apply_review_policy,
    compute_row_flags,
    flag_strings,
)
from ledgerclf.io import ReadInfo, read_table
from ledgerclf.jsonsafe import to_jsonable
from ledgerclf.labels import CANONICAL_CLASSES, normalize_labels
from ledgerclf.schema import (
    INPUT_COLUMNS,
    LABEL_COLUMN,
    ROW_NUMBER,
    SOURCE_ROW,
    Severity,
    ValidationIssue,
    ValidationReport,
    coerce_values,
    make_issue,
    resolve_columns,
)

logger = logging.getLogger(__name__)

LabelScreen = Callable[[pd.DataFrame], tuple[np.ndarray, list[ValidationIssue]]]
"""Called with the coerced inputs (labels unread) before scoring; returns a mask
of the rows whose labels must stay unread, plus issues that explain why."""

PROBABILITY_COLUMNS = tuple(f"p_{c}" for c in CANONICAL_CLASSES)
PREDICTION_COLUMNS = (
    ROW_NUMBER,
    SOURCE_ROW,
    *INPUT_COLUMNS,
    "predicted_label",
    "confidence",
    "runner_up_label",
    "runner_up_probability",
    *PROBABILITY_COLUMNS,
    "review_required",
    "flags",
)


@dataclass
class PredictionResult:
    """Predictions for one file plus its validation report and optional metrics.

    ``frame`` (coerced inputs) and ``row_flags`` are kept for explanations.
    """

    predictions: pd.DataFrame
    report: ValidationReport
    metrics: dict[str, Any] | None
    model_used: str
    fallback_reason: str | None
    timings_ms: dict[str, float]
    frame: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    row_flags: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    read_info: ReadInfo | None = None

    def to_records(self) -> list[dict[str, Any]]:
        """Return the predictions as JSON-safe dicts (native types, None for missing)."""
        records: list[dict[str, Any]] = to_jsonable(self.predictions.to_dict(orient="records"))
        return records

    def to_csv(self) -> str:
        """Return the predictions as CSV text that is safe to open in a spreadsheet.

        Text cells starting with ``=``, ``+``, ``-``, ``@``, tab, or carriage
        return (other than plain numbers) get a leading apostrophe, so Excel
        shows them as text instead of running them as formulas.
        """
        safe = self.predictions.copy()
        for column in safe.select_dtypes(exclude=["number", "bool"]).columns:
            safe[column] = safe[column].map(_spreadsheet_safe)
        return safe.to_csv(index=False, lineterminator="\n")

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe summary for APIs and the dashboard."""
        counts = self.predictions["predicted_label"].value_counts()
        return to_jsonable(
            {
                "rows": len(self.predictions),
                "model_used": self.model_used,
                "fallback_reason": self.fallback_reason,
                "review_required": int(self.predictions["review_required"].sum()),
                "predicted_counts": {c: int(counts.get(c, 0)) for c in CANONICAL_CLASSES},
                "report": self.report.to_dict(),
                "metrics": self.metrics,
                "timings_ms": self.timings_ms,
                "read_info": self.read_info.to_dict() if self.read_info else None,
            }
        )


_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")
_PLAIN_NUMBER = re.compile(r"^[+-]?\d[\d,]*(\.\d+)?$")


def _spreadsheet_safe(value: object) -> object:
    if (
        isinstance(value, str)
        and value.startswith(_FORMULA_START)
        and not _PLAIN_NUMBER.match(value)
    ):
        return "'" + value
    return value


def predict_file(
    bundle: ModelBundle,
    source: Path | bytes,
    file_name: str,
    limits: Limits | None = None,
    fallback_reason: str | None = None,
    label_screen: LabelScreen | None = None,
) -> PredictionResult:
    """Read, validate, and predict a file end to end.

    Args:
        bundle: Loaded model bundle.
        source: Path or uploaded bytes.
        file_name: Original file name.
        limits: Input limits (defaults from config).
        fallback_reason: Why the fallback model is active, if it is.
        label_screen: Check that keeps some rows' labels unread. Without one, any
            label column is dropped unread (fail closed: holdout labels stay unread).

    Returns:
        The prediction result.

    Raises:
        FileReadError: When the file cannot be read.
        SchemaError: When the file structure cannot be mapped.
    """
    limits = limits or get_settings().limits
    started = time.perf_counter()
    raw, info, issues = read_table(source, file_name, limits)
    read_ms = (time.perf_counter() - started) * 1000
    result = predict_frame(bundle, raw, limits, issues, fallback_reason, label_screen)
    result.read_info = info
    result.timings_ms = {"read": read_ms, **result.timings_ms}
    result.timings_ms["total"] = (time.perf_counter() - started) * 1000
    return result


def predict_frame(
    bundle: ModelBundle,
    raw: pd.DataFrame,
    limits: Limits,
    read_issues: list[ValidationIssue] | None = None,
    fallback_reason: str | None = None,
    label_screen: LabelScreen | None = None,
) -> PredictionResult:
    """Predict a raw sheet (read with ``header=None``).

    Args:
        bundle: Loaded model bundle.
        raw: Raw cells as returned by ``io.read_table``.
        limits: Input limits.
        read_issues: Issues already found while reading.
        fallback_reason: Why the fallback model is active, if it is.
        label_screen: Check that keeps some rows' labels unread. Without one, any
            label column is dropped unread (fail closed: holdout labels stay unread).

    Returns:
        The prediction result.
    """
    started = time.perf_counter()
    resolved, mapping, resolve_issues = resolve_columns(raw, limits)
    if label_screen is None:
        resolved = resolved.drop(columns=[LABEL_COLUMN], errors="ignore")
    frame, coerce_issues = coerce_values(resolved, limits)
    issues = [*(read_issues or []), *resolve_issues, *coerce_issues]
    if label_screen is not None and LABEL_COLUMN in frame.columns:
        frame, screen_issues = _screen_labels(frame, label_screen)
        issues += screen_issues
    validated = time.perf_counter()
    proba = _predict_proba(bundle, frame, limits)
    row_flags = compute_row_flags(frame, bundle.reference)
    row_flags[ROW_STRUCTURE_SUSPECT] = _structure_suspects(frame, issues)
    flags, review = apply_review_policy(row_flags, proba.max(axis=1), bundle.policy)
    predictions = _prediction_table(frame, proba, flags, review)
    metrics, label_issues = _label_metrics(frame, predictions)
    predictions = _attach_labels(predictions, frame, metrics)
    finished = time.perf_counter()
    report = ValidationReport(
        issues=[*issues, *label_issues],
        rows_read=len(resolved),
        rows_dropped_empty=len(resolved) - len(frame),
        rows_predicted=len(predictions),
        column_mapping=mapping,
    )
    return PredictionResult(
        predictions=predictions,
        report=report,
        metrics=metrics,
        model_used=bundle.name,
        fallback_reason=fallback_reason,
        timings_ms={
            "validate": (validated - started) * 1000,
            "predict": (finished - validated) * 1000,
        },
        frame=frame,
        row_flags=flags,
    )


def _predict_proba(bundle: ModelBundle, frame: pd.DataFrame, limits: Limits) -> np.ndarray:
    """Predict in chunks so that very large files stay within memory budgets."""
    chunks = [
        np.asarray(
            bundle.pipeline.predict_proba(frame.iloc[start : start + limits.predict_chunk_rows])
        )
        for start in range(0, len(frame), limits.predict_chunk_rows)
    ]
    return np.vstack(chunks)


def _screen_labels(
    frame: pd.DataFrame, label_screen: LabelScreen
) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    """Blank the labels the screen withholds; drop the column when none remain."""
    withheld, issues = label_screen(frame.drop(columns=[LABEL_COLUMN]))
    if not withheld.any():
        return frame, issues
    if withheld.all():
        return frame.drop(columns=[LABEL_COLUMN]), issues
    screened = frame.copy()
    screened.loc[withheld, LABEL_COLUMN] = None
    return screened, issues


def _structure_suspects(frame: pd.DataFrame, issues: list[ValidationIssue]) -> np.ndarray:
    rows = {r for i in issues if i.code in ROW_STRUCTURE_CODES for r in i.row_numbers}
    return frame[SOURCE_ROW].isin(sorted(rows)).to_numpy()


def _prediction_table(
    frame: pd.DataFrame, proba: np.ndarray, flags: pd.DataFrame, review: np.ndarray
) -> pd.DataFrame:
    order = np.argsort(-proba, axis=1, kind="stable")
    classes = np.asarray(CANONICAL_CLASSES, dtype=object)
    rows = np.arange(len(frame))
    table = frame[[ROW_NUMBER, SOURCE_ROW, *INPUT_COLUMNS]].copy()
    table["predicted_label"] = classes[order[:, 0]]
    table["confidence"] = proba[rows, order[:, 0]]
    table["runner_up_label"] = classes[order[:, 1]]
    table["runner_up_probability"] = proba[rows, order[:, 1]]
    for column, values in zip(PROBABILITY_COLUMNS, proba.T, strict=True):
        table[column] = values
    table["review_required"] = review
    table["flags"] = flag_strings(flags)
    return table.reset_index(drop=True)[list(PREDICTION_COLUMNS)]


def _label_metrics(
    frame: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[dict[str, Any] | None, list[ValidationIssue]]:
    if LABEL_COLUMN not in frame.columns:
        return None, []
    canonical, _ = normalize_labels(frame[LABEL_COLUMN])
    valid = canonical.notna().to_numpy()
    issues = []
    unknown = frame.loc[~valid & frame[LABEL_COLUMN].notna().to_numpy(), SOURCE_ROW].tolist()
    if unknown:
        issues.append(
            make_issue(
                Severity.WARNING,
                "UNKNOWN_LABELS",
                f"{len(unknown)} label(s) do not match the six classes and were left out of the "
                "accuracy figures. Their rows were still predicted.",
                rows=unknown,
            )
        )
    if not valid.any():
        return None, issues
    y_true = canonical[valid].to_numpy(dtype=object)
    y_pred = predictions.loc[valid, "predicted_label"].to_numpy(dtype=object)
    proba = predictions.loc[valid, list(PROBABILITY_COLUMNS)].to_numpy()
    metrics: dict[str, Any] = compute_metrics(y_true, y_pred, proba)
    metrics |= {
        "labeled_rows": int(valid.sum()),
        "unlabeled_rows": int((~valid).sum()),
        "per_class": per_class_report(y_true, y_pred).to_dict(orient="records"),
        "confusion_matrix": confusion_table(y_true, y_pred).to_dict(orient="records"),
    }
    issues.append(
        make_issue(
            Severity.INFO,
            "LABELS_DETECTED",
            f"The file holds labels for {int(valid.sum())} row(s); accuracy is reported for them.",
        )
    )
    return to_jsonable(metrics), issues


def _attach_labels(
    predictions: pd.DataFrame, frame: pd.DataFrame, metrics: dict[str, Any] | None
) -> pd.DataFrame:
    if metrics is None:
        return predictions
    canonical, _ = normalize_labels(frame[LABEL_COLUMN])
    table = predictions.copy()
    table["true_label"] = canonical.to_numpy(dtype=object)
    table["correct"] = [
        None if label is None else bool(label == predicted)
        for label, predicted in zip(table["true_label"], table["predicted_label"], strict=True)
    ]
    return table


def self_test(settings: Settings, bundle: Any) -> tuple[int, int]:
    """Predict the bundled sample rows and compare with the expected predictions.

    Args:
        settings: Project settings.
        bundle: Loaded model bundle.

    Returns:
        Tuple of (matching rows, total rows).
    """
    sample = settings.paths.resources / SAMPLE_ROWS_FILE
    expected = pd.read_csv(settings.paths.artifacts / SAMPLE_PREDICTIONS_FILE, encoding="utf-8")
    result = predict_file(bundle, sample, sample.name, settings.limits)
    expected_labels = expected[f"predicted_label_{bundle.name}"].tolist()
    matched = sum(
        a == b for a, b in zip(result.predictions["predicted_label"], expected_labels, strict=True)
    )
    return matched, len(expected_labels)
