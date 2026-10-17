"""Preprocessing reports (docs/04 section 15, ``reports/preprocessing``).

Shows what each stage of the shared pipeline does to the data: stage
snapshots, before and after samples, the feature space by group, and a
catalog of every transformer with the decision behind it. The feature logic
itself lives only in ``features.py``; this module only describes it.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ledgerclf.config import Settings
from ledgerclf.eda import signed_log
from ledgerclf.features import TRANSFORMERS, build_feature_pipeline, normalize_text
from ledgerclf.reports import write_json, write_table
from ledgerclf.schema import (
    AMOUNT,
    COL_AMOUNT,
    COL_INVOICE,
    COL_PERIOD,
    INPUT_COLUMNS,
    PERIOD,
)
from ledgerclf.splitting import ROW_ID, TrainingData, load_training_frame
from ledgerclf.values import is_blank_cell, period_label

PRODUCER = "eda"
SAMPLE_ROWS = 5
TRANSFORMER_NOTES: dict[str, tuple[str, str]] = {
    "col1_tokens": (
        "Vendor tokens carry identity; one vectorizer per column keeps roles apart.",
        "D05",
    ),
    "col1_value": (
        "Whole vendor names recur; values seen once fold into an infrequent bucket.",
        "D05",
    ),
    "col4_tokens": (
        "Line descriptions are the strongest single signal (ablation: -4 points).",
        "D05",
    ),
    "col6_tokens": ("Account tokens separate classes within a vendor.", "D05"),
    "col6_value": ("Whole account names, infrequent values bucketed.", "D05"),
    "pair": ("Vendor-account pairs resolve vendors split across classes.", "D05"),
    "col7": ("Document flag; Doc3 holds most Category_4 rows.", "D05"),
    "amount": ("Signed log keeps credits negative and tames an 8.8 million maximum.", "D07"),
    "amount_flags": (
        "Negative and round amounts mark rare classes (Category_5, Category_3/4).",
        "D07",
    ),
    "period": (
        "Monthly period index, clipped to the training range to stop extrapolation.",
        "D08",
    ),
}


def _missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(c): int(frame[c].isna().sum()) for c in frame.columns}


def stage_snapshots(settings: Settings, training: TrainingData) -> dict[str, Any]:
    """Describe rows, columns, missing values, and dtypes at each pipeline stage."""
    pipeline = build_feature_pipeline(settings.feature)
    matrix = sp.csr_matrix(pipeline.fit_transform(training.frame))
    coerced = training.frame
    raw = training.raw
    missing_raw = {str(c): int(raw[c].map(is_blank_cell).sum()) for c in raw.columns}
    return {
        "resolved": {
            "scope": "training portion, original cells",
            "rows": len(raw),
            "columns": list(raw.columns),
            "blank_cells": missing_raw,
            "python_types": {
                str(c): dict(Counter(type(v).__name__ for v in raw[c]).most_common())
                for c in raw.columns
            },
        },
        "coerced": {
            "scope": "training portion",
            "rows": len(coerced),
            "columns": list(coerced.columns),
            "missing": _missing_counts(coerced),
            "dtypes": {str(c): str(t) for c, t in coerced.dtypes.items()},
        },
        "matrix": {
            "scope": "training portion",
            "rows": matrix.shape[0],
            "features": matrix.shape[1],
            "stored_values": int(matrix.nnz),
            "density": matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
            "format": "scipy CSR (sparse)",
        },
    }


def before_after_samples(settings: Settings, training: TrainingData) -> pd.DataFrame:
    """Show five training rows at three stages: raw cell, cleaned value, model input."""
    positions = np.linspace(0, len(training.frame) - 1, SAMPLE_ROWS).astype(int)
    clip_low, clip_high = training.frame[PERIOD].min(), training.frame[PERIOD].max()
    rows = []
    for position in positions:
        row, raw_row = training.frame.iloc[position], training.raw.iloc[position]
        for column in INPUT_COLUMNS:
            rows.append(
                {
                    ROW_ID: int(row[ROW_ID]),
                    "column": column,
                    "raw_value": str(raw_row[column]),
                    "cleaned_value": "" if row[column] is None else str(row[column]),
                    "model_input": _model_input(row, column, clip_low, clip_high),
                }
            )
    return pd.DataFrame(rows)


def _model_input(row: pd.Series, column: str, clip_low: float, clip_high: float) -> str:
    if column == COL_INVOICE:
        return "not used by the model (D04)"
    if column == COL_AMOUNT:
        value = row[AMOUNT]
        return f"signed_log={signed_log(np.array([value]))[0]:.4f}, negative={int(value < 0)}"
    if column == COL_PERIOD:
        clipped = int(np.clip(row[PERIOD], clip_low, clip_high))
        return f"period index {clipped} ({period_label(clipped)}), then standardized"
    return normalize_text(row[column])


def feature_space_summary(settings: Settings, training: TrainingData) -> pd.DataFrame:
    """Return features and sparsity per transformer of the fitted pipeline."""
    pipeline = build_feature_pipeline(settings.feature)
    matrix = sp.csr_matrix(pipeline.fit_transform(training.frame))
    names = pipeline.get_feature_names_out()
    transformers = np.array([str(n).split("__", 1)[0] for n in names])
    rows = []
    for name in dict.fromkeys(transformers):
        block = matrix[:, np.flatnonzero(transformers == name)]
        nonzero_per_row = np.diff(block.indptr)
        rows.append(
            {
                "transformer": name,
                "group": TRANSFORMERS[name][0],
                "features": block.shape[1],
                "rows_with_nonzero_share": float((nonzero_per_row > 0).mean()),
                "mean_nonzero_per_row": float(nonzero_per_row.mean()),
            }
        )
    return pd.DataFrame(rows)


def transformer_catalog(settings: Settings) -> dict[str, Any]:
    """List every transformer with its steps, parameters, reason, and decision id."""
    pipeline = build_feature_pipeline(settings.feature)
    catalog = []
    for name, transformer, columns in pipeline.transformers:
        reason, decision = TRANSFORMER_NOTES[name]
        steps = [
            {"step": step, "class": type(obj).__name__, "params": _plain_params(obj)}
            for step, obj in transformer.steps
        ]
        catalog.append(
            {
                "name": name,
                "columns": columns,
                "steps": steps,
                "reason": reason,
                "decision": decision,
            }
        )
    return {"transformers": catalog, "dropped": {"Col2": "Invoice number: display only (D04)."}}


def _plain_params(estimator: Any) -> dict[str, Any]:
    params = estimator.get_params(deep=False)
    plain = (str, int, float, bool, type(None))
    return {k: v if isinstance(v, plain) else str(v) for k, v in sorted(params.items())}


PREPROCESSING_CATALOG: dict[str, tuple[str, str]] = {
    "preprocessing/stage_snapshots.json": (
        "Pipeline stage snapshots",
        "Rows, columns, missing values, and dtypes at each stage: raw, resolved, coerced, "
        "and the final sparse matrix.",
    ),
    "preprocessing/before_after_samples.csv": (
        "Before and after samples",
        "Five training rows shown as the raw cell, the cleaned value, and the model input.",
    ),
    "preprocessing/feature_space_summary.csv": (
        "Feature space by transformer",
        "Number of features and sparsity per transformer of the fitted pipeline.",
    ),
    "preprocessing/transformer_catalog.json": (
        "Transformer catalog",
        "Every transformer with its steps, parameters, reason, and decision id.",
    ),
}


def run_preprocessing_reports(settings: Settings) -> list[str]:
    """Write every preprocessing report under ``reports/preprocessing``.

    Args:
        settings: Project settings.

    Returns:
        The report paths written, relative to ``reports/``.
    """
    training = load_training_frame(settings)
    results: dict[str, pd.DataFrame | dict[str, Any]] = {
        "preprocessing/stage_snapshots.json": stage_snapshots(settings, training),
        "preprocessing/before_after_samples.csv": before_after_samples(settings, training),
        "preprocessing/feature_space_summary.csv": feature_space_summary(settings, training),
        "preprocessing/transformer_catalog.json": transformer_catalog(settings),
    }
    for file, (title, description) in PREPROCESSING_CATALOG.items():
        result = results[file]
        if isinstance(result, pd.DataFrame):
            write_table(settings, file, result, title, PRODUCER, description)
        else:
            write_json(settings, file, result, title, PRODUCER, description)
    return list(PREPROCESSING_CATALOG)
