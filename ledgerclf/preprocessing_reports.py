"""Preprocessing reports (docs/04 section 15, ``reports/preprocessing``).

Shows what each stage of the shared pipeline does to the data: stage
snapshots, before and after samples, the feature space by group, and a
catalog of every transformer with the decision behind it. The feature logic
itself lives only in ``features.py``; this module only describes it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ledgerclf.config import EdaParams, Settings
from ledgerclf.features import TRANSFORMERS, build_feature_pipeline
from ledgerclf.reports import write_reports
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
    PERIOD,
)
from ledgerclf.splitting import ROW_ID, TrainingData, load_training_frame
from ledgerclf.values import is_blank_cell, period_label

PRODUCER = "eda"
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


@dataclass(frozen=True)
class FittedFeatures:
    """The feature pipeline fitted on the training portion, with its matrix."""

    pipeline: Any
    matrix: sp.csr_matrix
    transformer_of_column: np.ndarray


def fit_features(settings: Settings, training: TrainingData) -> FittedFeatures:
    """Fit the production feature pipeline once on the training rows."""
    pipeline = build_feature_pipeline(settings.feature)
    matrix = sp.csr_matrix(pipeline.fit_transform(training.frame))
    names = pipeline.get_feature_names_out()
    return FittedFeatures(pipeline, matrix, np.array([str(n).split("__", 1)[0] for n in names]))


def stage_snapshots(fitted: FittedFeatures, training: TrainingData) -> dict[str, Any]:
    """Describe the data at each stage: raw cells, coerced frame, engineered blocks, matrix."""
    raw, coerced, matrix = training.raw, training.frame, fitted.matrix
    return {
        "raw": {
            "scope": "training portion, original cells as read from the file",
            "rows": len(raw),
            "columns": list(raw.columns),
            "blank_cells": {str(c): int(raw[c].map(is_blank_cell).sum()) for c in raw.columns},
            "python_types": {
                str(c): dict(Counter(type(v).__name__ for v in raw[c]).most_common())
                for c in raw.columns
            },
        },
        "coerced": {
            "scope": "training portion after validation and cleaning",
            "rows": len(coerced),
            "columns": list(coerced.columns),
            "missing": _missing_counts(coerced),
            "dtypes": {str(c): str(t) for c, t in coerced.dtypes.items()},
        },
        "engineered": {
            "scope": "output of each transformer before concatenation",
            "features_per_transformer": {
                name: int((fitted.transformer_of_column == name).sum())
                for name in dict.fromkeys(fitted.transformer_of_column)
            },
            "non_finite_values": int((~np.isfinite(matrix.data)).sum()),
        },
        "matrix": {
            "scope": "training portion, input to every model",
            "rows": matrix.shape[0],
            "features": matrix.shape[1],
            "stored_values": int(matrix.nnz),
            "density": matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
            "format": "scipy CSR (sparse)",
        },
    }


def before_after_samples(
    fitted: FittedFeatures, training: TrainingData, count: int
) -> pd.DataFrame:
    """Show training rows at three stages: raw cell, cleaned value, model input.

    Model inputs come from the fitted pipeline's own transformers, so this
    table always matches what the model sees.
    """
    positions = np.linspace(0, len(training.frame) - 1, count).astype(int)
    rows = []
    for position in positions:
        row, raw_row = training.frame.iloc[[position]], training.raw.iloc[position]
        inputs = model_inputs(fitted.pipeline, row)
        for column in INPUT_COLUMNS:
            rows.append(
                {
                    ROW_ID: int(row[ROW_ID].iloc[0]),
                    "column": column,
                    "raw_value": str(raw_row[column]),
                    "cleaned_value": ""
                    if row[column].iloc[0] is None
                    else str(row[column].iloc[0]),
                    "model_input": inputs[column],
                }
            )
    return pd.DataFrame(rows)


def model_inputs(pipeline: Any, row: pd.DataFrame) -> dict[str, str]:
    """Describe what each fitted transformer feeds the model for one row (before scaling)."""
    named = pipeline.named_transformers_
    text = {
        column: str(named[name].named_steps["normalize"].transform(row[column])[0])
        for name, column in (
            ("col1_tokens", COL_VENDOR),
            ("col4_tokens", COL_DESCRIPTION),
            ("col6_tokens", COL_ACCOUNT),
        )
    }
    doc = str(named["col7"].named_steps["normalize"].transform(row[[COL_DOC_TYPE]])[0, 0])
    log_amount = float(named["amount"][:-1].transform(row[[AMOUNT]])[0, 0])
    negative, round_ = named["amount_flags"].transform(row[[AMOUNT]])[0]
    period = int(named["period"][:-1].transform(row[[PERIOD]])[0, 0])
    return {
        **text,
        COL_INVOICE: "not used by the model (D04)",
        COL_AMOUNT: (
            f"signed_log={log_amount:.4f}, is_negative={int(negative)}, is_round={int(round_)}"
        ),
        COL_PERIOD: f"period index {period} ({period_label(period)}), clipped to training range",
        COL_DOC_TYPE: doc,
    }


def feature_space_summary(fitted: FittedFeatures) -> pd.DataFrame:
    """Return features and sparsity per transformer of the fitted pipeline."""
    rows = []
    for name in dict.fromkeys(fitted.transformer_of_column):
        block = fitted.matrix[:, np.flatnonzero(fitted.transformer_of_column == name)]
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
    return {
        "transformers": catalog,
        "dropped": {COL_INVOICE: "Invoice number: display only (D04)."},
    }


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
    fitted = fit_features(settings, training)
    settings_rows = EdaParams().sample_rows
    results: dict[str, pd.DataFrame | dict[str, Any]] = {
        "preprocessing/stage_snapshots.json": stage_snapshots(fitted, training),
        "preprocessing/before_after_samples.csv": before_after_samples(
            fitted, training, settings_rows
        ),
        "preprocessing/feature_space_summary.csv": feature_space_summary(fitted),
        "preprocessing/transformer_catalog.json": transformer_catalog(settings),
    }
    return write_reports(settings, PREPROCESSING_CATALOG, results, PRODUCER)
