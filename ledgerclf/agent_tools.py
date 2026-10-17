"""Read-only tools the explanation agent calls (LLM or offline engine).

Every tool answers from the fitted model, the exact explanation payload, or
the training reference statistics, so every number an answer cites can be
traced to a tool result. Tools never read holdout labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from ledgerclf.exceptions import LedgerClfError
from ledgerclf.explain import explained_row
from ledgerclf.features import normalize_text
from ledgerclf.jsonsafe import to_jsonable
from ledgerclf.labels import CANONICAL_CLASSES
from ledgerclf.predict import PredictionResult
from ledgerclf.reference import PAIR_KEY, pair_key
from ledgerclf.schema import (
    AMOUNT,
    AMOUNT_WAS_MISSING,
    COL_ACCOUNT,
    COL_AMOUNT,
    COL_DESCRIPTION,
    COL_DOC_TYPE,
    COL_PERIOD,
    COL_VENDOR,
    INPUT_COLUMNS,
    PERIOD,
    PERIOD_WAS_MISSING,
    SOURCE_ROW,
    coerce_values,
)
from ledgerclf.values import period_label

LOOKUP_KINDS = {
    "vendor": COL_VENDOR,
    "account": COL_ACCOUNT,
    "doc_type": COL_DOC_TYPE,
    "description": COL_DESCRIPTION,
}
TOKEN_KINDS = {
    "vendor_token": COL_VENDOR,
    "description_token": COL_DESCRIPTION,
    "account_token": COL_ACCOUNT,
}
WHAT_IF_COLUMNS = (COL_VENDOR, COL_AMOUNT, COL_DESCRIPTION, COL_PERIOD, COL_ACCOUNT, COL_DOC_TYPE)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_prediction_explanation",
        "description": (
            "Return the model's prediction for the row being explained: label, calibrated "
            "probabilities, runner-up, the exact contribution of each input column to the margin "
            "between the predicted class and the runner-up (positive pushes toward the predicted "
            "class), the strongest individual features, training evidence for the row's values, "
            "risk flags, and context. Call this first."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "lookup_training_evidence",
        "description": (
            "Return how many training rows had a given value and how they were labeled. kind is "
            "one of vendor, account, pair, doc_type, description, vendor_token, description_token, "
            "account_token. For pair, value is 'vendor || account'. Tokens look like Word563."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"kind": {"type": "string"}, "value": {"type": "string"}},
            "required": ["kind", "value"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_similar_training_rows",
        "description": (
            "Return the training rows most similar to the explained row in the model's feature "
            "space, with their labels and similarity (0 to 1)."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "what_if",
        "description": (
            "Re-score the explained row with one input changed and return the new prediction. "
            f"column is one of {', '.join(WHAT_IF_COLUMNS)}; value is the new cell text "
            f"(an amount like 1680 for {COL_AMOUNT}, a date like 2018-08-31 for {COL_PERIOD})."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"column": {"type": "string"}, "value": {"type": "string"}},
            "required": ["column", "value"],
            "additionalProperties": False,
        },
    },
]


class ToolError(LedgerClfError):
    """A tool call had invalid arguments; the message goes back to the model."""

    default_code = "AGENT_TOOL_ERROR"


@dataclass
class ToolContext:
    """Everything the tools need about one explained row."""

    service: Any
    result: PredictionResult
    row_number: int
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.payload:
            self.payload = self.service.explain(self.result, self.row_number)


def run_tool(context: ToolContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute one tool by name.

    Args:
        context: The explained row and the service.
        name: Tool name from ``TOOL_DEFINITIONS``.
        arguments: Tool input.

    Returns:
        A JSON-safe result.

    Raises:
        ToolError: For unknown tools or invalid arguments.
    """
    handlers = {
        "get_prediction_explanation": lambda: explanation_view(context.payload),
        "lookup_training_evidence": lambda: lookup_evidence(
            context, str(arguments.get("kind", "")), str(arguments.get("value", ""))
        ),
        "find_similar_training_rows": lambda: similar_rows(context),
        "what_if": lambda: what_if(
            context, str(arguments.get("column", "")), str(arguments.get("value", ""))
        ),
    }
    if name not in handlers:
        raise ToolError(f"Unknown tool '{name}'. Available: {', '.join(sorted(handlers))}.")
    result: dict[str, Any] = to_jsonable(handlers[name]())
    return result


def explanation_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the payload fields the model needs (drops the deterministic text)."""
    return {k: v for k, v in payload.items() if k != "fallback_text"}


def lookup_evidence(context: ToolContext, kind: str, value: str) -> dict[str, Any]:
    """Return training row counts by class for one value or token."""
    reference = context.service.bundle.reference
    if kind in LOOKUP_KINDS:
        counts = reference.value_class_counts[LOOKUP_KINDS[kind]].get(normalize_text(value), {})
    elif kind == "pair":
        vendor, _, account = value.partition("||")
        counts = reference.value_class_counts[PAIR_KEY].get(pair_key(vendor, account), {})
    elif kind in TOKEN_KINDS:
        counts = reference.token_class_counts[TOKEN_KINDS[kind]].get(normalize_text(value), {})
    else:
        raise ToolError(
            f"Unknown kind '{kind}'. Use one of: "
            + ", ".join([*LOOKUP_KINDS, "pair", *TOKEN_KINDS])
            + "."
        )
    total = int(sum(counts.values()))
    return {
        "kind": kind,
        "value": value,
        "training_rows": total,
        "class_counts": dict(sorted(counts.items())),
        "class_shares": {c: round(n / total, 4) for c, n in sorted(counts.items())}
        if total
        else {},
        "seen_in_training": total > 0,
    }


def similar_rows(context: ToolContext) -> dict[str, Any]:
    """Return the nearest training rows by cosine similarity of feature vectors."""
    index = context.service.training_index()
    pipeline = context.service.bundle.pipeline
    inputs = explained_row(context.result, context.row_number).inputs
    vector = normalize(pipeline.named_steps["features"].transform(inputs))
    scores = np.asarray((index.matrix @ vector.T).toarray()).ravel()
    count = context.service.settings.agent.similar_rows
    order = np.argsort(-scores, kind="stable")[:count]
    rows = []
    for i in order:
        row = index.frame.iloc[i]
        rows.append(
            {
                "similarity": round(float(scores[i]), 4),
                "label": str(index.y[i]),
                COL_VENDOR: row[COL_VENDOR],
                COL_DESCRIPTION: row[COL_DESCRIPTION],
                COL_ACCOUNT: row[COL_ACCOUNT],
                COL_DOC_TYPE: row[COL_DOC_TYPE],
                "amount": None if pd.isna(row[AMOUNT]) else round(float(row[AMOUNT]), 2),
                "period": None if pd.isna(row[PERIOD]) else period_label(int(row[PERIOD])),
            }
        )
    labels = pd.Series([r["label"] for r in rows]).value_counts()
    return {
        "rows": rows,
        "label_counts": {c: int(labels.get(c, 0)) for c in CANONICAL_CLASSES if labels.get(c, 0)},
    }


def what_if(context: ToolContext, column: str, value: str) -> dict[str, Any]:
    """Re-score the row with one input changed."""
    if column not in WHAT_IF_COLUMNS:
        raise ToolError(f"Column must be one of {', '.join(WHAT_IF_COLUMNS)}.")
    original = explained_row(context.result, context.row_number).inputs
    edited = original[[SOURCE_ROW, *INPUT_COLUMNS]].copy()
    edited[column] = value
    try:  # the same cleaning as an upload, so the answer matches re-uploading the row
        row, _ = coerce_values(edited.reset_index(drop=True), context.service.settings.limits)
    except LedgerClfError as error:
        raise ToolError(f"'{value}' cannot be scored: {error.message}") from error
    if column == COL_AMOUNT and bool(row[AMOUNT_WAS_MISSING].iloc[0]):
        raise ToolError(f"'{value}' is not a readable amount.")
    if column == COL_PERIOD and bool(row[PERIOD_WAS_MISSING].iloc[0]):
        raise ToolError(f"'{value}' is not a readable date.")
    proba = context.service.bundle.pipeline.predict_proba(row)[0]
    order = np.argsort(-proba)
    original = context.payload["prediction"]
    return {
        "changed": {"column": column, "value": value},
        "original_prediction": {"label": original["label"], "probability": original["probability"]},
        "new_prediction": {
            "label": CANONICAL_CLASSES[order[0]],
            "probability": round(float(proba[order[0]]), 4),
        },
        "new_probabilities": {
            c: round(float(p), 4) for c, p in zip(CANONICAL_CLASSES, proba, strict=True)
        },
    }


def tool_result_text(result: dict[str, Any]) -> str:
    """Serialize a tool result compactly for the model."""
    return json.dumps(result, separators=(",", ":"), sort_keys=True)
