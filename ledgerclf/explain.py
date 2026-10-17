"""Exact per-row explanation payloads.

The payload explains the margin between the predicted class and the
runner-up: every feature's contribution to that margin, aggregated by source
column, with training evidence for the row's values. Contributions add up
exactly to the model's logit margin, so the agent never estimates numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ledgerclf.artifacts import ModelBundle
from ledgerclf.config import ExplainParams
from ledgerclf.exceptions import LedgerClfError
from ledgerclf.features import (
    GROUP_ACCOUNT,
    GROUP_AMOUNT,
    GROUP_DESCRIPTION,
    GROUP_DOC_TYPE,
    GROUP_PAIR,
    GROUP_PERIOD,
    GROUP_VENDOR,
    TRANSFORMERS,
    feature_groups,
    normalize_text,
    readable_feature_name,
)
from ledgerclf.flags import FLAG_MESSAGES
from ledgerclf.jsonsafe import to_jsonable
from ledgerclf.labels import CANONICAL_CLASSES
from ledgerclf.predict import PredictionResult
from ledgerclf.reference import PAIR_KEY, ReferenceStats, pair_key
from ledgerclf.schema import (
    AMOUNT,
    COL_ACCOUNT,
    COL_DESCRIPTION,
    COL_DOC_TYPE,
    COL_VENDOR,
    INPUT_COLUMNS,
    ROW_NUMBER,
    SOURCE_ROW,
)
from ledgerclf.values import period_label

SCHEMA_VERSION = "1.0"
CONFIDENCE_WORDS: tuple[tuple[float, str], ...] = (
    (0.95, "very confident"),
    (0.8, "confident"),
    (0.6, "only moderately confident"),
    (0.0, "unsure"),
)
GROUP_LABELS = {
    GROUP_VENDOR: "vendor (Col1)",
    GROUP_PAIR: "vendor-account pair",
    GROUP_AMOUNT: "amount (Col3)",
    GROUP_DESCRIPTION: "line description (Col4)",
    GROUP_PERIOD: "posting period (Col5)",
    GROUP_ACCOUNT: "GL account (Col6)",
    GROUP_DOC_TYPE: "document flag (Col7)",
}
TOKEN_SOURCE = {
    "col1_tokens": COL_VENDOR,
    "col4_tokens": COL_DESCRIPTION,
    "col6_tokens": COL_ACCOUNT,
}


class ExplanationError(LedgerClfError):
    """The requested row cannot be explained."""

    default_code = "EXPLAIN_ROW_INVALID"


@dataclass(frozen=True)
class ExplainedRow:
    """One row of a prediction result: its table entry and its coerced inputs."""

    prediction: pd.Series
    inputs: pd.DataFrame
    frame_position: int


def explained_row(result: PredictionResult, row_number: object) -> ExplainedRow:
    """Find a row by its ``row_number`` value (not its position, so sorted tables work).

    Raises:
        ExplanationError: When the row number is not a whole number or does not exist.
    """
    total = len(result.predictions)
    if isinstance(row_number, bool) or not isinstance(row_number, int | np.integer):
        raise ExplanationError(f"Row number must be a whole number between 1 and {total}.")
    position = _locate(result.predictions, int(row_number))
    if position is None:
        raise ExplanationError(
            f"Row {row_number} does not exist. Choose a row between 1 and {total}."
        )
    frame_position = _locate(result.frame, int(row_number))
    if frame_position is None:
        raise ExplanationError(
            "The predictions no longer match their inputs. Predict the file again."
        )
    return ExplainedRow(
        result.predictions.iloc[position], result.frame.iloc[[frame_position]], frame_position
    )


def _locate(table: pd.DataFrame, row_number: int) -> int | None:
    found = np.flatnonzero(table[ROW_NUMBER].to_numpy() == row_number)
    return int(found[0]) if len(found) else None


def _require_same_model(bundle: ModelBundle, result: PredictionResult) -> None:
    if result.model_used != bundle.name:
        raise ExplanationError(
            f"These predictions came from the {result.model_used} model, but the "
            f"{bundle.name} model is loaded now. Predict the file again, then explain the row."
        )


def explain_row(bundle: ModelBundle, result: PredictionResult, row_number: int) -> dict[str, Any]:
    """Build the explanation payload of one predicted row.

    Args:
        bundle: The bundle that produced ``result``.
        result: Prediction result holding the coerced frame and row flags.
        row_number: 1-based row number as shown in the predictions table.

    Returns:
        A JSON-safe payload (schema version 1.0).

    Raises:
        ExplanationError: When the row number is invalid or another model made the predictions.
    """
    params = ExplainParams()
    _require_same_model(bundle, result)
    target = explained_row(result, row_number)
    prediction, row = target.prediction, target.inputs
    features = bundle.pipeline.named_steps["features"]
    model = bundle.pipeline.named_steps["model"]
    matrix = features.transform(row)
    parts = model.contributions(matrix)
    names = [str(n) for n in features.get_feature_names_out()]
    k = CANONICAL_CLASSES.index(str(prediction["predicted_label"]))
    r = CANONICAL_CLASSES.index(str(prediction["runner_up_label"]))
    margin = parts.values[0, k, :] - parts.values[0, r, :]
    bias = float(parts.bias[0, k] - parts.bias[0, r])
    by_group = _by_group(margin, names)
    top = _top_features(margin, names, np.asarray(matrix[0].toarray()).ravel(), params.top_features)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "row_number": int(prediction[ROW_NUMBER]),
        "source_row": int(prediction[SOURCE_ROW]),
        "input": {c: prediction[c] for c in INPUT_COLUMNS},
        "prediction": _prediction_block(
            prediction, float(parts.logits[0, k] - parts.logits[0, r]), params
        ),
        "probabilities": {
            c: round(float(prediction[f"p_{c}"]), params.display_decimals)
            for c in CANONICAL_CLASSES
        },
        "model": {
            "name": bundle.name,
            "linear_weight": model.effective_weight,
            "temperature": float(model.temperature),
            "fallback_active": bundle.is_fallback,
        },
        "contributions": {
            "target": "predicted_minus_runner_up",
            "total": float(sum(g["contribution"] for g in by_group) + bias),
            "bias": bias,
            "by_group": by_group,
            "top_features": top,
        },
        "evidence": _evidence(
            row, top, bundle.reference, str(prediction["predicted_label"]), params.evidence_tokens
        ),
        "flags": _flags(result, target.frame_position),
        "context": _context(bundle.reference, row),
    }
    payload["fallback_text"] = render_fallback_text(payload)
    return to_jsonable(payload)


def _prediction_block(
    prediction: pd.Series, margin: float, params: ExplainParams
) -> dict[str, Any]:
    decimals = params.display_decimals
    return {
        "label": prediction["predicted_label"],
        "probability": round(float(prediction["confidence"]), decimals),
        "runner_up": {
            "label": prediction["runner_up_label"],
            "probability": round(float(prediction["runner_up_probability"]), decimals),
        },
        "logit_margin": margin,
        "review_required": bool(prediction["review_required"]),
    }


def _by_group(margin: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    groups = feature_groups(names)
    rows = [
        {"group": g, "label": GROUP_LABELS[g], "contribution": float(margin[idx].sum())}
        for g, idx in groups.items()
    ]
    return sorted(rows, key=lambda item: -abs(item["contribution"]))


def _top_features(
    margin: np.ndarray, names: list[str], values: np.ndarray, count: int
) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(margin), kind="stable")[:count]
    return [
        {
            "group": TRANSFORMERS[names[j].split("__", 1)[0]][0],
            "feature": readable_feature_name(names[j]),
            "transformer": names[j].split("__", 1)[0],
            "present_in_row": bool(values[j] != 0),
            "contribution": float(margin[j]),
        }
        for j in order
        if margin[j] != 0
    ]


def _counts_entry(
    group: str, value: str, display: object, counts: dict[str, dict[str, int]], predicted: str
) -> dict[str, Any]:
    classes = counts.get(value, {})
    total = int(sum(classes.values()))
    return {
        "group": group,
        "value": display,
        "seen_in_training": total > 0,
        "training_rows": total,
        "class_counts": dict(sorted(classes.items())),
        "share_predicted_class": (classes.get(predicted, 0) / total) if total else None,
    }


def _evidence(
    row: Any, top: list[dict[str, Any]], reference: ReferenceStats, predicted: str, tokens: int
) -> list[dict[str, Any]]:
    first = row.iloc[0]
    values = reference.value_class_counts
    items = [
        _counts_entry(
            GROUP_VENDOR,
            normalize_text(first[COL_VENDOR]),
            first[COL_VENDOR],
            values[COL_VENDOR],
            predicted,
        ),
        _counts_entry(
            GROUP_ACCOUNT,
            normalize_text(first[COL_ACCOUNT]),
            first[COL_ACCOUNT],
            values[COL_ACCOUNT],
            predicted,
        ),
        _counts_entry(
            GROUP_PAIR,
            pair_key(first[COL_VENDOR], first[COL_ACCOUNT]),
            f"{first[COL_VENDOR]} || {first[COL_ACCOUNT]}",
            values[PAIR_KEY],
            predicted,
        ),
        _counts_entry(
            GROUP_DOC_TYPE,
            normalize_text(first[COL_DOC_TYPE]),
            first[COL_DOC_TYPE],
            values[COL_DOC_TYPE],
            predicted,
        ),
    ]
    shown = 0
    for feature in top:
        source = TOKEN_SOURCE.get(feature["transformer"])
        if source is None or not feature["present_in_row"] or shown >= tokens:
            continue
        token = feature["feature"].split(":", 1)[1]
        entry = _counts_entry(
            TRANSFORMERS[feature["transformer"]][0],
            token,
            f"token {token} in {source}",
            reference.token_class_counts[source],
            predicted,
        )
        items.append(entry)
        shown += 1
    return items


def _flags(result: PredictionResult, position: int) -> list[dict[str, str]]:
    if result.row_flags.empty:
        return []
    row = result.row_flags.iloc[position]
    return [
        {"code": code, "message": FLAG_MESSAGES[code]}
        for code in result.row_flags.columns
        if bool(row[code])
    ]


def _context(reference: ReferenceStats, row: Any) -> dict[str, Any]:
    total = reference.training_rows
    amount = (
        float(row.iloc[0][AMOUNT]) if not np.isnan(row.iloc[0][AMOUNT]) else reference.amount_median
    )
    return {
        "training_rows": total,
        "class_prior": {c: reference.class_counts[c] / total for c in CANONICAL_CLASSES},
        "period_range": [period_label(reference.period_min), period_label(reference.period_max)],
        "amount_percentile": reference.amount_percentile(amount),
    }


def render_fallback_text(payload: dict[str, Any], extra_facts: Sequence[str] = ()) -> str:
    """Render the deterministic explanation used when no LLM is available.

    The layout matches the Claude agent's answer: a plain-language summary,
    the exact model facts, and what a reviewer should check.

    Args:
        payload: An explanation payload (``fallback_text`` not required).
        extra_facts: Further Markdown bullet lines for the facts section.

    Returns:
        Markdown text.
    """
    prediction = payload["prediction"]
    label, runner = prediction["label"], prediction["runner_up"]
    drivers = ", ".join(
        f"{g['label']} {g['contribution']:+.2f}" for g in payload["contributions"]["by_group"][:3]
    )
    facts = [
        f"- Prediction: {label} at {prediction['probability']:.1%} (runner-up "
        f"{runner['label']} at {runner['probability']:.1%}), calibrated probability.",
        f"- Drivers (log-odds, {label} versus {runner['label']}; positive supports "
        f"{label}): {drivers}; class base rate {payload['contributions']['bias']:+.2f}.",
        *(f"- History: {line}." for line in _history_lines(payload)),
        *extra_facts,
    ]
    sections = [
        f"**Summary:** {_summary(payload)}",
        "",
        "**Model facts**",
        *facts,
        "",
        f"**What to check:** {_check_text(payload)}",
    ]
    return "\n".join(sections)


def _summary(payload: dict[str, Any]) -> str:
    prediction = payload["prediction"]
    label, probability = prediction["label"], prediction["probability"]
    review = (
        "A person should check it before it is used."
        if prediction["review_required"]
        else "It can be used without manual review."
    )
    return (
        f"This invoice line most likely belongs to {label}. The model is "
        f"{_confidence_words(probability)} ({probability:.0%}), {_main_reason(payload)} {review}"
    )


def _confidence_words(probability: float) -> str:
    for floor, words in CONFIDENCE_WORDS:
        if probability >= floor:
            return words
    return CONFIDENCE_WORDS[-1][1]


def _main_reason(payload: dict[str, Any]) -> str:
    contributions = payload["contributions"]
    label = payload["prediction"]["label"]
    if contributions["bias"] > contributions["total"]:
        prior = payload["context"]["class_prior"][label]
        runner = payload["prediction"]["runner_up"]["label"]
        return (
            f"mainly because most invoices ({prior:.0%}) belong to {label}; this line's own "
            f"details lean toward {runner}."
        )
    supporting = [g for g in contributions["by_group"] if g["contribution"] > 0]
    history = _best_history(payload, [g["group"] for g in supporting])
    if history is not None:
        return f"mainly because {history}."
    if supporting:
        return f"mainly because of its {supporting[0]['label']}."
    return "although no single detail stands out."


def _best_history(payload: dict[str, Any], groups: list[str]) -> str | None:
    """Past-invoice evidence for the strongest supporting input that has any."""
    label = payload["prediction"]["label"]
    by_group = {e["group"]: e for e in payload["evidence"]}
    for group in groups:
        entry = by_group.get(group)
        count = entry["class_counts"].get(label, 0) if entry else 0
        if entry and count:
            name, total = GROUP_LABELS[group], entry["training_rows"]
            return f"{count} of {total} past invoices with the same {name} were {label}"
    return None


def _history_lines(payload: dict[str, Any]) -> list[str]:
    label = payload["prediction"]["label"]
    lines = []
    for entry in payload["evidence"][:3]:
        name = GROUP_LABELS[entry["group"]]
        if not entry["seen_in_training"]:
            lines.append(f"this {name} does not appear in the historical invoices")
            continue
        count, total = entry["class_counts"].get(label, 0), entry["training_rows"]
        lines.append(f"{count} of {total} past invoices with this {name} were {label}")
    return lines


def _check_text(payload: dict[str, Any]) -> str:
    messages = " ".join(f["message"] for f in payload["flags"])
    if payload["prediction"]["review_required"]:
        return "Review required. " + (messages or "Confirm the category against the invoice.")
    if messages:
        return "No review required, but note: " + messages
    return "No review required: every input is familiar and the model is confident."
