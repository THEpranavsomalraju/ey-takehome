"""Exact per-row explanation payloads (docs/02 section 5, docs/04 section 13).

The payload explains the margin between the predicted class and the
runner-up: every feature's contribution to that margin, aggregated by source
column, with training evidence for the row's values. Contributions add up
exactly to the model's logit margin, so the agent never estimates numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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
from ledgerclf.labels import CANONICAL_CLASSES
from ledgerclf.predict import PredictionResult
from ledgerclf.reference import PAIR_KEY, ReferenceStats, pair_key
from ledgerclf.reports import to_jsonable
from ledgerclf.schema import (
    AMOUNT,
    COL_ACCOUNT,
    COL_DESCRIPTION,
    COL_DOC_TYPE,
    COL_VENDOR,
    INPUT_COLUMNS,
)
from ledgerclf.values import period_label

SCHEMA_VERSION = "1.0"
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


def _position(result: PredictionResult, row_number: object) -> int:
    if isinstance(row_number, bool) or not isinstance(row_number, int | np.integer):
        raise ExplanationError(
            f"Row number must be a whole number between 1 and {len(result.predictions)}."
        )
    if not 1 <= int(row_number) <= len(result.predictions):
        raise ExplanationError(
            f"Row {row_number} does not exist. "
            f"Choose a row between 1 and {len(result.predictions)}."
        )
    return int(row_number) - 1


def explain_row(bundle: ModelBundle, result: PredictionResult, row_number: int) -> dict[str, Any]:
    """Build the explanation payload of one predicted row.

    Args:
        bundle: The bundle that produced ``result``.
        result: Prediction result holding the coerced frame and row flags.
        row_number: 1-based row number as shown in the predictions table.

    Returns:
        A JSON-safe payload (schema version 1.0).

    Raises:
        ExplanationError: When the row number is invalid.
    """
    params = ExplainParams()
    position = _position(result, row_number)
    prediction = result.predictions.iloc[position]
    row = result.frame.iloc[[position]]
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
    evidence = _evidence(
        row, top, bundle.reference, str(prediction["predicted_label"]), params.evidence_tokens
    )
    flags = _flags(result, position)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "row_number": int(prediction["row_number"]),
        "source_row": int(prediction["source_row"]),
        "input": {c: prediction[c] for c in INPUT_COLUMNS},
        "prediction": {
            "label": prediction["predicted_label"],
            "probability": round(float(prediction["confidence"]), params.display_decimals),
            "runner_up": {
                "label": prediction["runner_up_label"],
                "probability": round(
                    float(prediction["runner_up_probability"]), params.display_decimals
                ),
            },
            "logit_margin": float(parts.logits[0, k] - parts.logits[0, r]),
            "review_required": bool(prediction["review_required"]),
        },
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
        "evidence": evidence,
        "flags": flags,
        "context": _context(bundle.reference, row),
    }
    payload["fallback_text"] = render_fallback_text(payload)
    return to_jsonable(payload)


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


def _toward(contribution: float, payload: dict[str, Any]) -> str:
    prediction = payload["prediction"]
    winner = prediction["label"] if contribution >= 0 else prediction["runner_up"]["label"]
    return f"toward {winner}"


def render_fallback_text(payload: dict[str, Any]) -> str:
    """Render the deterministic explanation used when no LLM is available.

    Args:
        payload: An explanation payload (``fallback_text`` not required).

    Returns:
        Four short sentences: prediction, drivers, evidence, and flags.
    """
    prediction = payload["prediction"]
    label, runner = prediction["label"], prediction["runner_up"]
    drivers = ", ".join(
        f"{g['label']} ({_toward(g['contribution'], payload)})"
        for g in payload["contributions"]["by_group"][:3]
    )
    lines = [
        f"Predicted {label} with {prediction['probability']:.0%} confidence "
        f"(runner-up {runner['label']}, {runner['probability']:.0%}).",
        _base_rate_sentence(payload),
        f"Strongest drivers: {drivers}.",
        _evidence_sentence(payload),
        _flag_sentence(payload),
    ]
    return " ".join(line for line in lines if line)


def _base_rate_sentence(payload: dict[str, Any]) -> str:
    """Say so when the class base rate, not the row's values, carries the prediction."""
    contributions = payload["contributions"]
    if contributions["bias"] <= contributions["total"]:
        return ""
    label = payload["prediction"]["label"]
    prior = payload["context"]["class_prior"][label]
    return (
        f"Most of the margin comes from the base rate: {prior:.0%} of training rows are {label}, "
        "while the row's own values lean slightly the other way."
    )


def _evidence_sentence(payload: dict[str, Any]) -> str:
    label = payload["prediction"]["label"]
    ranking = {g["group"]: abs(g["contribution"]) for g in payload["contributions"]["by_group"]}
    seen = [e for e in payload["evidence"][:3] if e["seen_in_training"]]
    if not seen:
        return "Evidence: this vendor, account, and pair never appeared in training."
    best = max(seen, key=lambda e: ranking.get(e["group"], 0.0))
    count = best["class_counts"].get(label, 0)
    name = GROUP_LABELS[best["group"]]
    total = best["training_rows"]
    return f"Evidence: in training, {count} of {total} rows with this {name} were {label}."


def _flag_sentence(payload: dict[str, Any]) -> str:
    if not payload["flags"]:
        return "No risk flags."
    return "Flags: " + " ".join(f["message"] for f in payload["flags"])
