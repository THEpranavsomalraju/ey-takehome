"""Plain-language explanations of the dashboard's charts, on request.

Every chart has an entry in ``CHARTS`` naming the report files (or the upload
summary) it draws. The explanation is built only from that data, read on the
server, and every number Claude quotes is checked against it. When Claude is
not reachable the answer is empty and the dashboard shows the chart's own
built-in summary instead.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ledgerclf.agent import (
    ENGINE_CLAUDE,
    ENGINE_OFFLINE,
    failure_reason,
    make_client,
    resolve_connection,
    ungrounded_numbers,
)
from ledgerclf.exceptions import LedgerClfError
from ledgerclf.predict import PredictionResult
from ledgerclf.table_views import insights

logger = logging.getLogger(__name__)

CHART_SYSTEM_PROMPT = """You explain one chart from a dashboard about an invoice classification \
model. Readers are business people seeing it for the first time and AI engineers who want the \
details. The model suggests one of six categories (Category_1 to Category_6) for each accounts \
payable invoice line, from the vendor (Col1), amount (Col3), line description (Col4), posting \
month (Col5), GL account (Col6), and document flag (Col7). Category names and text values are \
anonymized, so never guess what they mean.

You receive the chart's title, what it shows, the reader's current view, and the exact data \
behind it as JSON, including precomputed totals. Quote only numbers that appear in that data \
(rounding is fine, and a share such as 0.954 may be written as 95.4%). Never add, subtract, or \
compute a new figure yourself; describe it in words instead.

Answer in Markdown in exactly this layout, at most 140 words:
**What you're looking at:** One or two sentences in everyday language.

**What it tells us:**
- Two or three bullets with the key numbers.

**Why it matters:** One sentence for a business reader. Then "Technical note:" and one short \
sentence for an engineer."""


@dataclass(frozen=True)
class ChartSpec:
    """One dashboard chart: its title, what it shows, and the data behind it."""

    title: str
    about: str
    files: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    upload: str | None = None  # which part of an upload's insights the chart draws


CHARTS: dict[str, ChartSpec] = {
    "category-balance": ChartSpec(
        "Category balance",
        "Training rows per category; the largest share is the accuracy of always guessing it.",
        ("eda/class_distribution.csv",),
    ),
    "monthly-volume": ChartSpec(
        "When each category occurs",
        "Training rows per posting month and category.",
        ("eda/period_by_class.csv",),
    ),
    "input-signal": ChartSpec(
        "Which inputs carry the label",
        "Association of each input with the category: Cramer's V (column 'label') and "
        "mutual information in nats.",
        ("eda/cramers_v.csv", "eda/mutual_information.csv"),
        ("variable", "label", "mutual_information_nats"),
    ),
    "category-profiles": ChartSpec(
        "What each category looks like",
        "Profile of each anonymized category in the training rows: size, top vendor and GL "
        "account with their shares, document flag, median amount, credit share, months.",
        ("eda/category_profiles.csv",),
    ),
    "repeated-rows": ChartSpec(
        "Repeated rows",
        "Rows identical on every model input except the invoice number, and the best accuracy "
        "possible if every repeated pattern were memorized.",
        ("eda/duplicates_summary.json", "eda/accuracy_ceiling.json"),
    ),
    "test-design": ChartSpec(
        "How the score depends on the test design",
        "The same models' accuracy and macro F1 under random, grouped, unseen-vendor, and "
        "out-of-time testing.",
        ("benchmark/validation_schemes.csv",),
        ("model", "scheme", "accuracy", "macro_f1_5"),
    ),
    "model-comparison": ChartSpec(
        "Model comparison",
        "Grouped cross-validation scores of every candidate model (3 seeds x 5 folds); "
        "logodds_blend_p025 is the final model.",
        ("benchmark/leaderboard.csv",),
        ("model", "accuracy_mean", "accuracy_std", "macro_f1_5_mean", "recall_Category_2"),
    ),
    "ablation": ChartSpec(
        "What each input is worth",
        "Change in accuracy and macro F1 of the logistic model when one input group is removed "
        "(or added).",
        ("benchmark/ablation.csv",),
        ("variant", "delta_accuracy", "delta_macro_f1_5"),
    ),
    "class-weights": ChartSpec(
        "Class weights",
        "Scores as rare categories get more weight (power 0 = none, 1 = balanced); "
        "oot_majority_rows_predicted_rare counts false rare alarms in later months.",
        ("benchmark/class_weight_dial.csv",),
    ),
    "calibration": ChartSpec(
        "Does 90% confident mean 90% right?",
        "Out-of-fold accuracy per confidence bin before and after temperature scaling, and the "
        "calibration error.",
        ("selection/calibration_reliability.csv", "selection/calibration_summary.json"),
        ("bin_low", "rows", "mean_confidence", "accuracy", "model", "stage"),
    ),
    "review-threshold": ChartSpec(
        "Review threshold explorer",
        "For each confidence threshold: share of rows decided automatically (coverage), their "
        "accuracy, and the accuracy of the rows sent to review; chosen marks the threshold used.",
        ("selection/threshold_table.csv",),
    ),
    "warning-reliability": ChartSpec(
        "How reliable is each warning?",
        "Accuracy of the rows carrying each warning flag, out of fold and on the holdout; "
        "critical flags send the row to review.",
        ("selection/accuracy_by_flag.csv", "holdout/holdout_accuracy_by_flag.csv"),
    ),
    "confusion": ChartSpec(
        "Where the model is right and wrong",
        "Confusion matrices (true category by predicted category) on the holdout and out of fold.",
        ("holdout/holdout_confusion_matrix.csv", "selection/oof_confusion_matrix.csv"),
    ),
    "per-class": ChartSpec(
        "Quality per category",
        "Precision, recall, and support per category on the holdout and out of fold.",
        ("holdout/holdout_classification_report.csv", "selection/oof_classification_report.csv"),
    ),
    "feature-pipeline": ChartSpec(
        "Feature pipeline",
        "Feature groups the model computes and how many features each produces.",
        ("preprocessing/feature_space_summary.csv",),
        ("transformer", "group", "features", "mean_nonzero_per_row"),
    ),
    "upload-categories": ChartSpec(
        "Predicted categories (this file)",
        "Rows per predicted category in the uploaded file, and how many need review.",
        upload="by_class",
    ),
    "upload-confidence": ChartSpec(
        "How confident the model is (this file)",
        "Rows per confidence bin in the uploaded file, split into automatic and review.",
        upload="confidence_histogram",
    ),
    "upload-warnings": ChartSpec(
        "Warnings raised (this file)",
        "How often each warning flag fired in the uploaded file.",
        upload="flags",
    ),
}


@dataclass
class ChartAnswer:
    """A chart explanation; ``text`` is None when only the built-in summary is available."""

    text: str | None
    engine: str
    model: str | None
    fallback_reason: str | None
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict for the API."""
        return {
            "text": self.text,
            "engine": self.engine,
            "model": self.model,
            "fallback_reason": self.fallback_reason,
            "seconds": round(self.seconds, 3),
        }


class ChartExplainError(LedgerClfError):
    """The chart is unknown or needs an upload that was not given."""

    default_code = "CHART_UNKNOWN"


def chart_data(
    service: Any, chart_id: str, result: PredictionResult | None, max_rows: int
) -> dict[str, Any]:
    """Return the exact data behind one chart, rounded for a compact prompt.

    Raises:
        ChartExplainError: For an unknown chart or an upload chart without an upload.
    """
    spec = CHARTS.get(chart_id)
    if spec is None:
        raise ChartExplainError(f"Unknown chart '{chart_id}'.")
    if spec.upload:
        if result is None:
            raise ChartExplainError(
                "This chart describes an uploaded file. Upload the file again, then ask.",
                code="CHART_NEEDS_UPLOAD",
            )
        bins = service.settings.server.histogram_bins
        summary = insights(result, service.bundle.policy.threshold, bins)
        rows = len(result.predictions)
        part = summary[spec.upload]
        if spec.upload != "confidence_histogram":  # add each entry's share of the file
            part = [entry | {"share_of_file": round(entry["rows"] / rows, 4)} for entry in part]
        data: dict[str, Any] = {"rows_in_file": rows, spec.upload: part}
    else:
        data = {f: _compact(service.load_report(f), spec.columns, max_rows) for f in spec.files}
    return data | {"totals": _totals(data)}


def _totals(data: dict[str, Any]) -> dict[str, Any]:
    """Row counts and column sums of every table (and accuracy of confusion matrices)."""
    totals: dict[str, Any] = {}
    for name, table in data.items():
        if not isinstance(table, list) or not table or not isinstance(table[0], dict):
            continue
        numeric = [
            key
            for key, value in table[0].items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        entry: dict[str, Any] = {"rows": len(table)}
        entry |= {f"sum_{k}": round(sum(float(r.get(k) or 0) for r in table), 4) for k in numeric}
        if "true_label" in table[0]:
            correct = sum(float(r.get(r["true_label"]) or 0) for r in table)
            total = sum(float(r.get(k) or 0) for r in table for k in numeric)
            entry |= {"correct": correct, "total": total}
            entry["accuracy"] = round(correct / total, 4) if total else None
        totals[name] = entry
    return totals


def _compact(report: pd.DataFrame | dict[str, Any], columns: tuple[str, ...], max_rows: int) -> Any:
    if isinstance(report, pd.DataFrame):
        keep = [c for c in columns if c in report.columns] or list(report.columns)
        table = report[keep].head(max_rows).round(4)
        return json.loads(table.to_json(orient="records"))
    return report


def explain_chart(
    service: Any,
    chart_id: str,
    view: str,
    result: PredictionResult | None,
    use_llm: bool = True,
    override_key: str | None = None,
    client: Any = None,
) -> ChartAnswer:
    """Explain one chart with Claude, grounded in the chart's own data.

    Args:
        service: Loaded PredictionService.
        chart_id: Key of ``CHARTS``.
        view: What the reader currently sees (a toggle or selection), plain text.
        result: The upload, for the upload charts.
        use_llm: False returns the offline answer (the dashboard's built-in summary).
        override_key: Session-only API key from the dashboard.
        client: Injected Anthropic-compatible client (tests).

    Returns:
        The answer; ``text`` is None when Claude was not used.
    """
    started = time.perf_counter()
    params = service.settings.agent
    data = chart_data(service, chart_id, result, params.chart_max_rows)
    if not use_llm:
        return ChartAnswer(None, ENGINE_OFFLINE, None, None, time.perf_counter() - started)
    if client is None:
        connection = resolve_connection(service.settings, override_key)
        client = make_client(connection, params) if connection is not None else None
    if client is None:
        return ChartAnswer(None, ENGINE_OFFLINE, None, "No API key or relay configured.")
    try:
        text = _ask(client, params, CHARTS[chart_id], view, data)
    except ChartExplainError as failure:
        reason: str = failure.message
    except Exception as error:  # boundary: any client or network failure falls back
        logger.debug("Chart explanation failed", exc_info=True)
        reason = failure_reason(error)
    else:
        return ChartAnswer(text, ENGINE_CLAUDE, params.model, None, time.perf_counter() - started)
    return ChartAnswer(None, ENGINE_OFFLINE, None, reason, time.perf_counter() - started)


def _ask(client: Any, params: Any, spec: ChartSpec, view: str, data: dict[str, Any]) -> str:
    view = " ".join(view.split())[: params.max_view_chars] or "default"
    request = (
        f"Chart: {spec.title}\nWhat it shows: {spec.about}\nCurrent view: {view}\n"
        f"Data (JSON):\n{json.dumps(data, separators=(',', ':'))}"
    )
    response = client.messages.create(
        model=params.model,
        max_tokens=params.max_tokens,
        system=CHART_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": request}],
        extra_body={"output_config": {"effort": params.effort}},
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        raise ChartExplainError(
            f"Claude stopped early ({response.stop_reason}).", code="CHART_FALLBACK"
        )
    text = " ".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
    if not text:
        raise ChartExplainError("Claude returned no text.", code="CHART_FALLBACK")
    cited = ungrounded_numbers(
        text, _grounding_calls(data), params.percent_tolerance, params.relative_tolerance
    )
    unsupported = [n for n in cited if not _small_count(n, params.chart_free_integers)]
    if unsupported:
        raise ChartExplainError(
            "Claude cited numbers not found in the chart data: " + ", ".join(unsupported[:5]) + ".",
            code="CHART_FALLBACK",
        )
    return text


def _grounding_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The chart data plus every number times 100, so 0.017 may be quoted as 1.7 points."""
    numbers: list[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int | float):
            numbers.append(float(value) * 100)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    return [{"output": data}, {"output": numbers}]


def _small_count(cited: str, limit: int) -> bool:
    """Small whole numbers ("3 categories", "Category 4") carry no risk of a false figure."""
    return cited.isdigit() and int(cited) <= limit


class ChartAnswerCache:
    """Bounded, thread-safe store of Claude chart answers (repeat clicks cost nothing)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: OrderedDict[tuple[str, ...], ChartAnswer] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[str, ...]) -> ChartAnswer | None:
        """Return a cached answer, or None."""
        with self._lock:
            answer = self._items.get(key)
            if answer is not None:
                self._items.move_to_end(key)
            return answer

    def put(self, key: tuple[str, ...], answer: ChartAnswer) -> None:
        """Store a Claude answer, evicting the oldest beyond capacity."""
        if answer.engine != ENGINE_CLAUDE:
            return
        with self._lock:
            self._items[key] = answer
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
