"""Chart explanations: grounded in each chart's report data, with an offline fallback.

A fake client stands in for Claude, so the tests are free, offline, and deterministic.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ledgerclf.chart_explainer import (
    CHART_SYSTEM_PROMPT,
    CHARTS,
    ChartAnswer,
    ChartAnswerCache,
    ChartExplainError,
    chart_data,
    explain_chart,
)
from ledgerclf.server import create_app


class FakeClaude:
    """Returns a fixed answer and records the request."""

    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.text, self.stop_reason, self.requests = text, stop_reason, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(
            stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=self.text)]
        )


@pytest.mark.parametrize("chart_id", sorted(c for c, s in CHARTS.items() if not s.upload))
def test_every_report_chart_has_data(service, chart_id):
    data = chart_data(service, chart_id, None, service.settings.agent.chart_max_rows)
    reports = {k: v for k, v in data.items() if k != "totals"}
    assert reports and all(reports.values())


def test_upload_charts_use_the_upload(service, holdout_result):
    for chart_id in ("upload-categories", "upload-confidence", "upload-warnings"):
        data = chart_data(service, chart_id, holdout_result, 60)
        assert data["rows_in_file"] == 590
    with pytest.raises(ChartExplainError):
        chart_data(service, "upload-categories", None, 60)
    with pytest.raises(ChartExplainError):
        chart_data(service, "no-such-chart", None, 60)


def test_grounded_answer_is_returned_with_the_chart_prompt(service):
    client = FakeClaude("**What you're looking at:** 88.4% of rows are Category_1 (4,694 rows).")
    answer = explain_chart(service, "category-balance", "Log scale", None, client=client)
    assert answer.engine == "claude" and answer.text.startswith("**What you're looking at:**")
    request = client.requests[0]
    assert request["system"] == CHART_SYSTEM_PROMPT and "tools" not in request
    assert "Current view: Log scale" in request["messages"][0]["content"]


def test_invented_numbers_fall_back_to_the_built_in_summary(service):
    client = FakeClaude("Category_1 is 91.7% of rows.")
    answer = explain_chart(service, "category-balance", "", None, client=client)
    assert answer.text is None and answer.engine == "offline"
    assert "91.7%" in answer.fallback_reason


def test_truncated_answer_and_offline_mode(service):
    truncated = FakeClaude("partial", stop_reason="max_tokens")
    assert explain_chart(service, "ablation", "", None, client=truncated).text is None
    offline = explain_chart(service, "ablation", "", None, use_llm=False)
    assert offline.engine == "offline" and offline.fallback_reason is None


def test_cache_keeps_only_claude_answers():
    cache = ChartAnswerCache(capacity=1)
    cache.put(("a",), ChartAnswer(None, "offline", None, "x"))
    assert cache.get(("a",)) is None
    cache.put(("a",), ChartAnswer("text", "claude", "m", None))
    cache.put(("b",), ChartAnswer("text", "claude", "m", None))
    assert cache.get(("a",)) is None and cache.get(("b",)) is not None


def test_api_explain_chart_offline_and_errors(service):
    api = TestClient(create_app(service))
    ok = api.post("/api/explain-chart", json={"chart": "confusion", "use_llm": False})
    assert ok.status_code == 200 and ok.json()["engine"] == "offline"
    unknown = api.post("/api/explain-chart", json={"chart": "nope", "use_llm": False})
    assert unknown.status_code == 422 and unknown.json()["code"] == "CHART_UNKNOWN"
    needs = api.post("/api/explain-chart", json={"chart": "upload-confidence", "use_llm": False})
    assert needs.json()["code"] == "CHART_NEEDS_UPLOAD"
