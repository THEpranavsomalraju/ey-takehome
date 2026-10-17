"""The explanation agent (with a fake Claude client) and the HTTP API.

No test calls the real API: a scripted fake client returns tool calls and
answers, so the tests are free, offline, and deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ledgerclf import agent as agent_module
from ledgerclf.agent import (
    ENGINE_CLAUDE,
    ENGINE_OFFLINE,
    explain_with_agent,
    read_env_file,
    resolve_connection,
    ungrounded_numbers,
)
from ledgerclf.agent_tools import ToolContext, ToolError, run_tool
from ledgerclf.config import AgentParams, Paths, Settings
from ledgerclf.server import ResultCache, create_app
from ledgerclf.service import PredictionService


def _tool_use(name: str, arguments: dict[str, Any], use_id: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=arguments, id=use_id)


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


@dataclass
class FakeClient:
    """Returns scripted responses; records every request."""

    script: list[list[SimpleNamespace]]
    stop_reasons: list[str] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot
        step = len(self.requests) - 1
        content = self.script[min(step, len(self.script) - 1)]
        stop = (
            self.stop_reasons[step]
            if step < len(self.stop_reasons)
            else ("tool_use" if any(b.type == "tool_use" for b in content) else "end_turn")
        )
        return SimpleNamespace(content=content, stop_reason=stop)


def _grounded_answer(payload: dict[str, Any]) -> str:
    label = payload["prediction"]["label"]
    percent = round(payload["prediction"]["probability"] * 100)
    return (
        f"The model predicts {label} with {percent}% confidence, driven by the GL account (Col6)."
    )


def test_claude_engine_runs_tools_and_returns_grounded_text(service, holdout_result):
    payload = service.explain(holdout_result, 4)
    client = FakeClient(
        [
            [_tool_use("get_prediction_explanation", {}, "t1")],
            [
                _tool_use(
                    "lookup_training_evidence",
                    {"kind": "vendor", "value": payload["input"]["Col1"]},
                    "t2",
                ),
                _tool_use("find_similar_training_rows", {}, "t3"),
            ],
            [_text(_grounded_answer(payload))],
        ]
    )
    answer = explain_with_agent(service, holdout_result, 4, client=client)
    assert answer.engine == ENGINE_CLAUDE and answer.fallback_reason is None
    assert [c["tool"] for c in answer.tool_calls] == [
        "get_prediction_explanation",
        "lookup_training_evidence",
        "find_similar_training_rows",
    ]
    request = client.requests[0]
    assert request["model"] == "claude-haiku-5-5"
    assert "temperature" not in request
    assert request["extra_body"] == {"output_config": {"effort": "low"}}
    second = client.requests[1]["messages"]
    assert second[-1]["content"][0]["type"] == "tool_result"
    assert (
        json.loads(second[-1]["content"][0]["content"])["prediction"]["label"]
        == payload["prediction"]["label"]
    )
    assert (
        len(client.requests[2]["messages"][-1]["content"]) == 2
    )  # parallel tool results in one message


def test_ungrounded_answer_falls_back_to_offline(service, holdout_result):
    client = FakeClient(
        [
            [_tool_use("get_prediction_explanation", {}, "t1")],
            [_text("Confidence is 42.7% because of 1234 similar rows.")],
        ]
    )
    answer = explain_with_agent(service, holdout_result, 4, client=client)
    assert answer.engine == ENGINE_OFFLINE
    assert "not found in the tool results" in answer.fallback_reason


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_refusal_or_truncation_falls_back(service, holdout_result, stop):
    client = FakeClient([[_text("partial")]], stop_reasons=[stop])
    answer = explain_with_agent(service, holdout_result, 2, client=client)
    assert answer.engine == ENGINE_OFFLINE and stop in answer.fallback_reason


def test_endless_tool_loop_stops_at_turn_limit(service, holdout_result):
    client = FakeClient([[_tool_use("get_prediction_explanation", {}, "t")]])
    answer = explain_with_agent(service, holdout_result, 2, client=client)
    assert answer.engine == ENGINE_OFFLINE and "turns" in answer.fallback_reason
    assert len(client.requests) == service.settings.agent.max_turns


def test_client_errors_fall_back(service, holdout_result):
    class Broken:
        messages = SimpleNamespace(
            create=lambda **kwargs: (_ for _ in ()).throw(ConnectionError("down"))
        )

    answer = explain_with_agent(service, holdout_result, 2, client=Broken())
    assert answer.engine == ENGINE_OFFLINE and "unreachable" in answer.fallback_reason


def test_bad_tool_arguments_return_errors_to_the_model(service, holdout_result):
    payload = service.explain(holdout_result, 3)
    client = FakeClient(
        [
            [_tool_use("what_if", {"column": "Col2", "value": "x"}, "t1")],
            [_tool_use("get_prediction_explanation", {}, "t2")],
            [_text(_grounded_answer(payload))],
        ]
    )
    answer = explain_with_agent(service, holdout_result, 3, client=client)
    assert answer.engine == ENGINE_CLAUDE
    assert "error" in answer.tool_calls[0]
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True


def test_offline_engine_uses_the_same_tools(service, holdout_result):
    answer = explain_with_agent(service, holdout_result, 5, use_llm=False)
    assert answer.engine == ENGINE_OFFLINE and answer.fallback_reason is None
    assert answer.text.startswith("Predicted Category_")
    assert "most similar training rows" in answer.text
    assert [c["tool"] for c in answer.tool_calls] == [
        "get_prediction_explanation",
        "find_similar_training_rows",
    ]


def test_tools_answer_from_model_and_reference(service, holdout_result):
    context = ToolContext(service, holdout_result, 1)
    vendor = context.payload["input"]["Col1"]
    evidence = run_tool(
        context, "lookup_training_evidence", {"kind": "vendor", "value": vendor.upper()}
    )
    assert evidence["training_rows"] == sum(evidence["class_counts"].values())
    similar = run_tool(context, "find_similar_training_rows", {})
    assert len(similar["rows"]) == service.settings.agent.similar_rows
    assert all(0 <= r["similarity"] <= 1.0000001 for r in similar["rows"])
    changed = run_tool(context, "what_if", {"column": "Col3", "value": "1680"})
    assert abs(sum(changed["new_probabilities"].values()) - 1) < 1e-3
    for name, arguments in (
        ("nope", {}),
        ("lookup_training_evidence", {"kind": "planet", "value": "x"}),
        ("what_if", {"column": "Col5", "value": "not a date"}),
        ("what_if", {"column": "Col3", "value": "abc"}),
    ):
        with pytest.raises(ToolError):
            run_tool(context, name, arguments)
    pair = run_tool(
        context,
        "lookup_training_evidence",
        {"kind": "pair", "value": f"{vendor} || {context.payload['input']['Col6']}"},
    )
    token = run_tool(
        context, "lookup_training_evidence", {"kind": "account_token", "value": "Word563"}
    )
    assert pair["kind"] == "pair" and token["seen_in_training"]


def test_grounding_check_rules():
    calls = [
        {
            "tool": "x",
            "input": {},
            "output": {"probability": 0.9511, "rows": 47, "count": "42 of 47"},
        }
    ]
    assert (
        ungrounded_numbers(
            "Confidence 95% with 42 of 47 rows; Category_2 and Col6 and Word563.", calls, 1.0
        )
        == []
    )
    assert ungrounded_numbers("Confidence 80%.", calls, 1.0) == ["80%"]
    assert ungrounded_numbers("It saw 1000 rows.", calls, 1.0) == ["1000"]


def test_grounding_reads_thousands_separators():
    calls = [{"tool": "x", "input": {}, "output": {"amount": 1680000.5, "rows": 1680}}]
    assert ungrounded_numbers("An amount of 1,680,000.50 across 1,680 rows.", calls, 1.0) == []
    assert ungrounded_numbers("An amount of 1,690,000.50.", calls, 1.0) == ["1,690,000.50"]


def test_local_tool_defect_becomes_a_tool_error_not_an_outage(service, holdout_result, monkeypatch):
    monkeypatch.setattr(agent_module, "run_tool", lambda *args: {}["missing"])
    calls: list[dict[str, Any]] = []
    use = _tool_use("what_if", {"column": "Col3", "value": "1"}, "t1")
    context = ToolContext(service, holdout_result, 1)
    result = agent_module.ClaudeEngine._execute(context, use, calls)
    assert result["is_error"] and "tool failed" in result["content"]


def test_connection_resolution_order(tmp_path, monkeypatch):
    settings = Settings(
        paths=Paths.from_root(tmp_path),
        agent=AgentParams(relay_url="https://relay.example/api", relay_token="t"),
    )
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_WORKSPACE_ID",
        "LEDGERCLF_RELAY_URL",
        "LEDGERCLF_RELAY_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    relay = resolve_connection(settings)
    assert (
        relay.route == "relay"
        and relay.base_url == "https://relay.example/api"
        and relay.headers == {"x-relay-token": "t"}
    )
    (tmp_path / ".env").write_text(
        "# comment\nANTHROPIC_API_KEY='sk-test'\nANTHROPIC_WORKSPACE_ID=wrkspc_1\n",
        encoding="utf-8",
    )
    direct = resolve_connection(settings)
    assert (
        direct.route == "direct"
        and direct.api_key == "sk-test"
        and direct.headers == {"anthropic-workspace-id": "wrkspc_1"}
    )
    assert resolve_connection(settings, override_key="sk-session").api_key == "sk-session"
    assert read_env_file(tmp_path / "missing") == {}


def test_default_route_is_the_relay_without_any_key(tmp_path, monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "LEDGERCLF_RELAY_URL", "LEDGERCLF_RELAY_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    connection = resolve_connection(Settings(paths=Paths.from_root(tmp_path / "empty")))
    assert connection is not None and connection.route == "relay"
    assert connection.base_url == AgentParams().relay_url


def test_offline_when_nothing_configured(service, holdout_result, monkeypatch):
    monkeypatch.setattr(
        agent_module, "resolve_connection", lambda settings, override_key=None: None
    )
    answer = explain_with_agent(service, holdout_result, 1)
    assert answer.engine == ENGINE_OFFLINE and "No API key" in answer.fallback_reason


@pytest.fixture(scope="module")
def api(service):
    return TestClient(create_app(service))


def test_api_predict_page_explain_download(api, holdout_xlsx, monkeypatch):
    monkeypatch.setattr(
        agent_module, "resolve_connection", lambda settings, override_key=None: None
    )
    response = api.post("/api/predict", files={"file": ("holdout_features.xlsx", holdout_xlsx)})
    assert response.status_code == 200
    body = response.json()
    upload = body["upload_id"]
    assert body["summary"]["rows"] == 590 and len(body["page"]["rows"]) == 25
    page = api.get(f"/api/predict/{upload}/rows", params={"offset": 580, "limit": 25}).json()
    assert page["total"] == 590 and len(page["rows"]) == 10 and page["rows"][0]["row_number"] == 581
    explained = api.post("/api/explain", json={"upload_id": upload, "row_number": 2}).json()
    assert explained["answer"]["engine"] == "offline" and explained["payload"]["row_number"] == 2
    download = api.get(f"/api/predict/{upload}/download")
    assert (
        download.status_code == 200
        and len(pd.read_csv(pd.io.common.BytesIO(download.content), encoding="utf-8-sig")) == 590
    )


def test_api_errors_are_clear_json(api):
    bad = api.post("/api/predict", files={"file": ("notes.pdf", b"%PDF-1.4 x")})
    assert bad.status_code == 422 and bad.json()["code"] == "UNSUPPORTED_FORMAT"
    assert api.get("/api/predict/unknown/rows").status_code == 404
    assert (
        api.post("/api/explain", json={"upload_id": "unknown", "row_number": 1}).status_code == 404
    )
    invalid = api.post("/api/explain", json={"upload_id": "x", "row_number": 0})
    assert invalid.status_code == 422 and invalid.json()["code"] == "INVALID_REQUEST"
    missing = api.get("/api/predict/unknown/rows").json()
    assert missing["code"] == "NOT_FOUND" and "expired" in missing["message"]


def test_api_unexpected_errors_are_json(service, monkeypatch):
    client = TestClient(create_app(service), raise_server_exceptions=False)
    monkeypatch.setattr(service, "model_info", lambda: 1 / 0)
    response = client.get("/api/model")
    assert response.status_code == 500 and response.json()["code"] == "INTERNAL"


def test_api_missing_sample_file_says_how_to_fix(service, tmp_path):
    empty = Settings(paths=Paths.from_root(tmp_path))
    client = TestClient(create_app(PredictionService(service.bundle, None, empty)))
    for response in (client.post("/api/predict/sample"), client.get("/api/sample/holdout")):
        assert response.status_code == 503
        assert response.json()["code"] == "SAMPLE_MISSING"
        assert "python -m ledgerclf split" in response.json()["message"]


def test_serve_refuses_a_port_in_use():
    import socket

    from ledgerclf.exceptions import ConfigError
    from ledgerclf.server import _require_free_port

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        with pytest.raises(ConfigError, match="--port"):
            _require_free_port("127.0.0.1", taken.getsockname()[1])


def test_api_reports_model_status_sample(api):
    assert api.get("/api/health").json()["exit_code"] == 0
    assert api.get("/api/model").json()["model_used"] == "champion"
    assert any(
        r["file"] == "selection/champion_decision.json" for r in api.get("/api/reports").json()
    )
    table = api.get("/api/reports/eda/class_distribution.csv").json()
    assert table["columns"] == ["class", "rows", "share"]
    assert "trail" in api.get("/api/reports/selection/champion_decision.json").json()
    assert api.get("/api/agent/status").json()["route"] in ("direct", "relay", "offline")
    sample = api.post("/api/predict/sample").json()
    assert sample["summary"]["rows"] == 590
    assert api.get("/api/sample/holdout").status_code == 200
    assert api.get("/").status_code == 200


def test_result_cache_evicts_oldest(holdout_result):
    cache = ResultCache(capacity=2)
    ids = [cache.add(holdout_result) for _ in range(3)]
    with pytest.raises(HTTPException):
        cache.get(ids[0])
    assert cache.get(ids[2]) is holdout_result


def test_labeled_holdout_file_is_predicted_without_reading_labels(service, settings):
    data = (settings.paths.holdout / "holdout_with_labels.xlsx").read_bytes()
    result = service.predict_file(data, "holdout_with_labels.xlsx")
    assert len(result.predictions) == 590
    assert result.metrics is None
    assert "true_label" not in result.predictions.columns
    assert "HOLDOUT_METRICS_WITHHELD" in [i.code for i in result.report.issues]


def test_holdout_flag_counts_use_exact_codes(settings, tmp_path):
    from ledgerclf.holdout_eval import _score_model

    predictions = pd.DataFrame(
        {
            "predicted_label": ["Category_1"] * 3,
            "flags": ["UNSEEN_VENDOR_ACCOUNT_PAIR", "UNSEEN_VENDOR|LOW_CONFIDENCE", ""],
            "review_required": [False, True, False],
            **{f"p_Category_{i}": [1.0 if i == 1 else 0.0] * 3 for i in range(1, 7)},
        }
    )
    custom = Settings(paths=Paths.from_root(tmp_path))
    scored = _score_model(
        custom, "fallback_logistic", predictions, np.array(["Category_1"] * 3, dtype=object)
    )
    counts = {row["flag"]: row["rows"] for row in scored["accuracy_by_flag"]}
    assert counts == {"LOW_CONFIDENCE": 1, "UNSEEN_VENDOR": 1, "UNSEEN_VENDOR_ACCOUNT_PAIR": 1}


def test_relay_accepts_exactly_the_agent_system_prompt():
    import hashlib
    from pathlib import Path

    relay = (
        Path(agent_module.__file__).resolve().parent.parent / "relay" / "api" / "v1" / "messages.js"
    )
    digest = hashlib.sha256(agent_module.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert f'SYSTEM_PROMPT_SHA256 = "{digest}"' in relay.read_text(encoding="utf-8")


def test_download_neutralizes_spreadsheet_formulas(api):
    rows = "Col1,Col2,Col3,Col4,Col5,Col6,Col7\n"
    rows += '"=HYPERLINK(""http://x"")",1,-617.60,@SUM(A1),2018-08-31,+cmd,NoDoc\n'
    upload = api.post("/api/predict", files={"file": ("f.csv", rows.encode("utf-8"))}).json()
    text = api.get(f"/api/predict/{upload['upload_id']}/download").content.decode("utf-8-sig")
    assert "'=HYPERLINK" in text and "'@SUM(A1)" in text and "'+cmd" in text
    assert ",-617.60," in text
