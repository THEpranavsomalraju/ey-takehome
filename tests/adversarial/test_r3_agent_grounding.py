"""Claude's answer must cite only numbers from tool results, and bad tool input goes back to it.

``agent.ungrounded_numbers`` promises to return "numbers cited in text that no
tool result supports", and the system prompt says "Use only numbers that appear
in tool results". ``_tool_numbers`` also harvests the model's own inputs,
including inputs of calls that failed, so any invented number becomes
"grounded" after one deliberately failing call.

``ToolError``'s contract: "A tool call had invalid arguments; the message goes
back to the model." A tool_use whose input is not an object (the model sent a
JSON string) fails in ``dict(use.input)`` outside the try block, so the whole
Claude answer is dropped instead.

A scripted fake client stands in for Claude; nothing touches the network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ledgerclf.agent import ENGINE_CLAUDE, ENGINE_OFFLINE, explain_with_agent
from tests.adversarial._support import ROW_A, csv_text


class ScriptedClient:
    """Return one scripted response per call; record the messages sent."""

    def __init__(self, script: list[list[SimpleNamespace]]) -> None:
        self.script = script
        self.sent: list[list[dict[str, Any]]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.sent.append(list(kwargs["messages"]))
        content = self.script[min(len(self.sent) - 1, len(self.script) - 1)]
        uses = any(block.type == "tool_use" for block in content)
        return SimpleNamespace(content=content, stop_reason="tool_use" if uses else "end_turn")


def _use(name: str, arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=arguments, id="t1")


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _result(service):
    return service.predict_file(csv_text([ROW_A]).encode("utf-8"), "one_row.csv")


def test_number_from_a_failed_tool_call_input_is_not_grounding(service):
    client = ScriptedClient(
        [
            [_use("lookup_training_evidence", {"kind": "audited 97.3", "value": "x"})],
            [_text("An audit found 97.3% of similar rows were Category_1.")],
        ]
    )
    answer = explain_with_agent(service, _result(service), 1, client=client)
    assert answer.engine == ENGINE_OFFLINE, f"invented number accepted: {answer.text!r}"
    assert "97.3" in (answer.fallback_reason or "")


def test_tool_input_that_is_not_an_object_goes_back_to_claude_as_a_tool_error(service):
    client = ScriptedClient(
        [
            [_use("get_prediction_explanation", '{"row": 1}')],
            [_text("The model predicts this row as shown in the table.")],
        ]
    )
    answer = explain_with_agent(service, _result(service), 1, client=client)
    assert answer.engine == ENGINE_CLAUDE, answer.fallback_reason
    tool_results = client.sent[1][-1]["content"]
    assert tool_results[0]["is_error"] is True
