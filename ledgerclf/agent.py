"""The explanation agent: Claude with read-only tools, or a deterministic offline engine.

Runs only when a user asks for one row's explanation (never preloaded).
Engine order: the user's own ``ANTHROPIC_API_KEY``, then the hosted relay
(the project owner's key stays server-side), then the offline engine. Any
failure, timeout, or ungrounded answer falls back to the offline engine,
which calls the same tools in a fixed order (DECISIONS.md D25).
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ledgerclf.agent_tools import (
    TOOL_DEFINITIONS,
    ToolContext,
    ToolError,
    run_tool,
    tool_result_text,
)
from ledgerclf.config import AgentParams, Settings
from ledgerclf.exceptions import LedgerClfError

logger = logging.getLogger(__name__)

ENGINE_CLAUDE = "claude"
ENGINE_OFFLINE = "offline"
WORKSPACE_HEADER = "anthropic-workspace-id"
RELAY_TOKEN_HEADER = "x-relay-token"
SYSTEM_PROMPT = """You explain one prediction of an accounts payable ledger classifier to a tax \
reviewer. The model assigns each invoice line to Category_1 to Category_6. Vendor, description, \
and account values are anonymized tokens such as Word563, so never guess what a token means.

Always call get_prediction_explanation first. Its contributions are exact: each input column's \
contribution is in log-odds units toward the predicted class versus the runner-up (positive \
supports the prediction). Call lookup_training_evidence, find_similar_training_rows, or what_if \
when they help answer why. Use only numbers that appear in tool results; never estimate.

Write at most 130 words of plain text, no headings, in this order: the prediction and its \
confidence; the two or three inputs that drove it, with direction; the training evidence \
behind them; risk flags and whether the row needs review."""


@dataclass
class AgentAnswer:
    """One explanation answer with its provenance."""

    text: str
    engine: str
    model: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    fallback_reason: str | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict for the API."""
        return {
            "text": self.text,
            "engine": self.engine,
            "model": self.model,
            "tool_calls": self.tool_calls,
            "fallback_reason": self.fallback_reason,
            "seconds": round(self.seconds, 3),
        }


@dataclass(frozen=True)
class ClaudeConnection:
    """How to reach Claude: direct with a key, or through the relay."""

    route: str
    api_key: str
    base_url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


def read_env_file(root: Path) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from ``<root>/.env`` (missing file: empty dict)."""
    path = root / ".env"
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_connection(
    settings: Settings, override_key: str | None = None
) -> ClaudeConnection | None:
    """Choose how to reach Claude, or None for the offline engine.

    Args:
        settings: Project settings (relay URL and token).
        override_key: A key typed into the dashboard for this session only.

    Returns:
        The connection, or None when no key and no relay are configured.
    """
    env = {**read_env_file(settings.paths.root), **os.environ}
    key = override_key or env.get("ANTHROPIC_API_KEY", "")
    if key:
        workspace = env.get("ANTHROPIC_WORKSPACE_ID", "")
        headers = {WORKSPACE_HEADER: workspace} if workspace else {}
        return ClaudeConnection("direct", key, None, headers)
    relay_url = env.get("LEDGERCLF_RELAY_URL", settings.agent.relay_url)
    if relay_url:
        token = env.get("LEDGERCLF_RELAY_TOKEN", settings.agent.relay_token)
        return ClaudeConnection("relay", "relay", relay_url, {RELAY_TOKEN_HEADER: token})
    return None


def _client(connection: ClaudeConnection, params: AgentParams) -> Any:
    try:
        import truststore

        truststore.inject_into_ssl()  # use the operating system's certificates (corporate proxies)
    except ImportError:
        logger.debug("truststore not installed; using bundled certificates")
    import anthropic

    return anthropic.Anthropic(
        api_key=connection.api_key,
        base_url=connection.base_url,
        default_headers=connection.headers or None,
        timeout=params.timeout_seconds,
        max_retries=1,
    )


def explain_with_agent(
    service: Any,
    result: Any,
    row_number: int,
    use_llm: bool = True,
    override_key: str | None = None,
    client: Any = None,
) -> AgentAnswer:
    """Explain one predicted row with the best available engine.

    Args:
        service: Loaded PredictionService.
        result: Prediction result holding the row.
        row_number: 1-based row number.
        use_llm: False forces the offline engine.
        override_key: Session-only API key from the dashboard.
        client: Injected Anthropic-compatible client (tests).

    Returns:
        The answer, with the engine used and every tool call.
    """
    started = time.perf_counter()
    context = ToolContext(service, result, row_number)
    reason: str | None
    params = service.settings.agent
    connection = (
        resolve_connection(service.settings, override_key) if use_llm and client is None else None
    )
    if use_llm and (client is not None or connection is not None):
        try:
            answer = ClaudeEngine(client or _client(connection, params), params).run(context)  # type: ignore[arg-type]
            answer.seconds = time.perf_counter() - started
            return answer
        except AgentFallbackError as failure:
            reason = failure.message
        except Exception as error:  # boundary: any client or network failure falls back
            logger.debug("Claude engine failed", exc_info=True)
            reason = f"Claude was unreachable ({type(error).__name__})."
    else:
        reason = None if not use_llm else "No API key or relay configured."
    answer = offline_answer(context)
    answer.fallback_reason = reason
    answer.seconds = time.perf_counter() - started
    return answer


class AgentFallbackError(LedgerClfError):
    """The LLM answer could not be used; the offline engine answers instead."""

    default_code = "AGENT_FALLBACK"


class ClaudeEngine:
    """Manual tool-use loop with a turn limit and a grounding check."""

    def __init__(self, client: Any, params: AgentParams) -> None:
        self.client = client
        self.params = params

    def run(self, context: ToolContext) -> AgentAnswer:
        """Run the loop until Claude answers, then verify the numbers it cites.

        Raises:
            AgentFallbackError: On refusal, truncation, turn limit, or ungrounded numbers.
        """
        request = f"Explain why row {context.row_number} received its prediction."
        messages: list[dict[str, Any]] = [{"role": "user", "content": request}]
        calls: list[dict[str, Any]] = []
        for _ in range(self.params.max_turns):
            response = self.client.messages.create(
                model=self.params.model,
                max_tokens=self.params.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
                extra_body={"output_config": {"effort": self.params.effort}},
            )
            if response.stop_reason in ("refusal", "max_tokens"):
                raise AgentFallbackError(f"Claude stopped early ({response.stop_reason}).")
            uses = [b for b in response.content if getattr(b, "type", "") == "tool_use"]
            if not uses:
                text = " ".join(
                    b.text for b in response.content if getattr(b, "type", "") == "text"
                ).strip()
                return self._verified(text, calls)
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {"role": "user", "content": [self._execute(context, use, calls) for use in uses]}
            )
        raise AgentFallbackError(f"Claude did not finish within {self.params.max_turns} turns.")

    @staticmethod
    def _execute(context: ToolContext, use: Any, calls: list[dict[str, Any]]) -> dict[str, Any]:
        arguments = dict(use.input or {})
        try:
            output = run_tool(context, use.name, arguments)
            calls.append({"tool": use.name, "input": arguments, "output": output})
            return {
                "type": "tool_result",
                "tool_use_id": use.id,
                "content": tool_result_text(output),
            }
        except (ToolError, LedgerClfError) as error:
            calls.append({"tool": use.name, "input": arguments, "error": error.message})
            return {
                "type": "tool_result",
                "tool_use_id": use.id,
                "content": f"Error: {error.message}",
                "is_error": True,
            }

    def _verified(self, text: str, calls: list[dict[str, Any]]) -> AgentAnswer:
        if not text:
            raise AgentFallbackError("Claude returned no text.")
        unsupported = ungrounded_numbers(text, calls, self.params.percent_tolerance)
        if unsupported:
            raise AgentFallbackError(
                "Claude cited numbers not found in the tool results: "
                + ", ".join(unsupported[:5])
                + "."
            )
        return AgentAnswer(text, ENGINE_CLAUDE, self.params.model, calls)


_NUMBER = re.compile(r"(?<![\w.])(-?\d+(?:[.,]\d+)?)(\s*%)?")
_IGNORED_CONTEXT = re.compile(r"(Category_|Col|Word|Doc|row )\s*$", re.IGNORECASE)


def _tool_numbers(calls: list[dict[str, Any]]) -> list[float]:
    numbers: list[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int | float):
            numbers.append(float(value))
        elif isinstance(value, str):
            numbers.extend(float(m.group(1).replace(",", "")) for m in _NUMBER.finditer(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for call in calls:
        walk(call.get("output", {}))
        walk(call.get("input", {}))
    return numbers


def ungrounded_numbers(
    text: str, calls: list[dict[str, Any]], percent_tolerance: float
) -> list[str]:
    """Return numbers cited in ``text`` that no tool result supports.

    Percentages may match a tool probability or share (0 to 1) within the
    tolerance in percentage points; other numbers must match a tool number
    after rounding to the cited precision. Labels, column names, tokens,
    and row numbers are ignored.
    """
    known = _tool_numbers(calls)
    unsupported = []
    for match in _NUMBER.finditer(text):
        if _IGNORED_CONTEXT.search(text[: match.start()]):
            continue
        raw, percent = match.group(1).replace(",", ""), bool(match.group(2))
        value = float(raw)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if percent:
            ok = any(
                abs(value - k * 100) <= percent_tolerance or abs(value - k) <= percent_tolerance
                for k in known
            )
        else:
            tolerance = 0.5 * 10 ** (-decimals) + 1e-9
            ok = any(
                abs(value - k) <= tolerance or abs(abs(value) - abs(k)) <= tolerance for k in known
            )
        if not ok:
            unsupported.append(match.group(0).strip())
    return unsupported


def offline_answer(context: ToolContext) -> AgentAnswer:
    """Deterministic explanation from the same tools, used without an LLM.

    Calls the explanation tool, the evidence tool for the strongest seen
    value, and the similar-rows tool, then fills a fixed template.
    """
    calls: list[dict[str, Any]] = []
    explanation = run_tool(context, "get_prediction_explanation", {})
    calls.append({"tool": "get_prediction_explanation", "input": {}, "output": explanation})
    lines = [context.payload["fallback_text"]]
    try:
        similar = run_tool(context, "find_similar_training_rows", {})
        calls.append({"tool": "find_similar_training_rows", "input": {}, "output": similar})
        lines.append(_similar_sentence(similar, explanation["prediction"]["label"]))
    except LedgerClfError as error:
        calls.append({"tool": "find_similar_training_rows", "input": {}, "error": error.message})
    return AgentAnswer(" ".join(lines), ENGINE_OFFLINE, None, calls)


def _similar_sentence(similar: dict[str, Any], label: str) -> str:
    rows = similar["rows"]
    matching = sum(r["label"] == label for r in rows)
    best = rows[0]["similarity"] if rows else 0.0
    return (
        f"Of the {len(rows)} most similar training rows (best similarity {best:.2f}), "
        f"{matching} were labeled {label}."
    )
