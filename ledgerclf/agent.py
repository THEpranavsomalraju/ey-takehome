"""The explanation agent: Claude with read-only tools, or a deterministic offline engine.

Runs only when a user asks for one row's explanation (never preloaded).
Engine order: the user's own ``ANTHROPIC_API_KEY``, then the hosted relay
(the project owner's key stays server-side), then the offline engine. Any
failure, timeout, or ungrounded answer falls back to the offline engine,
which calls the same tools in a fixed order (DECISIONS.md D25).
"""

from __future__ import annotations

import functools
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
from ledgerclf.explain import render_fallback_text

logger = logging.getLogger(__name__)

ENGINE_CLAUDE = "claude"
ENGINE_OFFLINE = "offline"
WORKSPACE_HEADER = "anthropic-workspace-id"
RELAY_TOKEN_HEADER = "x-relay-token"
SYSTEM_PROMPT = """You explain one prediction of an invoice classification model to a client's \
finance team. Readers range from analysts to managers without data science training: give them a \
plain explanation first, then the exact facts.

The model reads one accounts payable invoice line and suggests one of six categories (Category_1 \
to Category_6). Inputs: Col1 vendor, Col3 amount, Col4 line description, Col5 posting date, Col6 \
GL account, Col7 document flag with values such as NoDoc and Doc1 (Col2, the invoice number, is \
not used). Text values are anonymized tokens such as Word563, and the document flag values are \
codes too, so never guess what any of them mean. About 88% of historical \
invoices are Category_1. "Training rows" in tool results are the historical invoices the model \
learned from; call them past invoices.

Call get_prediction_explanation first, then find_similar_training_rows. The explanation's \
contributions are exact log-odds of the predicted category versus the runner-up (positive \
supports the prediction; the class base rate is the part that comes from how common each \
category is). Call lookup_training_evidence or what_if only when they add useful evidence. Quote only numbers \
that appear in tool results; never estimate or round differently.

Answer in Markdown in exactly this layout, at most 160 words, one line per bullet, no nested \
bullets, no statements about tools you did not use:
**Summary:** Two or three sentences in everyday language, no jargon: the suggested category, how \
sure the model is (in words and as a percentage), the main reason in terms of past invoices, and \
whether a person should check it.

**Model facts**
- Prediction: category and probability, runner-up and probability.
- Drivers: the two or three inputs with the largest contributions, each with its signed log-odds \
value, plus the class base rate.
- History: the past-invoice counts behind those drivers.
- Similar invoices: how many of the most similar past invoices had the predicted category.

**What to check:** Whether review is required and what exactly to verify. Describe warnings in \
plain words, never as codes such as UNSEEN_VENDOR."""


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


@functools.cache
def _use_system_certificates() -> None:
    """Trust the operating system's certificate store once (corporate TLS proxies)."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        logger.debug("truststore not installed; using bundled certificates")


def _client(connection: ClaudeConnection, params: AgentParams) -> Any:
    _use_system_certificates()
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
    params = service.settings.agent
    answer: AgentAnswer | None = None
    reason: str | None = None
    if use_llm:
        if client is None:
            connection = resolve_connection(service.settings, override_key)
            client = _client(connection, params) if connection is not None else None
        if client is None:
            reason = "No API key or relay configured."
        else:
            answer, reason = _claude_answer(client, params, context)
    if answer is None:
        answer = offline_answer(context)
        answer.fallback_reason = reason
    answer.seconds = time.perf_counter() - started
    return answer


def _claude_answer(
    client: Any, params: AgentParams, context: ToolContext
) -> tuple[AgentAnswer | None, str | None]:
    """Return Claude's verified answer, or None with the reason it could not be used."""
    try:
        return ClaudeEngine(client, params).run(context), None
    except AgentFallbackError as failure:
        return None, failure.message
    except Exception as error:  # boundary: any client or network failure falls back
        logger.debug("Claude engine failed", exc_info=True)
        return None, _failure_reason(error)


def _failure_reason(error: Exception) -> str:
    """Name network and API errors as unreachable; anything else as an engine failure."""
    name = type(error).__name__
    from_client = type(error).__module__.split(".")[0] in ("anthropic", "httpx")
    if from_client or isinstance(error, OSError | TimeoutError):
        return f"Claude was unreachable ({name})."
    return f"The Claude engine failed ({name})."


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
        deadline = time.perf_counter() + self.params.deadline_seconds
        for _ in range(self.params.max_turns):
            if time.perf_counter() > deadline:
                raise AgentFallbackError("Claude took longer than the time limit.")
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
        arguments = use.input if isinstance(use.input, dict) else {}
        try:
            if not isinstance(use.input, dict):
                raise ToolError("Tool input must be a JSON object.")
            output = run_tool(context, use.name, arguments)
            calls.append({"tool": use.name, "input": arguments, "output": output})
            return {
                "type": "tool_result",
                "tool_use_id": use.id,
                "content": tool_result_text(output),
            }
        except (ToolError, LedgerClfError) as error:
            message = error.message
        except Exception:  # boundary: a local tool defect must not end the conversation
            logger.debug("Tool %s failed", use.name, exc_info=True)
            message = "The tool failed on this input. Answer from the other tool results."
        calls.append({"tool": use.name, "input": arguments, "error": message})
        return {
            "type": "tool_result",
            "tool_use_id": use.id,
            "content": f"Error: {message}",
            "is_error": True,
        }

    def _verified(self, text: str, calls: list[dict[str, Any]]) -> AgentAnswer:
        if not text:
            raise AgentFallbackError("Claude returned no text.")
        unsupported = ungrounded_numbers(
            text, calls, self.params.percent_tolerance, self.params.relative_tolerance
        )
        if unsupported:
            raise AgentFallbackError(
                "Claude cited numbers not found in the tool results: "
                + ", ".join(unsupported[:5])
                + "."
            )
        return AgentAnswer(text, ENGINE_CLAUDE, self.params.model, calls)


_NUMBER = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(\s*%)?")
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
        walk(call.get("output", {}))  # outputs only: a failed call's input proves nothing
    return numbers


def ungrounded_numbers(
    text: str,
    calls: list[dict[str, Any]],
    percent_tolerance: float,
    relative_tolerance: float = 0.0,
) -> list[str]:
    """Return numbers cited in ``text`` that no tool result supports.

    Percentages may match a tool probability or share (0 to 1) within the
    tolerance in percentage points; other numbers must match a tool number
    after rounding to the cited precision, or within ``relative_tolerance``
    of it. Labels, column names, tokens, and row numbers are ignored.
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
            tolerance = max(0.5 * 10 ** (-decimals), relative_tolerance * abs(value)) + 1e-9
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
    extra: list[str] = []
    try:
        similar = run_tool(context, "find_similar_training_rows", {})
        calls.append({"tool": "find_similar_training_rows", "input": {}, "output": similar})
        extra.append(_similar_sentence(similar, explanation["prediction"]["label"]))
    except LedgerClfError as error:
        calls.append({"tool": "find_similar_training_rows", "input": {}, "error": error.message})
    text = render_fallback_text(context.payload, extra)
    return AgentAnswer(text, ENGINE_OFFLINE, None, calls)


def _similar_sentence(similar: dict[str, Any], label: str) -> str:
    rows = similar["rows"]
    matching = sum(r["label"] == label for r in rows)
    best = rows[0]["similarity"] if rows else 0.0
    return (
        f"- Similar invoices: {matching} of the {len(rows)} most similar past invoices "
        f"(best similarity {best:.2f}) were {label}."
    )
