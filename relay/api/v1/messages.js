// Relay for the ledger classifier's explanation agent.
//
// The local app calls this endpoint instead of api.anthropic.com, so graders need no API
// key and the author's key never leaves Vercel. The relay only accepts requests
// shaped like the agent's: a fixed model, a capped output length, exactly the agent's four
// tools, and the agent's own system prompt (checked by hash). Any other field is dropped.
import { createHash } from "node:crypto";

const UPSTREAM = "https://api.anthropic.com/v1/messages";
const ALLOWED_MODEL = "claude-haiku-5-5";
const MAX_OUTPUT_TOKENS = 4000;
const MAX_BODY_BYTES = 200_000;
const MAX_MESSAGES = 16;
const ALLOWED_TOOLS = new Set([
  "get_prediction_explanation",
  "lookup_training_evidence",
  "find_similar_training_rows",
  "what_if",
]);
// sha256 of ledgerclf.agent.SYSTEM_PROMPT; tests/integration/test_agent_and_server.py keeps it in sync.
const SYSTEM_PROMPT_SHA256 = "b3f27101d827a60f8c5604222bf2780230e7e6a309e4b4fb7527bb8636ccac11";
const ALLOWED_EFFORT = "low";
const RATE_LIMIT_PER_MINUTE = 40;
const recentRequests = new Map(); // per instance, best effort

function reject(res, status, message) {
  res.status(status).json({ type: "error", error: { type: "relay_error", message } });
}

function rateLimited(ip) {
  const now = Date.now();
  const window = (recentRequests.get(ip) || []).filter((t) => now - t < 60_000);
  window.push(now);
  recentRequests.set(ip, window);
  return window.length > RATE_LIMIT_PER_MINUTE;
}

function validate(body) {
  if (!body || typeof body !== "object") return "Body must be a JSON object.";
  if (JSON.stringify(body).length > MAX_BODY_BYTES) return "Request too large.";
  if (!Array.isArray(body.messages) || body.messages.length > MAX_MESSAGES) {
    return "Too many messages.";
  }
  const names = Array.isArray(body.tools) ? body.tools.map((tool) => tool && tool.name) : [];
  if (names.length !== ALLOWED_TOOLS.size || names.some((name) => !ALLOWED_TOOLS.has(name))) {
    return "Requests must carry exactly the agent's tools.";
  }
  if (typeof body.system !== "string") return "Missing system prompt.";
  const hash = createHash("sha256").update(body.system, "utf8").digest("hex");
  if (hash !== SYSTEM_PROMPT_SHA256) return "Unknown system prompt.";
  return null;
}

export default async function handler(req, res) {
  if (req.method !== "POST") return reject(res, 405, "Use POST.");
  const expected = process.env.RELAY_TOKEN || "";
  if (!expected) return reject(res, 503, "Relay is not configured.");
  if (req.headers["x-relay-token"] !== expected) {
    return reject(res, 401, "Invalid relay token.");
  }
  const ip = String(req.headers["x-forwarded-for"] || "unknown").split(",")[0].trim();
  if (rateLimited(ip)) return reject(res, 429, "Too many requests; wait a minute.");
  const problem = validate(req.body);
  if (problem) return reject(res, 400, problem);

  const body = {
    model: ALLOWED_MODEL,
    max_tokens: Math.min(Number(req.body.max_tokens) || MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS),
    system: req.body.system,
    tools: req.body.tools,
    messages: req.body.messages,
    output_config: { effort: ALLOWED_EFFORT },
  };
  const headers = {
    "content-type": "application/json",
    "x-api-key": process.env.ANTHROPIC_API_KEY || "",
    "anthropic-version": req.headers["anthropic-version"] || "2023-06-01",
  };
  if (process.env.ANTHROPIC_WORKSPACE_ID) {
    headers["anthropic-workspace-id"] = process.env.ANTHROPIC_WORKSPACE_ID;
  }
  try {
    const upstream = await fetch(UPSTREAM, { method: "POST", headers, body: JSON.stringify(body) });
    const text = await upstream.text();
    res.status(upstream.status).setHeader("content-type", "application/json").send(text);
  } catch (error) {
    reject(res, 502, "Upstream request failed.");
  }
}
