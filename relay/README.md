# Agent relay

A small Vercel function that forwards the explanation agent's requests to Claude. The API key
lives only in this deployment's environment variables, so evaluators need no key of their own.

Safeguards: a shared token header, a fixed model (`claude-haiku-5-5`), an output cap of 4,000
tokens, the agent's four tools only, a request size limit, and a per-instance rate limit. Spend is
also capped on the Anthropic side by the workspace limit.

Environment variables (set in Vercel, never committed): `ANTHROPIC_API_KEY`, optional
`ANTHROPIC_WORKSPACE_ID` (needed for user-scoped keys), and `RELAY_TOKEN`.

Deploy: `vercel deploy --prod` from this folder.
