# Agent relay

A small Vercel function that forwards the dashboard's AI requests to Claude, so graders need no API key. The key lives only in this deployment's environment variables.

It accepts only two kinds of request: the row agent (its four tools and its system prompt) and the chart explainer (no tools and its system prompt). Both prompts are checked by hash. The model (`claude-haiku-5-5`) and the output length are fixed, unknown fields are dropped, and a shared token, a size limit, and a rate limit apply. Spend is also capped in the Anthropic workspace.

Environment variables (set in Vercel, never committed): `ANTHROPIC_API_KEY`, `RELAY_TOKEN`, and optionally `ANTHROPIC_WORKSPACE_ID`.

Deploy with `vercel deploy --prod` from this folder.
