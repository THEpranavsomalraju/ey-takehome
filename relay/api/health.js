// Lets the dashboard's "Test connection" button check the relay without calling Claude.
export default function handler(req, res) {
  res.status(200).json({
    status: "ok",
    model: "claude-haiku-5-5",
    key_configured: Boolean(process.env.ANTHROPIC_API_KEY),
  });
}
