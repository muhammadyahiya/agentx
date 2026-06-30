#!/usr/bin/env bash
# AgentX-Kit — local setup smoke test. No API keys required.
# Verifies the install, lists providers, scaffolds a demo project, and exercises
# the prompt-insights + cache so you know everything works end-to-end.
#
#   bash examples/demo_local.sh
set -euo pipefail

# Resolve the CLI: prefer `agentx` on PATH, else `python -m agentx.cli`.
if command -v agentx >/dev/null 2>&1; then
  AGENTX="agentx"
else
  AGENTX="python -m agentx.cli"
fi
echo "▶ using: $AGENTX"

echo; echo "==> 1) version"
$AGENTX version

echo; echo "==> 2) providers (no keys needed to list)"
$AGENTX providers

WORK="$(mktemp -d)"; trap 'echo; echo "demo project left at: $WORK"' EXIT
echo; echo "==> 3) scaffold a demo project (local Ollama provider, no key)"
( cd "$WORK" && $AGENTX new --yes --name demo-agent --provider ollama \
    --prompt "You are a helpful onboarding assistant." --no-venv )
echo "files:"
find "$WORK/demo-agent" -maxdepth 3 -type f | sed "s|$WORK/||" | sort

echo; echo "==> 4) prompt insights (offline)"
python - <<'PY'
from agentx import analyze_prompt, count_tokens
a = analyze_prompt("You are a support agent. Goal: answer in JSON. Do not invent policy.", "gpt-4o-mini")
print(f"  quality={a.quality_score}/100  tokens={a.tokens}  suggestions={len(a.suggestions)}")
PY

echo; echo "==> 5) response cache stats"
$AGENTX cache stats || true

echo; echo "✅ Local setup looks good."
echo "Next: cd $WORK/demo-agent && uv sync && uv run demo-agent"
echo "      $AGENTX dashboard           # prompt observability UI (needs [dashboard] extra)"
