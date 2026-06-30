# AgentX-Kit demos

Two runnable demos to confirm your setup — **no API keys required** (scaffolding
and insights are offline; LLM calls are optional).

## 1. Local setup test
Verifies the install, lists providers, scaffolds a demo project, and exercises
prompt insights + the response cache.

```bash
pip install "agentx-kit[all]"
bash examples/demo_local.sh
```

## 2. MCP connector test (the Claude / Copilot / Codex path)
Spawns `agentx mcp` over stdio, does a real MCP handshake, lists the tools, and
scaffolds a complete project from a one-line problem statement — exactly what an
assistant does when connected.

```bash
pip install "agentx-kit[connector]"
python examples/demo_mcp.py
```

Then wire it into Claude for real:
```bash
claude mcp add agentx-kit -- agentx mcp
```
