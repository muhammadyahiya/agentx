#!/usr/bin/env python
"""AgentX-Kit — MCP connector demo (the same path Claude / Copilot / Codex use).

Spawns `agentx mcp` over stdio, performs a real MCP handshake, lists tools, and
calls them to scaffold a complete project from a one-line problem statement —
exactly what an MCP client (Claude) does. No API keys required (scaffolding is
template generation).

    pip install "agentx-kit[connector]"
    python examples/demo_mcp.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile


def _extract(result) -> dict:
    """Pull a dict out of an MCP CallToolResult (structured or text content)."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured.get("result", structured)
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}
    return {}


async def main() -> int:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("Install the connector extra first:  pip install 'agentx-kit[connector]'")
        return 1

    # Launch the same server Claude would (use this interpreter for portability).
    params = StdioServerParameters(command=sys.executable, args=["-m", "agentx.cli", "mcp"])

    print("▶ connecting to `agentx mcp` over stdio …")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("✓ connected. tools:", [t.name for t in tools.tools])

            problem = "Build a customer-support agent that answers from our product docs and serves a REST API."
            print(f"\n▶ recommend_project('{problem[:50]}…')")
            rec = _extract(await session.call_tool("recommend_project", {"problem_statement": problem}))
            print(f"  framework={rec.get('framework')}  features={rec.get('features')}")

            out_dir = tempfile.mkdtemp(prefix="agentx-mcp-demo-")
            print("\n▶ create_agent_project(…)")
            res = _extract(await session.call_tool(
                "create_agent_project",
                {"problem_statement": problem, "output_dir": out_dir + "/support-bot"},
            ))
            if not res.get("ok"):
                print("  ✗ failed:", res.get("error"))
                return 1
            print(f"  ✓ generated {len(res['file_tree'])} files at {res['target_dir']}")
            print("    features:", res["features"])
            print("    run:", res["next_steps"][-1])

    print("\n✅ MCP connector works — Claude/Copilot/Codex can drive this exact flow.")
    print("   Add it to Claude:  claude mcp add agentx-kit -- agentx mcp")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
