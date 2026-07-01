"""Autonomous working agent — a self-directed ReAct agent that can plan tasks,
browse the web, read/write files, execute shell commands, and fetch URLs.

Inspired by OpenClaw / AutoGPT-style autonomous loops.  Unlike a simple chat
agent this one receives a high-level *goal* and decomposes it into sub-tasks
using a plan-act-observe loop, persisting intermediate artifacts to a workspace.

Usage::

    from agentx.agents import AutonomousAgent

    agent = AutonomousAgent.create(
        goal="Research the top 5 open-source RAG frameworks and write a comparison report.",
        provider="openai",
        model="gpt-4o",
        workspace="./workspace",
    )
    result = agent.run()
    print(result.summary)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic config
# ──────────────────────────────────────────────────────────────────────────────

class AutonomousAgentConfig(BaseModel):
    """Configuration for an AutonomousAgent."""

    provider: str = "openai"
    model: str = ""
    goal: str = Field(..., description="High-level goal the agent should accomplish.")
    workspace: str = "./workspace"
    max_iterations: int = Field(default=20, description="Hard cap on plan-act-observe loops.")
    max_web_results: int = 5
    allow_shell: bool = Field(
        default=False,
        description="Allow the agent to run shell commands (sandboxed to workspace).",
    )
    temperature: float = 0.2
    verbose: bool = True

    model_config = {"extra": "allow"}


# ──────────────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    goal: str
    summary: str
    iterations: int
    artifacts: list[Path]
    success: bool
    error: str | None = None

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"{status} Goal: {self.goal}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Artifacts: {[str(a) for a in self.artifacts]}\n"
            f"  Summary: {self.summary[:300]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Built-in tools
# ──────────────────────────────────────────────────────────────────────────────

def _make_tools(workspace: Path, cfg: AutonomousAgentConfig) -> list:
    """Build the agent's tool set as LangChain tools."""
    from langchain_core.tools import tool

    ws = workspace

    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the web for up-to-date information. Returns titles + snippets + URLs."""
        from agentx.tools.builtin import web_search as _search
        return _search(query, max_results=min(max_results, cfg.max_web_results))

    @tool
    def fetch_url(url: str) -> str:
        """Fetch and return the plain-text content of a URL (max 8000 chars)."""
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "agentx-bot/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read(65536).decode("utf-8", errors="replace")
            # Strip HTML tags
            import re
            text = re.sub(r"<[^>]+>", " ", raw)
            text = re.sub(r"\s{2,}", " ", text).strip()
            return text[:8000]
        except Exception as exc:
            return f"Failed to fetch {url}: {exc}"

    @tool
    def read_file(filename: str) -> str:
        """Read a file from the workspace directory. filename is relative to workspace."""
        fp = ws / filename
        if not fp.exists():
            return f"File not found: {filename}"
        try:
            return fp.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error reading {filename}: {exc}"

    @tool
    def write_file(filename: str, content: str) -> str:
        """Write content to a file in the workspace directory."""
        fp = ws / filename
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        logger.info("Agent wrote file: %s (%d chars)", fp, len(content))
        return f"Wrote {len(content)} chars to {filename}"

    @tool
    def list_files(subdirectory: str = "") -> str:
        """List files in the workspace (or a subdirectory of it)."""
        target = ws / subdirectory if subdirectory else ws
        if not target.exists():
            return f"Directory not found: {subdirectory or 'workspace'}"
        files = [str(p.relative_to(ws)) for p in target.rglob("*") if p.is_file()]
        return "\n".join(files) if files else "(empty)"

    @tool
    def think(reasoning: str) -> str:
        """Use this tool to think out loud before acting. Reasoning is logged but not sent."""
        logger.info("Agent thinking: %s", reasoning[:300])
        return "Thinking recorded."

    tools = [web_search, fetch_url, read_file, write_file, list_files, think]

    if cfg.allow_shell:
        @tool
        def run_shell(command: str) -> str:
            """Run a shell command inside the workspace directory (DANGEROUS — only when authorized)."""
            try:
                result = subprocess.run(
                    command, shell=True, cwd=str(ws),
                    capture_output=True, text=True, timeout=30,
                )
                out = result.stdout[-4000:] if result.stdout else ""
                err = result.stderr[-2000:] if result.stderr else ""
                return f"stdout:\n{out}\nstderr:\n{err}\nreturncode: {result.returncode}"
            except subprocess.TimeoutExpired:
                return "Command timed out (30s limit)."
            except Exception as exc:
                return f"Shell error: {exc}"

        tools.append(run_shell)

    return tools


# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = textwrap.dedent("""\
    You are an autonomous AI agent. Your goal is:

    {goal}

    You have a workspace at: {workspace}

    Work methodically:
    1. Plan your approach before acting.
    2. Use tools to gather information, process it, and write artifacts.
    3. Save important results to files using write_file.
    4. When done, provide a clear final summary starting with "FINAL ANSWER:".

    Use the 'think' tool to reason before complex decisions.
    Be thorough but efficient. Avoid repeating the same search queries.
    The user cannot give you additional input — work autonomously.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

class AutonomousAgent:
    """An autonomous, goal-directed agent with web search, file I/O, and planning.

    Use ``AutonomousAgent.create()`` as the entry point.
    """

    def __init__(self, config: AutonomousAgentConfig) -> None:
        self.config = config
        self.workspace = Path(config.workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._artifacts: list[Path] = []

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        goal: str,
        provider: str = "openai",
        model: str = "",
        workspace: str = "./workspace",
        max_iterations: int = 20,
        allow_shell: bool = False,
        **kwargs: Any,
    ) -> "AutonomousAgent":
        """Create an AutonomousAgent from keyword arguments."""
        return cls(AutonomousAgentConfig(
            goal=goal, provider=provider, model=model,
            workspace=workspace, max_iterations=max_iterations,
            allow_shell=allow_shell, **kwargs,
        ))

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self) -> AgentResult:
        """Run the agent synchronously until the goal is reached or max_iterations hit."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.arun())
        finally:
            loop.close()

    async def arun(self) -> AgentResult:
        """Async run — use this from an async context."""
        from agentx.providers import get_chat_model

        cfg = self.config
        llm = get_chat_model(cfg.provider, cfg.model or None, temperature=cfg.temperature)
        tools = _make_tools(self.workspace, cfg)

        system = _SYSTEM_TEMPLATE.format(
            goal=cfg.goal,
            workspace=str(self.workspace),
        )

        try:
            from langgraph.prebuilt import create_react_agent  # type: ignore
            from langgraph.checkpoint.memory import MemorySaver  # type: ignore
            from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

            agent = create_react_agent(llm, tools, prompt=system, checkpointer=MemorySaver())

            logger.info("AutonomousAgent starting: goal=%r iterations_cap=%d", cfg.goal[:80], cfg.max_iterations)

            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=cfg.goal)]},
                config={"configurable": {"thread_id": "auto"}, "recursion_limit": cfg.max_iterations * 2},
            )

            messages = result.get("messages", [])
            final_content = ""
            for m in reversed(messages):
                c = getattr(m, "content", "")
                if c and isinstance(c, str):
                    final_content = c
                    break

            # Collect artifacts written during the run
            self._artifacts = list(self.workspace.rglob("*"))
            self._artifacts = [f for f in self._artifacts if f.is_file()]

            summary = final_content
            if "FINAL ANSWER:" in final_content:
                summary = final_content.split("FINAL ANSWER:", 1)[1].strip()

            logger.info("AutonomousAgent finished. Artifacts: %d", len(self._artifacts))
            return AgentResult(
                goal=cfg.goal,
                summary=summary,
                iterations=len([m for m in messages if getattr(m, "type", "") == "tool"]),
                artifacts=self._artifacts,
                success=True,
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("AutonomousAgent failed")
            return AgentResult(
                goal=cfg.goal,
                summary="",
                iterations=0,
                artifacts=[],
                success=False,
                error=str(exc),
            )
