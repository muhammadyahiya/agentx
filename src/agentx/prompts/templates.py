"""Prompt templates and a small renderer.

Kept as plain strings (not framework-specific) so they work with both LangChain
and CrewAI. ``render_agent_system`` lets callers inject role, goal and optional
skills/RAG guidance without pulling in a templating engine at runtime.
"""
from __future__ import annotations

AGENT_SYSTEM_PROMPT = (
    "You are {role}.\n"
    "Your goal: {goal}\n"
    "Guidelines:\n"
    "- Be precise, concise, and grounded in available context/tools.\n"
    "- If you are unsure, say so rather than inventing facts.\n"
    "- Prefer using your tools over guessing.\n"
)

RAG_SYSTEM_PROMPT = (
    "Answer the user using ONLY the retrieved context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n"
)

_SKILLS_BLOCK = "\nApply these skills/standards:\n{skills}\n"
_RAG_BLOCK = "\nYou have a knowledge base; retrieve before answering when relevant.\n"


def render_agent_system(
    role: str,
    goal: str,
    skills: str | None = None,
    with_rag: bool = False,
) -> str:
    """Render an agent system prompt with optional skills + RAG hints."""
    out = AGENT_SYSTEM_PROMPT.format(role=role, goal=goal)
    if skills:
        out += _SKILLS_BLOCK.format(skills=skills)
    if with_rag:
        out += _RAG_BLOCK
    return out
