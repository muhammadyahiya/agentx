"""AgentX — provider-agnostic agentic framework + project scaffolder.

Quick start (library):

    from agentx import get_chat_model
    llm = get_chat_model("openai", "gpt-4o-mini")
    print(llm.invoke("Hello").content)

Quick start (scaffolder):

    $ agentx new            # interactive wizard → generates a uv project

Public API is intentionally small and stable; capability modules (rag, memory,
tools, skills, frameworks) are imported lazily so installing one provider extra
is enough to get started.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .providers import (  # noqa: E402
    ProviderSpec,
    get_chat_model,
    get_crewai_llm,
    list_providers,
)

__all__ = [
    "__version__",
    "ProviderSpec",
    "get_chat_model",
    "get_crewai_llm",
    "list_providers",
]
