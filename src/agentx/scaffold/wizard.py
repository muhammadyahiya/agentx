"""Interactive wizard — collects a ``ProjectSpec`` one option at a time.

Uses ``questionary`` for arrow-key selection. Every prompt has a sensible
default so power users can blast through with Enter.
"""
from __future__ import annotations

import questionary

from ..providers import all_specs, get_spec
from .spec import AgentSpec, ProjectSpec

_FRAMEWORKS = [
    ("LangGraph (LangChain)", "langgraph"),
    ("CrewAI", "crewai"),
]
_MEMORY = [
    ("None", "none"),
    ("Short-term (windowed buffer)", "short"),
    ("Long-term (persistent JSONL)", "long"),
    ("Both", "both"),
]


def _select(message: str, choices: list[tuple[str, str]], default_value: str) -> str:
    options = [questionary.Choice(title=label, value=value) for label, value in choices]
    default = next((o for o in options if o.value == default_value), options[0])
    return questionary.select(message, choices=options, default=default).ask()


def run_wizard(name: str | None = None) -> ProjectSpec | None:
    """Run the interactive flow; returns a ProjectSpec, or None if cancelled."""
    questionary.print("🧬  AgentX — new project\n", style="bold fg:cyan")

    name = name or questionary.text("Project name:", default="my-agent").ask()
    if not name:
        return None

    framework = _select("Agent framework:", _FRAMEWORKS, "langgraph")
    if framework is None:
        return None

    # Provider + model
    provider_choices = [
        (f"{s.label}  ·  needs: {', '.join(s.env_vars) or 'no key (local)'}", s.id)
        for s in all_specs()
    ]
    provider = _select("LLM provider:", provider_choices, "openai")
    if provider is None:
        return None
    pspec = get_spec(provider)
    model = questionary.text(
        f"Model id ({pspec.label}):", default=pspec.default_model
    ).ask() or pspec.default_model

    # Agents
    n_str = questionary.text("How many agents?", default="1").ask() or "1"
    try:
        n_agents = max(1, min(10, int(n_str)))
    except ValueError:
        n_agents = 1
    agents: list[AgentSpec] = []
    for i in range(n_agents):
        questionary.print(f"\nAgent {i + 1} of {n_agents}", style="bold")
        a_name = questionary.text("  name:", default=f"agent_{i + 1}" if n_agents > 1 else "assistant").ask()
        a_role = questionary.text("  role:", default="Helpful Assistant").ask()
        a_goal = questionary.text("  goal:", default="Help the user accomplish their task accurately.").ask()
        a_prompt = questionary.text(
            "  system prompt (optional — blank = auto from role/goal):", default="",
            multiline=True,
        ).ask()
        agents.append(AgentSpec(
            name=a_name or f"agent_{i + 1}",
            role=a_role or "Assistant",
            goal=a_goal or "Help the user.",
            system_prompt=(a_prompt or "").strip(),
        ))

    # Capabilities — one by one
    use_rag = questionary.confirm("Add a RAG module (knowledge base)?", default=False).ask()
    memory = _select("Agent memory:", _MEMORY, "none")
    use_mcp = questionary.confirm("Integrate MCP tools?", default=False).ask()
    use_skills = questionary.confirm("Add a skills registry?", default=False).ask()
    custom_prompts = questionary.confirm("Scaffold custom prompt templates (vs defaults)?", default=False).ask()

    create_venv = questionary.confirm("Create a .venv with `uv` now?", default=True).ask()
    run_sync = False
    if create_venv:
        run_sync = questionary.confirm("Install dependencies now (`uv sync`)? (needs network)", default=False).ask()

    return ProjectSpec(
        name=name,
        framework=framework,
        provider=provider,
        model=model,
        agents=agents,
        use_rag=bool(use_rag),
        memory=memory or "none",
        use_mcp=bool(use_mcp),
        use_skills=bool(use_skills),
        prompt_style="custom" if custom_prompts else "default",
        create_venv=bool(create_venv),
        run_sync=bool(run_sync),
    )
